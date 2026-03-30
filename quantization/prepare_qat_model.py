"""
prepare_qat_model.py

Purpose:
    Convert a pretrained CMUNeXt model into a QAT-ready model that simulates
    the final integer inference target.

Design choices for this project:
    - Preserve architecture exactly; do NOT reorder Conv/GELU/BN
    - Keep GELU as a real operator during QAT, but wrap it with fake-quant
      boundaries so it can later export to LUT-based integer inference
    - Keep unfused BN as an explicit operator; do NOT try to force folding
    - Reuse the same integer runtime target as PTQ:
        Conv / BN / GELU / MaxPool / Upsample / Add / Concat

What this file does:
    1. Build the float CMUNeXt model
    2. Load the pretrained checkpoint
    3. Replace selected modules with QAT wrappers
    4. Return a trainable QAT-ready model + summary

What this file does NOT do:
    - train the QAT model
    - export final integer artifacts
    - decompose execution trace
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from prepare_model import (
    build_model,
    load_checkpoint,
    count_parameters,
    build_module_name_map,
)
from fake_quant_ops import (
    FakeQuantConfig,
    QATConv2d,
    QATBatchNorm2d,
    QATGELU,
)


# ---------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------

@dataclass
class PrepareQATSummary:
    """
    Summary of QAT model preparation.

    Fields:
        checkpoint_path:
            Source pretrained float checkpoint
        model_variant:
            base / small / large
        total_parameters:
            Total number of parameters after wrapping
        num_qat_convs:
            Number of Conv2d modules replaced by QATConv2d
        num_qat_gelus:
            Number of GELU modules replaced by QATGELU
        num_qat_bns:
            Number of BatchNorm2d modules replaced by QATBatchNorm2d
        replaced_module_names:
            Names of modules that were replaced
    """
    checkpoint_path: str
    model_variant: str
    total_parameters: int
    num_qat_convs: int
    num_qat_gelus: int
    num_qat_bns: int
    replaced_module_names: List[str]


# ---------------------------------------------------------------------
# Module-selection helpers
# ---------------------------------------------------------------------

def should_wrap_conv(module: nn.Module) -> bool:
    """
    Return True if this module should become QATConv2d.

    Current policy:
        - wrap every Conv2d

    Why:
        Conv is the main arithmetic target in the final integer runtime.
    """
    return isinstance(module, nn.Conv2d)


def should_wrap_gelu(module: nn.Module) -> bool:
    """
    Return True if this module should become QATGELU.

    Current policy:
        - wrap every GELU

    Why:
        final runtime will use LUT-based GELU, so QAT should simulate
        quantization boundaries around GELU.
    """
    return isinstance(module, nn.GELU)


def should_wrap_bn(module: nn.Module) -> bool:
    """
    Return True if this module should become QATBatchNorm2d.

    Current policy:
        - wrap every BatchNorm2d still present in the float model

    Why:
        you do not want to reorder architecture, and unfused BN must remain
        explicit for the final integer path.
    """
    return isinstance(module, nn.BatchNorm2d)


# ---------------------------------------------------------------------
# Wrapper construction helpers
# ---------------------------------------------------------------------

def make_default_qat_configs() -> Tuple[FakeQuantConfig, FakeQuantConfig]:
    """
    Build default activation and weight fake-quant configs.

    Recommended first-pass choices:
        activations:
            - int8
            - affine / asymmetric
            - per-tensor
        weights:
            - int8
            - symmetric
            - per-output-channel

    Returns:
        (activation_config, weight_config)
    """
    act_config = FakeQuantConfig(
        dtype="int8",
        symmetric=False,
        per_channel=False,
        channel_axis=1,
        eps=1e-8,
    )

    # For Conv2d weights, per-output-channel often means channel axis 0
    weight_config = FakeQuantConfig(
        dtype="int8",
        symmetric=True,
        per_channel=True,
        channel_axis=0,
        eps=1e-8,
    )

    return act_config, weight_config


def make_qat_wrapper(
    module: nn.Module,
    act_config: FakeQuantConfig,
    weight_config: FakeQuantConfig,
) -> nn.Module:
    """
    Convert one float module into its QAT wrapper.

    Args:
        module:
            Original float module
        act_config:
            Activation fake-quant config
        weight_config:
            Weight fake-quant config

    Returns:
        Wrapped module

    Raises:
        TypeError if module type is unsupported
    """
    if should_wrap_conv(module):
        return QATConv2d(
            conv=copy.deepcopy(module),
            act_config=act_config,
            weight_config=weight_config,
        )

    if should_wrap_gelu(module):
        return QATGELU(
            gelu=copy.deepcopy(module),
            act_config=act_config,
        )

    if should_wrap_bn(module):
        return QATBatchNorm2d(
            bn=copy.deepcopy(module),
            act_config=act_config,
        )

    raise TypeError(f"Unsupported module type for QAT wrapping: {type(module).__name__}")


# ---------------------------------------------------------------------
# Recursive replacement
# ---------------------------------------------------------------------

def replace_modules_for_qat(
    model: nn.Module,
    act_config: FakeQuantConfig,
    weight_config: FakeQuantConfig,
    prefix: str = "",
    replaced_module_names: List[str] | None = None,
) -> Tuple[nn.Module, List[str]]:
    """
    Recursively replace supported modules with QAT wrappers.

    Args:
        model:
            Float model or submodule
        act_config:
            Activation fake-quant config
        weight_config:
            Weight fake-quant config
        prefix:
            Current module-name prefix during recursion
        replaced_module_names:
            Running list of replaced module names

    Returns:
        (model, replaced_module_names)

    Important:
        - This function mutates the model in place
        - Architecture ordering is preserved
        - Only modules are wrapped; functional graph ops remain unchanged
    """
    if replaced_module_names is None:
        replaced_module_names = []

    # list(...) is important because we may replace children while iterating
    for child_name, child_module in list(model.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if (
            should_wrap_conv(child_module)
            or should_wrap_gelu(child_module)
            or should_wrap_bn(child_module)
        ):
            wrapped = make_qat_wrapper(
                module=child_module,
                act_config=act_config,
                weight_config=weight_config,
            )
            setattr(model, child_name, wrapped)
            replaced_module_names.append(full_name)
        else:
            replace_modules_for_qat(
                model=child_module,
                act_config=act_config,
                weight_config=weight_config,
                prefix=full_name,
                replaced_module_names=replaced_module_names,
            )

    return model, replaced_module_names


# ---------------------------------------------------------------------
# Preparation entry point
# ---------------------------------------------------------------------

def prepare_model_for_qat(
    checkpoint_path: str,
    variant: str,
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[nn.Module, PrepareQATSummary]:
    """
    End-to-end QAT preparation.

    Steps:
        1. Build float CMUNeXt
        2. Load pretrained checkpoint
        3. Switch to train mode later during QAT training, but prepare weights first
        4. Replace supported modules with QAT wrappers
        5. Return wrapped model and preparation summary

    Args:
        checkpoint_path:
            Path to pretrained float checkpoint
        variant:
            Model variant: base / small / large
        map_location:
            Where to load checkpoint
        strict:
            strict flag for state_dict loading

    Returns:
        (qat_model, summary)
    """
    # 1. Build float model
    model = build_model(variant=variant)

    # 2. Load pretrained checkpoint
    model = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        map_location=map_location,
        strict=strict,
    )

    # 3. Build QAT configs
    act_config, weight_config = make_default_qat_configs()

    # 4. Replace modules with QAT wrappers
    model, replaced_module_names = replace_modules_for_qat(
        model=model,
        act_config=act_config,
        weight_config=weight_config,
        prefix="",
        replaced_module_names=[],
    )

    # 5. Keep model trainable for QAT fine-tuning
    model.train()

    # 6. Build summary
    num_qat_convs = 0
    num_qat_gelus = 0
    num_qat_bns = 0

    for module in model.modules():
        if isinstance(module, QATConv2d):
            num_qat_convs += 1
        elif isinstance(module, QATGELU):
            num_qat_gelus += 1
        elif isinstance(module, QATBatchNorm2d):
            num_qat_bns += 1

    summary = PrepareQATSummary(
        checkpoint_path=checkpoint_path,
        model_variant=variant,
        total_parameters=count_parameters(model),
        num_qat_convs=num_qat_convs,
        num_qat_gelus=num_qat_gelus,
        num_qat_bns=num_qat_bns,
        replaced_module_names=replaced_module_names,
    )

    return model, summary


# ---------------------------------------------------------------------
# Optional debug / smoke test helpers
# ---------------------------------------------------------------------

def print_qat_summary(summary: PrepareQATSummary, max_names: int = 20) -> None:
    """
    Print a compact QAT preparation summary.
    """
    print("\n=== QAT Preparation Summary ===")
    print(f"Checkpoint path: {summary.checkpoint_path}")
    print(f"Model variant:   {summary.model_variant}")
    print(f"Total params:    {summary.total_parameters}")
    print(f"QAT Conv2d:      {summary.num_qat_convs}")
    print(f"QAT GELU:        {summary.num_qat_gelus}")
    print(f"QAT BatchNorm2d: {summary.num_qat_bns}")

    if len(summary.replaced_module_names) > 0:
        print("\nFirst replaced modules:")
        for name in summary.replaced_module_names[:max_names]:
            print(f"  {name}")

        if len(summary.replaced_module_names) > max_names:
            print(f"  ... and {len(summary.replaced_module_names) - max_names} more")


def save_qat_summary(summary: PrepareQATSummary, save_path: str) -> None:
    """
    Save QAT preparation summary to JSON.
    """
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)


def smoke_test_forward(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    device: str = "cpu",
) -> None:
    """
    Run one dummy forward pass to ensure the QAT-wrapped model executes.

    Important:
        This does not validate numerical correctness, only basic graph integrity.
    """
    model = model.to(device)
    x = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        y = model(x)

    if not isinstance(y, torch.Tensor):
        raise RuntimeError("QAT model forward did not return a tensor")

    print("\n=== QAT Smoke Test ===")
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print("Forward pass successful.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    CLI args for QAT preparation smoke test.
    """
    parser = argparse.ArgumentParser(description="Prepare CMUNeXt model for QAT")

    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to pretrained float .pth checkpoint",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "small", "large"],
        help="CMUNeXt model variant",
    )
    parser.add_argument(
        "--map-location",
        type=str,
        default="cpu",
        help="Checkpoint load device mapping",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict checkpoint loading",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default="qat_prepare_summary.json",
        help="Path to save QAT preparation summary",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one dummy forward pass after QAT wrapping",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[prepare_qat] Loading checkpoint: {args.checkpoint_path}")
    print(f"[prepare_qat] Variant: {args.variant}")

    model, summary = prepare_model_for_qat(
        checkpoint_path=args.checkpoint_path,
        variant=args.variant,
        map_location=args.map_location,
        strict=args.strict,
    )

    print_qat_summary(summary)
    save_qat_summary(summary, args.save_summary)
    print(f"[prepare_qat] Summary saved to: {args.save_summary}")

    if args.smoke_test:
        smoke_test_forward(
            model=model,
            input_shape=(1, 3, 256, 256),
            device="cpu",
        )


if __name__ == "__main__":
    main()