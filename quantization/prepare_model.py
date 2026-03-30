"""
prepare_model.py

Purpose:
    Prepare a trained CMUNeXt model for post-training quantization.

Main responsibilities:
    1. Load the trained float32 .pth checkpoint.
    2. Instantiate the correct CMUNeXt architecture from model_def.py.
    3. Put the model into deterministic eval mode.
    4. Fold BatchNorm2d into the preceding Conv2d wherever valid.
    5. Assign stable names to important modules for later calibration/tracing.
    6. Optionally run a sanity check comparing original vs folded outputs.

Why this file matters:
    For PTQ integer inference, you want to quantize the *inference graph*,
    not the original training graph. In your model, many blocks contain
    Conv2d + BatchNorm2d, and these should be folded first so later files
    quantize the correct effective weights and biases.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from network.CMUNeXt import cmunext, cmunext_l, cmunext_s
import argparse


@dataclass
class FoldRecord:
    """
    Stores metadata for one Conv2d + BatchNorm2d folding event.

    Useful for:
        - debugging
        - reporting which layers were folded
        - exporting fold info for later trace/debug
    """

    conv_name: str
    bn_name: str
    conv_out_channels: int
    conv_in_channels: int
    kernel_size: Tuple[int, int]
    groups: int
    eps: float


@dataclass
class PrepareSummary:
    """
    High-level summary of model preparation.

    This can later be written to JSON so you know exactly what was done.
    """

    checkpoint_path: str
    model_variant: str
    total_parameters: int
    folded_pairs: List[Dict[str, Any]]


def build_model(variant: str = "base") -> nn.Module:
    if variant == "base":
        return cmunext()

    if variant == "small":
        return cmunext_s()

    if variant == "large":
        return cmunext_l()
    raise ValueError(f"Unsupported model variant: {variant}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    map_location: str = "cpu",
    strict: bool = True,
) -> nn.Module:
    """
    Load a trained .pth checkpoint into an already-instantiated model.

    Args:
        model:
            The architecture instance created by build_model().
        checkpoint_path:
            Path to the trained .pth file.
        map_location:
            Usually "cpu" for PTQ preparation.
        strict:
            Whether state_dict loading should be exact.

    Returns:
        The same model object with loaded weights.

    Behavior:
        - Supports either:
            1. raw state_dict checkpoints, or
            2. dict checkpoints containing a 'state_dict' key.
        - Removes common wrappers like 'module.' if model was trained with DataParallel.

    Raises:
        FileNotFoundError:
            If checkpoint_path does not exist.
        RuntimeError:
            If the checkpoint shape/names do not match the architecture
            and strict=True.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Extract state_dict if checkpoint is a dict with 'state_dict' key
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix from DataParallel-trained models
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=strict)
    return model
    


def prepare_eval_model(model: nn.Module) -> nn.Module:
    """
    Put the model into a stable inference-ready state.

    Responsibilities:
        - model.eval()
        - disable gradients on all parameters
        - optionally set deterministic flags for reproducibility

    Args:
        model:
            The loaded float model.

    Returns:
        The same model in eval mode.

    Why:
        BatchNorm folding uses running_mean and running_var from inference mode,
        not batch statistics from training mode.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return model


def fold_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fold a BatchNorm2d layer into the preceding Conv2d layer.

    Args:
        conv:
            The convolution layer that comes immediately before bn.
        bn:
            The batch normalization layer to absorb into conv.

    Returns:
        A NEW nn.Conv2d whose weight and bias already include the BN effect.

    Mathematical idea:
        For inference, BatchNorm is an affine transform:
            y = gamma * (x - mean) / sqrt(var + eps) + beta

        If x is the output of conv, this BN transform can be absorbed into
        conv's weights and biases, producing a single equivalent Conv2d layer.

    Requirements:
        - bn.num_features must match conv.out_channels
        - conv may or may not have an original bias
        - grouped/depthwise conv is supported as long as output channels match BN

    Important:
        Return a *new* Conv2d module instead of mutating the original in place.
        That is safer and easier to debug.
    """
    assert bn.num_features == conv.out_channels, \
        f"Conv out_channels ({conv.out_channels}) must match BN num_features ({bn.num_features})"
    
    # Create a new Conv2d with the same structure
    folded_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,  # Folded conv always needs bias
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    
    # Get BN parameters
    bn_weight = bn.weight
    bn_bias = bn.bias
    bn_running_mean = bn.running_mean
    bn_running_var = bn.running_var
    bn_eps = bn.eps
    
    # Compute BN scale: gamma / sqrt(var + eps)
    bn_scale = bn_weight / torch.sqrt(bn_running_var + bn_eps)
    
    # Fold into Conv weights: w_new = scale * w_old
    folded_conv.weight.data = conv.weight.data * bn_scale.reshape(-1, 1, 1, 1)
    
    # Fold into Conv bias: b_new = scale * (b_old - mean) + beta
    if conv.bias is not None:
        conv_bias = conv.bias
    else:
        conv_bias = torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
    
    folded_conv.bias.data = bn_scale * (conv_bias - bn_running_mean) + bn_bias
    
    return folded_conv


def is_conv_bn_pattern(modules: List[nn.Module], idx: int) -> bool:
    """
    Check whether modules[idx] and modules[idx + 1] form a Conv2d -> BatchNorm2d pair.

    Args:
        modules:
            A list of child modules from a Sequential container.
        idx:
            Candidate starting index.

    Returns:
        True if modules[idx] is Conv2d and modules[idx + 1] is BatchNorm2d.

    Why:
        Many parts of CMUNeXt store layers inside nn.Sequential blocks, so
        folding is easiest if we iterate through sequential children and detect
        foldable adjacent pairs.
    """
    if idx + 1 >= len(modules):
        return False
    return isinstance(modules[idx], nn.Conv2d) and isinstance(modules[idx + 1], nn.BatchNorm2d)


def fold_sequential_block(
    seq: nn.Sequential,
    prefix: str = "",
    fold_records: Optional[List[FoldRecord]] = None,
) -> nn.Sequential:
    """
    Return a new Sequential in which every eligible Conv2d -> BatchNorm2d pair
    has been replaced by a single folded Conv2d.

    Args:
        seq:
            The input Sequential block.
        prefix:
            A name prefix used only for readable fold record names.
        fold_records:
            Optional list that gets appended with metadata about each fold.

    Returns:
        A new nn.Sequential with BN layers removed where folding was possible.

    Example:
        [Conv2d, BatchNorm2d, ReLU]  ->  [FoldedConv2d, ReLU]

    Notes:
        - Only fold adjacent Conv2d -> BatchNorm2d pairs.
        - Leave GELU, ReLU, Upsample, MaxPool, Residual wrappers, etc. untouched.
        - Preserve order of all non-folded layers.
    """
    if fold_records is None:
        fold_records = []
    
    modules = list(seq.children())
    new_modules = []
    i = 0
    
    while i < len(modules):
        if is_conv_bn_pattern(modules, i):
            # Found a Conv2d -> BatchNorm2d pair
            conv = modules[i]
            bn = modules[i + 1]
            
            # Fold the pair
            folded_conv = fold_conv_bn_pair(conv, bn)
            new_modules.append(folded_conv)
            
            # Record the fold
            conv_name = f"{prefix}.{i}" if prefix else str(i)
            bn_name = f"{prefix}.{i + 1}" if prefix else str(i + 1)
            
            fold_record = FoldRecord(
                conv_name=conv_name,
                bn_name=bn_name,
                conv_out_channels=conv.out_channels,
                conv_in_channels=conv.in_channels,
                kernel_size=conv.kernel_size,
                groups=conv.groups,
                eps=bn.eps,
            )
            fold_records.append(fold_record)
            
            # Skip both the conv and bn
            i += 2
        else:
            # Keep non-foldable modules as-is
            new_modules.append(modules[i])
            i += 1
    
    return nn.Sequential(*new_modules)


def fold_batchnorms_in_module(
    module: nn.Module,
    prefix: str = "",
    fold_records: Optional[List[FoldRecord]] = None,
) -> nn.Module:
    """
    Recursively traverse a model and fold all eligible Conv2d + BatchNorm2d pairs.

    Args:
        module:
            Any module in the model tree.
        prefix:
            Hierarchical name prefix for readable debugging.
        fold_records:
            Optional list collecting fold metadata.

    Returns:
        A new module with the same structure, except foldable Conv+BN pairs
        inside Sequential blocks are replaced by folded Conv layers.

    Strategy:
        - If module is nn.Sequential, process with fold_sequential_block().
        - Otherwise deep-copy the module and recursively replace its child modules.
        - Do not try to algebraically fold across arbitrary wrappers unless the
          layers are actually adjacent in execution order and structurally safe.

    Important:
        This function should preserve the overall CMUNeXt architecture.
        It should not alter Residual connections, concat behavior, pooling, or GELU.
    """
    if fold_records is None:
        fold_records = []
    
    # If this is a Sequential, use the specialized folding function
    if isinstance(module, nn.Sequential):
        return fold_sequential_block(module, prefix=prefix, fold_records=fold_records)
    
    # Deep-copy the module to avoid mutating the original
    new_module = copy.deepcopy(module)
    
    # Recursively process child modules
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        folded_child = fold_batchnorms_in_module(
            child,
            prefix=child_prefix,
            fold_records=fold_records,
        )
        setattr(new_module, name, folded_child)
    
    return new_module


def build_module_name_map(model: nn.Module) -> Dict[int, str]:
    """
    Build a mapping from Python object id(module) to stable hierarchical module name.

    Args:
        model:
            The prepared model.

    Returns:
        A dict mapping object ids to names like:
            'stem.conv.0'
            'encoder3.block.0.0.fn.0'
            'Up_conv4.conv.3'

    Why:
        Later files (calibrate.py, trace.py, validate.py) will need stable names
        for hooks, saved tensors, and debug messages.
    """
    name_map: Dict[int, str] = {}
    
    def register_modules(module: nn.Module, prefix: str = "") -> None:
        """Recursively register all named modules with their hierarchical names."""
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            name_map[id(child)] = child_name
            register_modules(child, prefix=child_name)
    
    register_modules(model)
    return name_map


def count_parameters(model: nn.Module) -> int:
    """
    Count total parameters in the model.

    Args:
        model:
            Any PyTorch model.

    Returns:
        Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def compare_model_outputs(
    original_model: nn.Module,
    folded_model: nn.Module,
    input_tensor: torch.Tensor,
) -> Dict[str, float]:
    """
    Compare outputs of the original float model and the BN-folded float model.

    Args:
        original_model:
            The loaded model before BN folding.
        folded_model:
            The model after BN folding.
        input_tensor:
            Example input batch, e.g. shape [1, 3, H, W].

    Returns:
        A dict containing useful error metrics, for example:
            {
                "max_abs_diff": ...,
                "mean_abs_diff": ...,
                "rmse": ...
            }

    Why:
        Before doing PTQ, verify that BN folding preserved model behavior.
        Differences should be very small if folding was implemented correctly.
    """
    original_output = original_model(input_tensor)
    folded_output = folded_model(input_tensor)
    
    # Compute element-wise absolute difference
    abs_diff = torch.abs(original_output - folded_output)
    
    # Calculate metrics
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    rmse = torch.sqrt(torch.mean(abs_diff ** 2)).item()
    
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rmse": rmse,
    }


def prepare_model_for_ptq(
    checkpoint_path: str,
    variant: str = "base",
    map_location: str = "cpu",
    strict: bool = True,
    save_summary_path: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, PrepareSummary]:
    """
    End-to-end model preparation entry point for PTQ.

    Pipeline:
        1. Build architecture.
        2. Load checkpoint.
        3. Put model in eval mode.
        4. Deep-copy original model.
        5. Fold BatchNorm into Conv where possible.
        6. Collect fold summary.

    Args:
        checkpoint_path:
            Path to the trained .pth file.
        variant:
            Model variant name.
        input_channels, num_classes, dims, depths, kernels:
            Architecture settings.
        map_location:
            Usually 'cpu'.
        strict:
            Strict checkpoint loading.
        save_summary_path:
            Optional JSON path for saving a preparation summary.

    Returns:
        folded_model:
            The float model after BN folding.
        summary:
            Metadata describing what was loaded and folded.

    Important:
        This function does NOT quantize the model.
        It only prepares the float inference graph that will be used by
        calibrate.py and quantize.py later.
    """
    # Step 1: Build the architecture
    model = build_model(variant=variant)
    
    # Step 2: Load checkpoint
    model = load_checkpoint(
        model,
        checkpoint_path=checkpoint_path,
        map_location=map_location,
        strict=strict,
    )
    
    # Step 3: Put into eval mode
    model = prepare_eval_model(model)
    
    # Step 4: Deep-copy original for later comparison
    original_model = copy.deepcopy(model)
    
    # Step 5: Fold BatchNorm layers
    fold_records: List[FoldRecord] = []
    folded_model = fold_batchnorms_in_module(
        model,
        prefix="",
        fold_records=fold_records,
    )
    
    # Step 6: Collect summary
    total_params = count_parameters(folded_model)
    folded_pairs = [asdict(record) for record in fold_records]
    
    summary = PrepareSummary(
        checkpoint_path=checkpoint_path,
        model_variant=variant,
        total_parameters=total_params,
        folded_pairs=folded_pairs,
    )
    
    # Optionally save summary to JSON
    if save_summary_path is not None:
        resolved_summary_path = os.path.abspath(save_summary_path)
        summary_dir = os.path.dirname(resolved_summary_path)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(resolved_summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)
    
    return original_model, folded_model, summary


def main() -> None:
    """
    Minimal standalone test for prepare_model.py.

    Suggested behavior:
        - parse a checkpoint path
        - prepare the folded model
        - create a dummy input tensor
        - compare original vs folded outputs
        - print summary and error stats

    This helps verify prepare_model.py before you touch calibrate.py.
    """
    
    parser = argparse.ArgumentParser(description="Prepare CMUNeXt model for PTQ")
    parser.add_argument("checkpoint_path", type=str, help="Path to trained .pth checkpoint")
    parser.add_argument("--variant", type=str, default="base", choices=["base", "small", "large"],
                        help="Model variant")
    parser.add_argument("--map-location", type=str, default="cpu", help="Device for loading")
    parser.add_argument("--save-summary", type=str, default=None, help="Path to save JSON summary")
    parser.add_argument("--input-height", type=int, default=256, help="Input image height")
    parser.add_argument("--input-width", type=int, default=256, help="Input image width")
    
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint_path}")
    original_model, folded_model, summary = prepare_model_for_ptq(
        checkpoint_path=args.checkpoint_path,
        variant=args.variant,
        map_location=args.map_location,
        save_summary_path=args.save_summary,
    )
    dummy_input = torch.randn(1, 3, args.input_height, args.input_width, device=args.map_location)
    metrics = compare_model_outputs(original_model, folded_model, dummy_input)

    print("\n=== Folding Sanity Check ===")
    print(f"max_abs_diff:  {metrics['max_abs_diff']:.8e}")
    print(f"mean_abs_diff: {metrics['mean_abs_diff']:.8e}")
    print(f"rmse:          {metrics['rmse']:.8e}")

    print(f"\n=== Preparation Summary ===")
    print(f"Model variant: {summary.model_variant}")
    print(f"Total parameters: {summary.total_parameters:,}")
    print(f"Folded Conv+BN pairs: {len(summary.folded_pairs)}")
    
    if summary.folded_pairs:
        print(f"\nFirst few folded layers:")
        for i, pair in enumerate(summary.folded_pairs[:3]):
            print(f"  {pair['conv_name']} + {pair['bn_name']}")
    
    print(f"\nModel prepared successfully!")
    if args.save_summary:
        print(f"Summary saved to: {args.save_summary}")


if __name__ == "__main__":
    main()
