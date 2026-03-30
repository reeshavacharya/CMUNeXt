"""
calibrate.py

Purpose:
    Run a prepared CMUNeXt model on a representative calibration subset of BUSI
    and collect activation statistics needed for post-training quantization.

Main responsibilities:
    1. Load the BN-prepared float model from prepare_model.py.
    2. Build a calibration dataset from BUSI images (and optionally masks).
    3. Register forward hooks on selected modules.
    4. Run inference on calibration samples in eval mode.
    5. Track activation min/max for each hooked module.
    6. Save collected statistics to JSON for use in quantize.py.

Important notes:
    - This file does NOT quantize anything yet.
    - This file does NOT modify model weights.
    - This file only collects representative tensor ranges.
    - For first implementation, plain global min/max per activation is enough.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize

# Import from prepare_model.py in the same folder.
# These functions are the natural entry points from the previous stage.
from prepare_model import (
    prepare_model_for_ptq,
    build_module_name_map,
)


@dataclass
class ActivationStats:
    """
    Running activation statistics for one module output.

    Fields:
        name:
            Stable module name, e.g. 'stem.conv.0'
        op_type:
            Module class name, e.g. 'Conv2d', 'GELU', 'MaxPool2d'
        min_val:
            Global minimum observed over all calibration samples
        max_val:
            Global maximum observed over all calibration samples
        num_updates:
            How many batches / outputs contributed to this record
        shape:
            Most recent observed output shape (mainly for debugging)
    """

    name: str
    op_type: str
    min_val: float
    max_val: float
    num_updates: int
    shape: List[int]


class BUSICalibrationDataset(Dataset):
    """
    Calibration dataset for BUSI, aligned with the shown MedicalDataSets logic.

    Expected directory layout:
        ../data/busi/calibration/
            images/
                xxx.png
                yyy.png
                ...
            masks/
                0/
                    xxx.png
                    yyy.png
                    ...

    Notes:
        - For PTQ calibration, only images are strictly required.
        - Masks are optional and mainly useful for debugging.
        - This class matches the visible MedicalDataSets preprocessing:
            * cv2.imread() image loading
            * optional resize transform
            * image.astype(float32) / 255
            * HWC -> CHW
            * same style for masks
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        return_mask: bool = False,
    ) -> None:
        """
        Args:
            image_dir:
                Path to calibration images directory.
            mask_dir:
                Optional path to mask directory, e.g. ../data/busi/calibration/masks/0
            image_size:
                Target resize as (height, width), matching training usage.
            return_mask:
                If True, also load and return masks.

        Important:
            This implementation intentionally does NOT apply Albumentations Normalize(),
            because the provided MedicalDataSets class only divides by 255 after transform.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.return_mask = return_mask

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if self.return_mask and self.mask_dir is not None and not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.image_files: List[str] = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith(".png")]
        )

        if len(self.image_files) == 0:
            raise RuntimeError(f"No .png images found in: {self.image_dir}")

        # Match the visible transform style from training:
        # resize only, then manual /255 afterward.
        self.transform = Compose([
            Resize(self.image_size[0], self.image_size[1]),
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            {
                "image": torch.FloatTensor [3, H, W],
                "image_path": str,
                "image_name": str,
                "mask": torch.FloatTensor [1, H, W]   # only if requested
            }

        Notes:
            - Image loading intentionally follows MedicalDataSets:
              no explicit BGR->RGB conversion.
            - Image is scaled to [0,1] by dividing by 255.
        """
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        mask = None
        if self.return_mask and self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, image_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"Failed to load mask: {mask_path}")
                mask = mask[..., None]  # Match MedicalDataSets label shape

        # Apply resize transform
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Match MedicalDataSets exactly:
        # image.astype('float32') / 255
        # image.transpose(2, 0, 1)
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image)

        result: Dict[str, Any] = {
            "image": image_tensor,
            "image_path": image_path,
            "image_name": image_name,
        }

        if mask is not None:
            mask = mask.astype(np.float32) / 255.0
            mask = mask.transpose(2, 0, 1)
            result["mask"] = torch.from_numpy(mask)

        return result


def should_calibrate_module(module: nn.Module) -> bool:
    """
    Decide whether a module's output activation should be tracked during calibration.

    First-pass recommended policy:
        - track Conv2d outputs
        - track GELU outputs
        - track MaxPool2d outputs
        - optionally track Upsample outputs

    Args:
        module:
            PyTorch module instance.

    Returns:
        True if this module should receive a forward hook.

    Notes:
        This policy can be refined later.
        For now, it gives enough activation boundaries for PTQ.
    """
    calibrate_types = (
        nn.Conv2d,
        nn.GELU,
        nn.BatchNorm2d,
        nn.MaxPool2d,
        nn.Upsample,
    )
    if isinstance(module, calibrate_types):
        return True

    if module.__class__.__name__ == "Residual":
        return True

    return False


def create_empty_stats(name: str, module: nn.Module) -> ActivationStats:
    """
    Create an ActivationStats object for one module before calibration starts.

    Args:
        name:
            Stable hierarchical module name.
        module:
            PyTorch module instance.

    Returns:
        ActivationStats with initialized min/max values.

    Implementation detail:
        Use:
            min_val = +inf
            max_val = -inf
        so the first update replaces them correctly.
    """
    return ActivationStats(
        name=name,
        op_type=module.__class__.__name__,
        min_val=float('inf'),
        max_val=float('-inf'),
        num_updates=0,
        shape=[],
    )

def extract_tensor_list(obj: Any) -> List[torch.Tensor]:
    """
    Extract all torch.Tensor objects from a nested hook input/output object.

    Args:
        obj:
            Hook input or output object. Could be:
                - a Tensor
                - a tuple/list of Tensors
                - nested structures

    Returns:
        Flat list of torch.Tensor objects

    Why:
        Forward hooks often receive inputs as tuples.
        This helper makes tensor extraction robust.
    """
    tensors: List[torch.Tensor] = []

    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(extract_tensor_list(item))
    elif isinstance(obj, dict):
        for item in obj.values():
            tensors.extend(extract_tensor_list(item))

    return tensors

def update_input_activation_stats(
    stats_dict: Dict[str, ActivationStats],
    module_name: str,
    module: nn.Module,
    inputs: Tuple[Any, ...],
) -> None:
    """
    Record calibration stats for module inputs.

    For now, record only tensor inputs and key them as:
        "<module_name>:input0"
        "<module_name>:input1"
        ...

    This is especially useful when the producer op is functional
    (e.g. torch.cat or tensor addition) and therefore is not present
    in the execution trace as a named module output.
    """
    input_tensors = extract_tensor_list(inputs)

    for idx, tensor in enumerate(input_tensors):
        input_key = f"{module_name}:input{idx}"

        if input_key not in stats_dict:
            stats_dict[input_key] = ActivationStats(
                name=input_key,
                op_type=f"{module.__class__.__name__}_input",
                min_val=float("inf"),
                max_val=float("-inf"),
                num_updates=0,
                shape=[],
            )

        update_activation_stats(stats_dict[input_key], tensor)

def update_activation_stats(
    stats: ActivationStats,
    output_tensor: torch.Tensor,
) -> None:
    """
    Update running min/max statistics using one module output tensor.

    Args:
        stats:
            Existing ActivationStats object to update.
        output_tensor:
            Output tensor from a forward hook.

    Behavior:
        - detach from graph
        - move to CPU only if necessary
        - compute tensor min/max safely
        - update global min/max
        - increment num_updates
        - record latest shape

    Important:
        For calibration, use the raw float output of the prepared model.
        Do not quantize here.
    """
    # Detach from computation graph and move to CPU
    output_tensor = output_tensor.detach().cpu()
    
    # Compute tensor min/max
    tensor_min = float(output_tensor.min().item())
    tensor_max = float(output_tensor.max().item())
    
    # Update global min/max
    stats.min_val = min(stats.min_val, tensor_min)
    stats.max_val = max(stats.max_val, tensor_max)
    
    # Increment update counter
    stats.num_updates += 1
    
    # Record latest shape
    stats.shape = list(output_tensor.shape)


def register_calibration_hooks(
    model: nn.Module,
    name_map: Dict[int, str],
    stats_dict: Dict[str, ActivationStats],
) -> List[Any]:
    """
    Register forward hooks and forward pre-hooks on selected modules.

    Output hook records:
        "<module_name>"

    Input pre-hook records:
        "<module_name>:input0", "<module_name>:input1", ...

    Returns:
        List of hook handles to remove later.
    """
    hook_handles: List[Any] = []

    def create_output_hook(module_name: str) -> Any:
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                return

            if module_name not in stats_dict:
                stats_dict[module_name] = create_empty_stats(module_name, module)

            update_activation_stats(stats_dict[module_name], output)

        return hook

    def create_input_hook(module_name: str) -> Any:
        def pre_hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
            update_input_activation_stats(stats_dict, module_name, module, inputs)

        return pre_hook

    for module in model.modules():
        if not should_calibrate_module(module):
            continue

        module_id = id(module)
        if module_id not in name_map:
            continue

        module_name = name_map[module_id]

        # output hook
        out_handle = module.register_forward_hook(create_output_hook(module_name))
        hook_handles.append(out_handle)

        # input pre-hook
        in_handle = module.register_forward_pre_hook(create_input_hook(module_name))
        hook_handles.append(in_handle)

    return hook_handles


def remove_hooks(hook_handles: List[Any]) -> None:
    """
    Remove all registered forward hook handles.

    Args:
        hook_handles:
            List returned by register_calibration_hooks().

    Why:
        Avoid accidental duplicate hooks or memory leaks in later runs.
    """
    for handle in hook_handles:
        handle.remove()


def build_calibration_dataloader(
    image_dir: str,
    mask_dir: Optional[str],
    batch_size: int = 1,
    num_workers: int = 0,
    image_size: Optional[Tuple[int, int]] = None,
) -> DataLoader:
    """
    Build the calibration DataLoader.

    Args:
        image_dir:
            Path to BUSI calibration images.
        mask_dir:
            Optional path to masks/0.
        batch_size:
            Batch size for calibration. Start with 1 for simplicity.
        num_workers:
            DataLoader workers.
        image_size:
            Optional resize target (height, width). If None, defaults to (256, 256).

    Returns:
        PyTorch DataLoader over BUSICalibrationDataset.

    Notes:
        - shuffle=False for deterministic calibration
        - drop_last=False (we want all samples)
        - pin_memory=False (CPU is fine for PTQ calibration)
    """

    # Default to training size if not provided
    if image_size is None:
        image_size = (256, 256)

    dataset = BUSICalibrationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        return_mask=False,   # masks not needed for calibration
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,       # IMPORTANT: deterministic calibration
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return dataloader

def print_diagnostic_summary(stats_dict: Dict[str, ActivationStats], top_k: int = 10) -> None:
    """
    Print diagnostic summaries:
        1. Top-k layers by largest absolute activation value
        2. Top-k layers by widest activation range

    Args:
        stats_dict:
            Mapping from module name to ActivationStats.
        top_k:
            Number of layers to display in each ranking.
    """
    if not stats_dict:
        print("[calib] No activation stats available for diagnostics.")
        return

    stats_list = list(stats_dict.values())

    # Largest absolute max magnitude:
    # use max(abs(min_val), abs(max_val)) for each layer
    by_abs_max = sorted(
        stats_list,
        key=lambda s: max(abs(s.min_val), abs(s.max_val)),
        reverse=True
    )

    # Widest range:
    # range width = max_val - min_val
    by_range_width = sorted(
        stats_list,
        key=lambda s: (s.max_val - s.min_val),
        reverse=True
    )

    print(f"\n[calib] Top {top_k} layers by largest absolute activation magnitude:")
    for s in by_abs_max[:top_k]:
        abs_max = max(abs(s.min_val), abs(s.max_val))
        print(
            f"[calib] {s.name}: "
            f"abs_max={abs_max:.4f}, min={s.min_val:.4f}, max={s.max_val:.4f}, shape={s.shape}"
        )

    print(f"\n[calib] Top {top_k} layers by widest activation range:")
    for s in by_range_width[:top_k]:
        width = s.max_val - s.min_val
        print(
            f"[calib] {s.name}: "
            f"range_width={width:.4f}, min={s.min_val:.4f}, max={s.max_val:.4f}, shape={s.shape}"
        )

def save_activation_stats(
    stats_dict: Dict[str, ActivationStats],
    save_path: str,
) -> None:
    """
    Save collected activation statistics to JSON.

    Args:
        stats_dict:
            Dict of module_name -> ActivationStats
        save_path:
            Output JSON path, e.g. 'artifacts/calibration_stats.json'

    Behavior:
        - create parent directory if needed
        - convert dataclasses to plain dict
        - write nicely formatted JSON

    Output structure example:
        {
          "stem.conv.0": {
            "name": "stem.conv.0",
            "op_type": "Conv2d",
            "min_val": -1.234,
            "max_val": 2.345,
            "num_updates": 50,
            "shape": [1, 16, 256, 256]
          },
          ...
        }
    """
    # Create parent directory if needed
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Convert dataclasses to plain dicts
    stats_as_dicts = {
        module_name: asdict(stats)
        for module_name, stats in stats_dict.items()
    }
    
    # Write nicely formatted JSON
    with open(save_path, 'w') as f:
        json.dump(stats_as_dicts, f, indent=2)

@torch.no_grad()
def calibrate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    max_samples: Optional[int] = None,
) -> int:
    """
    Run the model over calibration data once.

    Args:
        model:
            Prepared float model with hooks already registered.
        dataloader:
            Calibration dataloader.
        max_samples:
            Optional cap on total images processed.

    Returns:
        Number of processed samples.

    Behavior:
        - loop over dataloader
        - feed image tensors through model
        - do nothing with outputs directly
        - hooks handle the statistics collection

    Notes:
        Since hooks do the actual tracking, this function mostly just executes
        forward passes over representative samples.
    """
    model.eval()
    num_processed = 0
    
    for batch in dataloader:
        # Extract image tensor from batch
        image_tensor = batch["image"]
        
        # Move to same device as model
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Run forward pass (hooks will collect statistics)
        _ = model(image_tensor)
        
        # Update counter
        num_processed += image_tensor.shape[0]
        
        # Check if we've reached the max samples limit
        if max_samples is not None and num_processed >= max_samples:
            break
    
    return num_processed

@torch.no_grad()
def run_calibration(
    checkpoint_path: str,
    variant: str,
    image_dir: str,
    mask_dir: Optional[str],
    save_path: str,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    image_size: Optional[Tuple[int, int]] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, ActivationStats]:
    """
    End-to-end PTQ calibration pipeline.

    Steps:
        1. Prepare the float inference model using prepare_model_for_ptq().
        2. Build a stable module name map.
        3. Register forward hooks on selected modules.
        4. Build calibration dataloader.
        5. Run calibration images through the model.
        6. Update activation min/max per hooked module.
        7. Remove hooks.
        8. Save stats to JSON.

    Args:
        checkpoint_path:
            Path to trained CMUNeXt checkpoint.
        variant:
            Model variant, e.g. 'base'.
        image_dir:
            Path to calibration images folder.
        mask_dir:
            Path to masks/0 folder, can be None.
        save_path:
            JSON output path for activation stats.
        max_samples:
            Optional cap on number of calibration images to use.
        batch_size:
            DataLoader batch size.
        num_workers:
            DataLoader workers.
        image_size:
            Optional resize.
        map_location:
            Usually 'cpu'.
        strict:
            Whether state_dict loading must be exact.

    Returns:
        stats_dict:
            Dict of collected ActivationStats.

    Important:
        This runs on the prepared float model, not the future integer engine.
    """
    # Step 1: Prepare the float inference model
    _, model, _ = prepare_model_for_ptq(
        checkpoint_path=checkpoint_path,
        variant=variant,
        map_location=map_location,
        strict=strict,
    )
    
    # Step 2: Build stable module name map
    name_map = build_module_name_map(model)
    
    # Step 3: Initialize stats dict and register hooks
    stats_dict: Dict[str, ActivationStats] = {}
    hook_handles = register_calibration_hooks(model, name_map, stats_dict)
    print(f"[calib] Registered hooks on {len(hook_handles)} modules")
    try:
        # Step 4: Build calibration dataloader
        dataloader = build_calibration_dataloader(
            image_dir=image_dir,
            mask_dir=mask_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
        )
        
        # Step 5-6: Run calibration images through model
        print(f"[calib] Running calibration with {len(dataloader)} batches...")
        num_processed = calibrate_one_epoch(
            model=model,
            dataloader=dataloader,
            max_samples=max_samples,
        )
        print(f"[calib] Processed {num_processed} calibration samples")
        
    finally:
        # Step 7: Remove hooks
        remove_hooks(hook_handles)
    
    # Step 8: Save stats to JSON
    print(f"[calib] Saving activation statistics to {save_path}")
    print_diagnostic_summary(stats_dict)
    save_activation_stats(stats_dict, save_path)
    
    return stats_dict


def print_calibration_summary(
    stats_dict: Dict[str, ActivationStats], max_items: int = 10
) -> None:
    """
    Print a short summary of collected activation statistics.

    Args:
        stats_dict:
            Module stats dictionary.
        max_items:
            How many modules to print.

    Suggested output:
        [calib] Collected stats for 48 modules
        [calib] stem.conv.0: min=-1.23 max=2.87 shape=[1,16,256,256]
        ...
    """
    print(f"[calib] Collected stats for {len(stats_dict)} modules")
    
    # Sort by module name for consistent output
    sorted_items = sorted(stats_dict.items())
    
    # Print up to max_items modules
    for module_name, stats in sorted_items[:max_items]:
        shape_str = str(stats.shape)
        print(f"[calib] {module_name}: min={stats.min_val:.2f} max={stats.max_val:.2f} shape={shape_str}")
    
    # If there are more items, indicate how many were skipped
    if len(stats_dict) > max_items:
        remaining = len(stats_dict) - max_items
        print(f"[calib] ... and {remaining} more modules")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for calibration.

    Recommended defaults for your setup:
        checkpoint_path:
            ../checkpoint/CMUNeXt_model_busi_train.pth
        image_dir:
            ../data/busi/calibration/images
        mask_dir:
            ../data/busi/calibration/masks/0
        save_path:
            calibration_stats.json
        variant:
            base
        batch_size:
            1
    """
    parser = argparse.ArgumentParser(description="PTQ calibration for CMUNeXt on BUSI")

    parser.add_argument(
        "checkpoint_path", type=str, help="Path to trained CMUNeXt .pth checkpoint"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "small", "large"],
        help="CMUNeXt model variant",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="../data/busi/calibration/images",
        help="Path to calibration images directory",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="../data/busi/calibration/masks/0",
        help="Path to calibration mask directory",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="calibration_stats.json",
        help="Output JSON path for activation statistics",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Calibration batch size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of calibration images",
    )
    parser.add_argument(
        "--input-width", type=int, default=None, help="Optional resize width"
    )
    parser.add_argument(
        "--input-height", type=int, default=None, help="Optional resize height"
    )
    parser.add_argument(
        "--map-location",
        type=str,
        default="cpu",
        help="Device mapping for checkpoint loading",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Use strict checkpoint loading"
    )

    return parser.parse_args()


def main() -> None:
    """
    Standalone entry point for calibration.

    Expected flow:
        1. Parse CLI args.
        2. Resolve optional image_size.
        3. Run calibration.
        4. Print summary.
        5. Report save path.

    Example usage:
        python3 calibrate.py ../checkpoint/CMUNeXt_model_busi_train.pth \
            --variant base \
            --image-dir ../data/busi/calibration/images \
            --mask-dir ../data/busi/calibration/masks/0 \
            --save-path calibration_stats.json
    """
    args = parse_args()

    image_size = None
    if args.input_width is not None and args.input_height is not None:
        image_size = (args.input_width, args.input_height)

    print(f"[calib] Loading checkpoint: {args.checkpoint_path}")
    print(f"[calib] Calibration images: {args.image_dir}")
    print(f"[calib] Saving stats to: {args.save_path}")

    stats_dict = run_calibration(
        checkpoint_path=args.checkpoint_path,
        variant=args.variant,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        save_path=args.save_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        map_location=args.map_location,
        strict=args.strict,
    )

    print_calibration_summary(stats_dict)
    print("[calib] Calibration completed successfully.")


if __name__ == "__main__":
    main()
