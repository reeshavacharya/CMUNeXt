from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any

import cv2
import numpy as np
import torch

from int_engine import (
    load_quantized_artifacts,
    run_integer_inference,
    load_input_image_as_tensor,
)

from utils import losses
from utils.metrics import iou_score


def load_mask_as_tensor(
    mask_path: str,
    image_size: tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """
    Load one BUSI mask using preprocessing aligned with the integer engine setup.

    Steps:
        - cv2.imread in grayscale
        - resize with nearest-neighbor
        - divide by 255
        - add channel dimension
        - add batch dimension

    Returns:
        Tensor of shape [1, 1, H, W], dtype float32
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to load mask: {mask_path}")

    # image_size is (H, W); cv2.resize expects (W, H)
    mask = cv2.resize(mask, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=0)   # [1, H, W]
    mask = np.expand_dims(mask, axis=0)   # [1, 1, H, W]

    return torch.from_numpy(mask).to(torch.float32)


def evaluate_integer_sample(
    artifacts_dir: str,
    image_path: str,
    mask_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate integer inference on a single sample.

    Args:
        artifacts_dir:
            Directory containing quantized artifacts.
        image_path:
            Path to input image.
        mask_path:
            Path to corresponding segmentation mask.
        device:
            torch device

    Returns:
        Dict containing output tensors and metric values.
    """
    artifacts = load_quantized_artifacts(artifacts_dir)

    # Load image using the exact same preprocessing as int_engine.py
    x_fp = load_input_image_as_tensor(image_path=image_path).to(device)

    # Load target mask
    target = load_mask_as_tensor(mask_path=mask_path).to(device)

    # Same criterion used in training/eval
    criterion = losses.__dict__['BCEDiceLoss']().to(device)

    with torch.no_grad():
        result = run_integer_inference(
            x_fp=x_fp,
            artifacts=artifacts,
            return_trace=False,
        )

        if "output_fp" not in result:
            raise RuntimeError(
                "run_integer_inference() did not return 'output_fp'. "
                "Make sure activation_qparams.json is exported and int_engine.py "
                "builds full output_qparams with scale."
            )

        output = result["output_fp"].to(device)

        loss = criterion(output, target)
        iou, dice, SE, PC, F1, _, ACC = iou_score(output, target)

    metrics = {
        "loss": float(loss.item()),
        "iou": float(iou),
        "dice": float(dice),
        "SE": float(SE),
        "PC": float(PC),
        "F1": float(F1),
        "ACC": float(ACC),
    }

    return {
        "image_path": image_path,
        "mask_path": mask_path,
        "output_int": result["output_int"],
        "output_qparams": result["output_qparams"],
        "output_fp": result["output_fp"],
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args for single-sample integer evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate integer inference on one BUSI sample"
    )

    parser.add_argument(
        "--artifacts-dir",
        type=str,
        required=True,
        help="Path to quantized_artifacts directory",
    )
    parser.add_argument(
        "--image-sample",
        type=str,
        required=True,
        help="Image filename, e.g. 'malignant (64).png'",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="../data/busi/images",
        help="Directory containing BUSI images",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="../data/busi/masks/0",
        help="Directory containing BUSI masks",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")

    image_path = os.path.join(args.image_dir, args.image_sample)
    mask_path = os.path.join(args.mask_dir, args.image_sample)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    result = evaluate_integer_sample(
        artifacts_dir=args.artifacts_dir,
        image_path=image_path,
        mask_path=mask_path,
        device=device,
    )

    metrics = result["metrics"]

    print("[evaluate] Integer sample evaluation completed.")
    print(f"[evaluate] Image: {result['image_path']}")
    print(f"[evaluate] Mask:  {result['mask_path']}")
    print(f"[evaluate] Output int shape: {tuple(result['output_int'].shape)}")
    print(f"[evaluate] Output int dtype: {result['output_int'].dtype}")
    print(
        "[evaluate] "
        f"loss={metrics['loss']:.4f} "
        f"iou={metrics['iou']:.4f} "
        f"dice={metrics['dice']:.4f} "
        f"SE={metrics['SE']:.4f} "
        f"PC={metrics['PC']:.4f} "
        f"F1={metrics['F1']:.4f} "
        f"ACC={metrics['ACC']:.4f}"
    )


if __name__ == "__main__":
    main()