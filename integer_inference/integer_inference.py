import argparse
import os
import sys
import json
import time
import random
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from int_quant_utils import get_device, load_image_and_mask, preprocess
from utils.metrics import iou_score
import utils.losses as losses
from network.CMUNeXt import CMUNeXt
from gelu_approx import approx_gelu
# NFGEN GELU
from NFGen.main import generate_nonlinear_config

# --- CONSTANT SCALE CONFIGURATION ---
CONSTANT_EXPONENT = 16
SCALE = 2**CONSTANT_EXPONENT


# =========================================================
# INTEGER LAYERS
# =========================================================

class IntConv2d(nn.Module):
    def __init__(self, weight, bias=None, stride=1, padding=0, groups=1):
        super().__init__()
        self.weight = weight.long()
        # Store bias as 1D and reshape for broadcasting during forward
        if bias is not None:
            self.bias = bias.view(-1).long()
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, x):
        y = F.conv2d(
            x,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

        # 🔥 CRITICAL: rescale after multiplication
        y = y >> CONSTANT_EXPONENT

        if self.bias is not None:
            # Broadcast bias over spatial dimensions
            y += self.bias.view(1, -1, 1, 1)

        return y


class IntReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0)


class IntUpsample(nn.Module):
    """Integer-friendly upsampling.

    Performs interpolation in float, then re-quantizes back to the
    integer grid defined by CONSTANT_EXPONENT / SCALE.
    """

    def __init__(self, scale_factor=2, mode="bilinear", align_corners=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        # x is integer-valued tensor scaled by SCALE.
        x_float = x.float()
        y = F.interpolate(
            x_float,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        # Re-quantize to integer grid.
        return torch.round(y).long()

class IntGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x is int (scaled by SCALE)

        # Convert to float ONLY for polynomial eval
        x_float = x.float() / SCALE

        y = approx_gelu(x_float)

        # Back to integer
        y_int = torch.round(y * SCALE).long()

        return y_int

# =========================================================
# LOAD INTEGER MODEL
# =========================================================

def load_integer_weights(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)


# =========================================================
# PATCH MODEL TO INTEGER
# =========================================================

def convert_model_to_int(model, param_dict, prefix: str = ""):
    """Replace Conv2d with IntConv2d using integer weights.

    `param_dict` keys come from `model.state_dict()` in record.py, so they
    use full hierarchical names like "down1.0.weight". Track the current
    module path via `prefix` to construct matching keys when recursing.
    """

    for name, module in model.named_children():

        full_name = f"{prefix}{name}" if prefix == "" else f"{prefix}{name}"

        if isinstance(module, nn.Conv2d):

            w_key = f"{full_name}.weight"
            b_key = f"{full_name}.bias"

            if w_key not in param_dict:
                raise KeyError(f"Missing integer weight for layer '{w_key}' in param_dict")

            # Create integer weight tensor on the same device as the
            # original module so input and weights share device.
            weight = torch.tensor(param_dict[w_key], device=module.weight.device).view(
                module.weight.shape
            )
            bias = None

            if module.bias is not None and b_key in param_dict:
                bias = torch.tensor(
                    param_dict[b_key], device=module.bias.device
                ).view(module.bias.shape)

            new_layer = IntConv2d(
                weight=weight,
                bias=bias,
                stride=module.stride,
                padding=module.padding,
                groups=module.groups
            )

            setattr(model, name, new_layer)

        elif isinstance(module, nn.BatchNorm2d):
            # ❌ REMOVE BN (assumed pre-folded)
            setattr(model, name, nn.Identity())

        elif isinstance(module, nn.GELU):
            # Replace GELU → ReLU (or NFGEN later)
            setattr(model, name, IntGELU())

        elif isinstance(module, nn.ReLU):
            setattr(model, name, IntReLU())

        elif isinstance(module, nn.Upsample):
            # Replace floating-point upsample with integer-friendly wrapper.
            new_layer = IntUpsample(
                scale_factor=module.scale_factor,
                mode=module.mode,
                align_corners=module.align_corners,
            )
            setattr(model, name, new_layer)

        else:
            convert_model_to_int(module, param_dict, prefix=f"{full_name}.")

    return model


# =========================================================
# MAIN INFERENCE
# =========================================================

def run_integer_inference(case_name=None, exp=None):

    global CONSTANT_EXPONENT, SCALE

    if exp is not None:
        CONSTANT_EXPONENT = exp
        SCALE = 2**CONSTANT_EXPONENT

    # Integer convolutions with Long tensors are not supported by
    # cuDNN on GPU, so we run integer inference on CPU even if CUDA
    # is available.
    device = torch.device("cpu")

    base_dir = os.path.join(PROJECT_ROOT, "data", "busi")
    val_list_path = os.path.join(base_dir, "busi_val.txt")
    params_json = "model_params.json"

    # 1. Load model structure ONLY
    model = CMUNeXt().to(device)

    # 2. Load integer weights
    param_dict = load_integer_weights(params_json)

    # 3. Convert model
    model = convert_model_to_int(model, param_dict, prefix="")
    model.eval()

    criterion = losses.BCEDiceLoss().to(device)

    # 4. Select case
    if case_name is None:
        with open(val_list_path, "r") as f:
            case_name = random.choice([ln.strip() for ln in f.readlines() if ln.strip()])

    image_bgr, mask_arr, image_name = load_image_and_mask(base_dir, case_name)
    image_np, mask_np = preprocess(image_bgr, mask_arr)

    # 🔥 INTEGER INPUT
    image_int = torch.from_numpy(image_np).unsqueeze(0).to(device)
    image_int = torch.round(image_int * SCALE).long()

    mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device).float()

    # 5. Inference
    print(f"Running INTEGER inference on '{image_name}'...")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():
        output_int = model(image_int)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    # 🔥 Convert to float ONLY here
    output = output_int.float() / SCALE

    # 6. Metrics
    loss_val = criterion(output, mask_t).item()
    iou, dice, SE, PC, F1, SP, ACC = iou_score(output, mask_t)

    metrics = {
        "image_name": image_name,
        "loss": float(loss_val),
        "iou": float(iou),
        "dice": float(dice),
        "F1": float(F1),
        "accuracy": float(ACC),
        "sensitivity": float(SE),
        "precision": float(PC),
        "specificity": float(SP),
        "inference_time_sec": float(end - start),
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integer_out")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"integer_metrics_{CONSTANT_EXPONENT}.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    prob = torch.sigmoid(output)
    pred_mask = (prob > 0.5).float().cpu().numpy()[0, 0]

    cv2.imwrite(
        os.path.join(out_dir, f"integer_pred_{CONSTANT_EXPONENT}.png"),
        (pred_mask * 255).astype(np.uint8)
    )

    print(f"INTEGER inference complete. F1 Score: {F1:.4f}")
    print(f"Results saved to {out_dir}/")

    return metrics


# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integer CMUNeXt inference")

    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--exp", type=int, default=CONSTANT_EXPONENT)

    args = parser.parse_args()

    run_integer_inference(case_name=args.case, exp=args.exp)