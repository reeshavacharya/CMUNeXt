import argparse
import csv
import os
import sys
import json
import time
import random

import cv2
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from quantization.quant_utils import (
    get_device,
    load_image_and_mask,
    load_model,
    preprocess,
)
from utils.metrics import iou_score

# --- CONSTANT SCALE CONFIGURATION ---
CONSTANT_EXPONENT = 16
SCALE = 2**CONSTANT_EXPONENT


def float_to_fixed_int(val):
    return int(round(val * SCALE))

def fold_batchnorm(model):
    for name, module in model.named_children():

        if isinstance(module, nn.Sequential):
            fold_batchnorm(module)

        if isinstance(module, nn.Conv2d):

            next_modules = list(model.named_children())

            # Find BN layer after this conv
            for i, (n, m) in enumerate(next_modules):
                if n == name and i + 1 < len(next_modules):
                    next_name, next_module = next_modules[i + 1]

                    if isinstance(next_module, nn.BatchNorm2d):

                        # Fold BN into Conv
                        W = module.weight.data
                        b = module.bias.data if module.bias is not None else torch.zeros(W.size(0))

                        gamma = next_module.weight.data
                        beta = next_module.bias.data
                        mean = next_module.running_mean
                        var = next_module.running_var
                        eps = next_module.eps

                        std = torch.sqrt(var + eps)

                        W_fold = W * (gamma / std).reshape(-1, 1, 1, 1)
                        b_fold = (b - mean) / std * gamma + beta

                        module.weight.data = W_fold
                        module.bias = nn.Parameter(b_fold)

                        # Remove BN
                        setattr(model, next_name, nn.Identity())

        else:
            fold_batchnorm(module)

    return model

def record(exp: int | None = None):
    device = get_device()
    checkpoint_path = os.path.join(
        PROJECT_ROOT, "checkpoint", "CMUNeXt_model_busi_train.pth"
    )

    # Optionally override global CONSTANT_EXPONENT / SCALE from CLI
    global CONSTANT_EXPONENT, SCALE
    if exp is not None:
        CONSTANT_EXPONENT = exp
        SCALE = 2**CONSTANT_EXPONENT

    # Load model and ensure it is in double precision (float64)
    # to match the 2^CONSTANT_EXPONENT precision requirement
    model = load_model(checkpoint_path, device).double()
    model = fold_batchnorm(model)
    param_dict = {}

    print(f"Extracting ALL state_dict entries at 2^{CONSTANT_EXPONENT} scale...")

    for name, tensor in model.state_dict().items():
        flat_data = tensor.detach().cpu().numpy().flatten()

        layer_data = []
        for val in flat_data:
            i = float_to_fixed_int(val)
            layer_data.append(i)

        param_dict[name] = layer_data
    param_dict["metadata"] = {"scale": SCALE, "exponent": CONSTANT_EXPONENT}
    out_name = f"model_params.json"
    with open(out_name, "w") as f:
        json.dump(param_dict, f, indent=2)

    print(f"Successfully saved verifiable parameters to {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record CMUNeXt parameters as fixed-point integers"
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=CONSTANT_EXPONENT,
        help="Exponent for the 2^exp fixed-point scale (default: 16)",
    )

    args = parser.parse_args()
    record(exp=args.exp)
