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

from quantization.quant_utils import get_device, load_image_and_mask, load_model, preprocess
from utils.metrics import iou_score

# --- CONSTANT SCALE CONFIGURATION ---
CONSTANT_EXPONENT = 16
SCALE = 2**CONSTANT_EXPONENT

def float_to_verifiable_int(val):
    """
    Converts a float to a verifiable sign/integer pair at 2^CONSTANT_EXPONENT scale.
    """
    # Ensure we work with plain Python numeric types for JSON compatibility
    v = float(val)
    sign = 1 if v < 0 else 0
    # Use round to handle the closest integer on the 2^CONSTANT_EXPONENT grid
    integer_val = int(round(abs(v) * SCALE))
    return v, sign, integer_val

def record(exp: int | None = None):
    device = get_device()
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoint", "CMUNeXt_model_busi_train.pth")
    
    # Optionally override global CONSTANT_EXPONENT / SCALE from CLI
    global CONSTANT_EXPONENT, SCALE
    if exp is not None:
        CONSTANT_EXPONENT = exp
        SCALE = 2**CONSTANT_EXPONENT

    # Load model and ensure it is in double precision (float64)
    # to match the 2^CONSTANT_EXPONENT precision requirement
    model = load_model(checkpoint_path, device).double()
    
    param_dict = {}
    
    print(f"Extracting ALL state_dict entries at 2^{CONSTANT_EXPONENT} scale...")
    
    for name, tensor in model.state_dict().items():
        flat_data = tensor.detach().cpu().numpy().flatten()
        
        layer_data = []
        for val in flat_data:
            _, s, i = float_to_verifiable_int(val)
            layer_data.append([s, i])
            
        param_dict[name] = layer_data

    # param_dict = {name: p.detach().cpu().numpy().tolist() for name, p in model.named_parameters()}
    out_name = f"model_params.json"
    with open(out_name, "w") as f:
        json.dump(param_dict, f, indent=2)
        
    print(f"Successfully saved verifiable parameters to {out_name}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record CMUNeXt parameters as fixed-point integers")
    parser.add_argument(
        "--exp",
        type=int,
        default=CONSTANT_EXPONENT,
        help="Exponent for the 2^exp fixed-point scale (default: 16)",
    )

    args = parser.parse_args()
    record(exp=args.exp)

    