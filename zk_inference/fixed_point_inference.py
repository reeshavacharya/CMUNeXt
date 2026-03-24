import argparse
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

from quantization.quant_utils import get_device, load_image_and_mask, preprocess
from utils.metrics import iou_score
import utils.losses as losses
from network.CMUNeXt import CMUNeXt

# --- CONSTANT SCALE CONFIGURATION ---
CONSTANT_EXPONENT = 16
SCALE = 2**CONSTANT_EXPONENT

def strict_enforcement_hook(module, input, output):
    """
    Enforces the fixed-point grid on every intermediate activation.
    This mimics the truncation/rounding required in an integer circuit.
    """
    if isinstance(output, torch.Tensor):
        return torch.round(output * SCALE) / SCALE
    return output

def rehydrate_model(json_path, model):
    """
    Loads ALL entries (parameters + buffers) from the integer JSON.
    Reconstructs floats using: (-1)^sign * (integer / 2^CONSTANT_EXPONENT)
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Verifiable parameters not found at: {json_path}")

    print(f"Re-hydrating model from {json_path}...")
    with open(json_path, "r") as f:
        param_dict = json.load(f)

    # Use the full state_dict to ensure buffers (like running_mean) are included
    current_state_dict = model.state_dict()
    new_state_dict = {}

    for name, tensor in current_state_dict.items():
        if name not in param_dict:
            print(f"Warning: {name} missing in JSON. Keeping current values.")
            new_state_dict[name] = tensor
            continue

        # Get the [sign, integer] list
        verifiable_data = param_dict[name]
        
        # Reconstruct high-precision floats
        reconstructed = [((-1)**s) * (float(i) / SCALE) for s, i in verifiable_data]
        
        # Convert to tensor and reshape to the target layer dimensions
        re_tensor = torch.tensor(reconstructed, dtype=torch.float64).to(tensor.device)
        new_state_dict[name] = re_tensor.view(tensor.shape)

    # Load the complete state_dict
    model.load_state_dict(new_state_dict)
    print("Model successfully re-hydrated with all weights and BN buffers.")
    return model

def choose_random_case(val_list_path: str) -> str:
    if not os.path.exists(val_list_path):
        raise FileNotFoundError(f"Validation list not found: {val_list_path}")
    with open(val_list_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return random.choice(lines) if lines else None

def run_fixed_point_inference(case_name: str | None = None, exp: int | None = None):
    """Runs inference using the re-hydrated 2^CONSTANT_EXPONENT integer model.

    If case_name is provided, it must match an entry in the BUSI validation
    list; otherwise a random case from that list is used.

    If exp is provided, it overrides the global CONSTANT_EXPONENT and SCALE
    used for fixed-point snapping and model rehydration.
    """
    global CONSTANT_EXPONENT, SCALE

    if exp is not None:
        CONSTANT_EXPONENT = exp
        SCALE = 2**CONSTANT_EXPONENT

    device = get_device()
    base_dir = os.path.join(PROJECT_ROOT, "data", "busi")
    val_list_path = os.path.join(base_dir, "busi_val.txt")
    params_json = "model_params.json"

    # 1. Initialize architecture and force to double precision (float64)
    # Using default CMUNeXt-S parameters; adjust if using L or custom dims
    model = CMUNeXt(
        input_channel=3, 
        num_classes=1, 
        dims=[16, 32, 128, 160, 256], 
        depths=[1, 1, 1, 3, 1], 
        kernels=[3, 3, 7, 7, 7]
    ).to(device).double()

    # 2. Re-hydrate weights from the verifiable JSON
    model = rehydrate_model(params_json, model)
    model.eval()

    print(f"Applying strict enforcement hooks at 2^{CONSTANT_EXPONENT} scale...")
    for name, module in model.named_modules():
        # We hook leaf modules (Conv, BN, GELU, etc.) to simulate per-op truncation
        if len(list(module.children())) == 0:
            module.register_forward_hook(strict_enforcement_hook)

    criterion = losses.BCEDiceLoss().to(device)

    # 3. Select case (CLI-specified or random from validation list)
    if case_name is not None:
        if not os.path.exists(val_list_path):
            raise FileNotFoundError(f"Validation list not found: {val_list_path}")
        with open(val_list_path, "r") as f:
            valid_cases = [ln.strip() for ln in f.readlines() if ln.strip()]
        if case_name not in valid_cases:
            raise ValueError(f"Requested case '{case_name}' not found in validation list")
    else:
        case_name = choose_random_case(val_list_path)
    image_bgr, mask_arr, image_name = load_image_and_mask(base_dir, case_name)
    image_np, mask_np = preprocess(image_bgr, mask_arr)

    # 4. Convert and "Snap" input image to the 2^CONSTANT_EXPONENT grid for bit-true consistency
    image_t = torch.from_numpy(image_np).unsqueeze(0).to(device).double()
    image_t = torch.round(image_t * SCALE) / SCALE 
    
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device).double()

    # 5. Inference
    print(f"Running fixed-point inference on '{image_name}'...")
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model(image_t)
    torch.cuda.synchronize(device)
    end = time.perf_counter()

    # 6. Metrics and Saving
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

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixed_point_out")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"fixed_point_metrics_{CONSTANT_EXPONENT}.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save mask
    prob = torch.sigmoid(output)
    pred_mask = (prob > 0.5).float().cpu().numpy()[0, 0]
    cv2.imwrite(os.path.join(out_dir, f"fixed_point_pred_{CONSTANT_EXPONENT}.png"), (pred_mask * 255).astype(np.uint8))

    print(f"Fixed-point inference complete. F1 Score: {F1:.4f}")
    print(f"Results saved to {out_dir}/")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed-point CMUNeXt inference")
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Case name from BUSI validation list (e.g. 'benign (68)'); if omitted, a random case is used.",
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=CONSTANT_EXPONENT,
        help="Exponent for the 2^exp fixed-point scale (default: 16)",
    )

    args = parser.parse_args()
    run_fixed_point_inference(case_name=args.case, exp=args.exp)