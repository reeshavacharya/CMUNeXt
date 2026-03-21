import os
import json
import random
import time
import torch
import torch.nn as nn
from quant_utils import (
    PROJECT_ROOT,
    get_device,
    load_model,
    load_image_and_mask,
    preprocess,
)
from utils.metrics import iou_score
import utils.losses as losses

# --- Configuration ---
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoint/CMUNeXt_model_busi_train.pth")
CALIBRATION_JSON = "./calibration.json"
VAL_LIST_PATH = os.path.join(PROJECT_ROOT, "data/busi/busi_val.txt")

# --- 1. Parameter Extraction ---
with open(CALIBRATION_JSON, "r") as f:
    calib_data = json.load(f)

weight_params = {}
act_params = {}

for name, entry in calib_data.items():
    entry_type = entry.get("type")
    scale = entry.get("scale_fp32")
    if entry_type == "weight":
        weight_params[name] = scale
    else:
        act_params[name] = scale


def _prepare_scale(scale, ref_tensor, is_weight: bool = False):
    s = torch.as_tensor(scale, dtype=ref_tensor.dtype, device=ref_tensor.device)
    if s.ndim == 0 or s.numel() == 1:
        return s

    shape = [1] * ref_tensor.dim()
    if is_weight:
        # Weights [OC, IC, H, W] -> Scale aligns with Dim 0
        shape[0] = s.numel()
    else:
        # Activations [B, C, H, W] -> Scale aligns with Dim 1
        shape[1] = s.numel()
    return s.view(*shape)


def _tensor_stats(t: torch.Tensor) -> dict:
    """Return basic statistics for a tensor, suitable for JSON logging."""

    if t.numel() == 0:
        return {"shape": list(t.shape), "dtype": str(t.dtype), "empty": True}

    t_cpu = t.detach().to("cpu", non_blocking=True)
    return {
        "shape": list(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
        "min": float(t_cpu.min().item()),
        "max": float(t_cpu.max().item()),
        "mean": float(t_cpu.mean().item()),
        "std": float(t_cpu.std(unbiased=False).item()),
    }

# --- 2. Core Quantization Logic ---
def quantize_initial_input(tensor, scale):
    scale_t = _prepare_scale(scale, tensor)
    q = torch.round(tensor / scale_t)
    return torch.clamp(q, -128, 127)

# --- 3. Integer-Only Re-quantization Hook ---
def make_requant_hook(layer_name: str, in_scale, w_scale, next_in_scale, act_logs: dict):
    """
    CORRECTED MATH:
    M = (S_in * S_weight) / S_next_in
    """
    # Weight scale is often per-channel, so M will be per-channel
    s_in = torch.tensor(in_scale)
    s_w = torch.tensor(w_scale)
    s_next = torch.tensor(next_in_scale)
    
    # Rescale multiplier
    effective_m = (s_in * s_w) / s_next

    def hook(module, input, output):
        # Broadcast multiplier M across the output channels
        m = _prepare_scale(effective_m, output)

        # Simulate the INT32 -> INT8 rescaling
        y = torch.round(output * m)
        y = torch.clamp(y, -128, 127)

        # Log activation statistics + integer checks for this layer (quantized path)
        inp_tensor = input[0] if isinstance(input, (list, tuple)) and input else input
        layer_entry = {}
        if torch.is_tensor(inp_tensor):
            layer_entry["input"] = {
                "stats": _tensor_stats(inp_tensor),
                "integer_check": _integer_check_stats(inp_tensor),
            }
        if torch.is_tensor(y):
            layer_entry["output"] = {
                "stats": _tensor_stats(y),
                "integer_check": _integer_check_stats(y),
            }
        act_logs[layer_name] = layer_entry

        return y
    return hook

# --- 4. Weight Pre-Quantization ---
def _integer_check_stats(t: torch.Tensor) -> dict:
    if t.numel() == 0:
        return {"all_integer": True, "max_abs_diff_from_round": 0.0}
    diff = (t - t.round()).abs()
    max_diff = float(diff.max().item())
    return {"all_integer": bool(max_diff < 1e-6), "max_abs_diff_from_round": max_diff}


def prepare_quantized_weights(model, weight_params, weight_logs):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            key = f"{name}.weight"
            if key not in weight_params:
                continue

            # Load scales as a tensor
            s = torch.tensor(weight_params[key], device=module.weight.device)

            if isinstance(module, nn.Conv2d):
                # Weight shape: [Out_Channels, In_Channels, K_H, K_W]
                # Scale shape must be [Out_Channels, 1, 1, 1]
                s = s.view(-1, 1, 1, 1)
            else:
                # Linear shape: [Out_Features, In_Features]
                # Scale shape must be [Out_Features, 1]
                s = s.view(-1, 1)

            # Original FP32 weights
            w_fp32 = module.weight.data.clone()
            orig_stats = _tensor_stats(w_fp32)

            # Quantize
            w_q = torch.round(w_fp32 / s)
            w_q = torch.clamp(w_q, -128, 127)
            int_stats = _integer_check_stats(w_q)
            q_stats = _tensor_stats(w_q)

            module.weight.data = w_q

            weight_logs[name] = {
                "original": orig_stats,
                "quantized": q_stats,
                "integer_check": int_stats,
            }


# --- 5. Main Inference Flow ---
def main(case_name: str | None = None):
    device = get_device()
    model = load_model(MODEL_PATH, device)

    # Containers for per-layer logs
    weight_logs: dict = {}
    act_logs: dict = {}

    # 1. Apply static weight quantization (and log FP32 vs quantized comparison)
    prepare_quantized_weights(model, weight_params, weight_logs)

    # 2. Register Hooks (Simulating Sequential Execution)
    # Note: For CMUNeXt, we map the output of layer i to the input requirements of i+1
    named_leaf_modules = [(n, m) for n, m in model.named_modules() if len(list(m.children())) == 0]
    
    for i in range(len(named_leaf_modules) - 1):
        curr_name, curr_mod = named_leaf_modules[i]
        next_name, _ = named_leaf_modules[i+1]
        
        # Check if we have all 3 required pieces for the M calculation
        has_weights = f"{curr_name}.weight" in weight_params
        has_scales = curr_name in act_params and next_name in act_params
        
        if has_weights and has_scales:
            # Get the scale of the data ENTERING this layer
            if i == 0:
                s_in = act_params.get("input", 1.0)
            else:
                # The input to this layer is the output scale of the previous layer
                s_in = act_params.get(named_leaf_modules[i-1][0], 1.0)
            
            # Handle per-channel list conversion for s_in if it's a list
            if isinstance(s_in, list): s_in = s_in[0] 
                
            s_w = weight_params[f"{curr_name}.weight"]
            s_next = act_params[curr_name]  # Output scale of current layer

            hook = make_requant_hook(curr_name, s_in, s_w, s_next, act_logs)
            curr_mod.register_forward_hook(hook)

    # 3. Load Sample (CLI-specified or random from validation list)
    with open(VAL_LIST_PATH, "r") as f:
        cases = [ln.strip() for ln in f.readlines() if ln.strip()]

    if case_name is not None:
        if case_name not in cases:
            raise ValueError(f"Requested file_name '{case_name}' not found in validation list")
    else:
        case_name = random.choice(cases)
    image, mask, image_id = load_image_and_mask(os.path.join(PROJECT_ROOT, "data/busi"), case_name)
    image, mask = preprocess(image, mask)

    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    target_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    in_scale = act_params.get("input", 1.0)
    image_int8 = quantize_initial_input(image_tensor, in_scale)

    # 4. Inference (timed)
    with torch.no_grad():
        start_time = time.perf_counter()
        output_int8 = model(image_int8)
        end_time = time.perf_counter()
        inference_time = end_time - start_time

    # 5. Final Dequantization
    final_scale = act_params.get("Conv_1x1", 1.0)
    if isinstance(final_scale, list):
        final_scale = final_scale[0]
    final_fp32 = output_int8 * float(final_scale)

    # 6. Loss and Metrics (match ../quantization/inference.json schema)
    criterion = losses.BCEDiceLoss()
    loss_val = float(criterion(final_fp32, target_t).item())
    iou, dice, SE, PC, F1, SP, ACC = iou_score(final_fp32, target_t)

    metrics = {
        "image_name": image_id,
        "checkpoint": MODEL_PATH,
        "loss": loss_val,
        "iou": float(iou),
        "dice": float(dice),
        "F1": float(F1),
        "accuracy": float(ACC),
        "sensitivity": float(SE),
        "precision": float(PC),
        "specificity": float(SP),
        "inference_time_sec": float(inference_time),
    }

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "infer_quantized.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Also log per-layer weight/activation statistics for this inference,
    # combining FP32 vs quantized weight comparison and activation stats.
    layers_combined = {}
    for lname in set(weight_logs.keys()) | set(act_logs.keys()):
        entry = {}
        if lname in weight_logs:
            entry["weights"] = weight_logs[lname]
        if lname in act_logs:
            entry["activations"] = act_logs[lname]
        layers_combined[lname] = entry

    log_entry = {
        "image_name": image_id,
        "checkpoint": MODEL_PATH,
        "layers": layers_combined,
    }

    logs_path = os.path.join(out_dir, "infer_quantized_logs.json")
    # Overwrite log file for each inference so it is always "clean"
    with open(logs_path, "w") as f:
        json.dump(log_entry, f, indent=4)

    print(f"Verified Case: {image_id}")
    print("Inference complete. Integer constraints maintained in hidden layers.")
    print(f"Metrics saved to: {out_path}")
    print(f"Layer logs appended to: {logs_path}")

    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual quantized inference for CMUNeXt")
    parser.add_argument(
        "--file_name",
        type=str,
        default=None,
        help="Case name from BUSI validation list (e.g. 'benign (68)'); if omitted, a random case is used.",
    )

    args = parser.parse_args()
    main(case_name=args.file_name)