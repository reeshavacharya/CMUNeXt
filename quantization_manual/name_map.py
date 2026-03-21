import os
import sys
import json
import torch

# Ensure project root (containing the `network` package) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from network.CMUNeXt import cmunext

# 1. Load the model
model = cmunext() 
checkpoint = torch.load("../checkpoint/CMUNeXt_model_busi_train.pth", map_location='cpu')
model.load_state_dict(checkpoint)

def generate_parameterized_map(qdq_params_path, pytorch_module_names):
    with open(qdq_params_path, 'r') as f:
        qdq_params = json.load(f)
    
    onnx_keys = list(qdq_params.keys())
    final_mapping = []

    # Suffixes that represent the output of a layer (Activations)
    activation_suffixes = [
        "Conv_output_0", 
        "Relu_output_0", 
        "BatchNormalization_output_0", 
        "Mul_1_output_0", # GELU
        "Resize_output_0", 
        "Add_output_0"
    ]

    for pt_name in pytorch_module_names:
        onnx_path_pattern = "/" + pt_name.replace(".", "/")
        
        # 1. Find Activation Mapping (The Hook Point)
        act_matches = [k for k in onnx_keys if onnx_path_pattern in k and any(s in k for s in activation_suffixes)]
        
        if act_matches:
            best_act_key = act_matches[-1]
            params = qdq_params[best_act_key]
            final_mapping.append({
                "torch": pt_name,
                "onnx": best_act_key,
                "scale": params["scale"],
                "zero_point": params["zero_point"],
                "type": "activation"
            })

        # 2. Find Weight Mapping (The Parameter Override)
        weight_key = f"{pt_name}.weight"
        if weight_key in qdq_params:
            params = qdq_params[weight_key]
            final_mapping.append({
                "torch": weight_key,
                "onnx": params["node_name"], # Weight nodes use their node name as ID
                "scale": params["scale"],
                "zero_point": params["zero_point"],
                "type": "weight"
            })

    # 3. Special case for the network input
    if "input" in qdq_params:
        final_mapping.append({
            "torch": "input",
            "onnx": "input",
            "scale": qdq_params["input"]["scale"],
            "zero_point": qdq_params["input"]["zero_point"],
            "type": "input"
        })

    return final_mapping

# Example usage:
pytorch_module_names = [n for n, m in model.named_modules() if len(list(m.children())) == 0]
name_map = generate_parameterized_map("qdq_params.json", pytorch_module_names)
with open("name_map.json", "w") as f:
    json.dump(name_map, f, indent=4)