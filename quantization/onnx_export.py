import os
import torch
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from network.CMUNeXt import cmunext

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "..", "checkpoint", "CMUNeXt_model_busi_train.pth")
EXPORT_PATH = os.path.join(SCRIPT_DIR, "..", "checkpoint", "CMUNeXt_model_busi_train.onnx")

# Build model and load trained weights
model = cmunext()
state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Dummy input: batch=1, 3-channel RGB, 256x256 (matches img_size in main.py)
dummy_input = torch.randn(1, 3, 256, 256)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    EXPORT_PATH,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Model exported to {os.path.abspath(EXPORT_PATH)}")
