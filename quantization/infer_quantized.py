import os
import random
import time
import json

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch

from quant_utils import PROJECT_ROOT, load_image_and_mask, preprocess
from utils.metrics import iou_score
import utils.losses as losses

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_PATH = os.path.join(PROJECT_ROOT, "checkpoint", "quantized", "cmunext_int8.plan")

# Load serialized engine
with open(EXPORT_PATH, "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)
if engine is None:
    print("FAILED: Engine could not be deserialized.")
    print(
        "Check if the TensorRT version and GPU architecture match the ones used to build the .path file."
    )
    exit(1)
# Allocate buffers
inputs = []
outputs = []
bindings = []

# Updated Buffer Allocation for TensorRT 10
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    size = trt.volume(engine.get_tensor_shape(name))
    dtype = trt.nptype(engine.get_tensor_dtype(name))

    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    # Check if it's input or output using get_tensor_mode
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        inputs.append({"host": host_mem, "device": device_mem})
    else:
        outputs.append({"host": host_mem, "device": device_mem})

    bindings.append(int(device_mem))

context = engine.create_execution_context()

# Prepare input data.
# 1) Pick a random case name from ../data/busi/busi_val.txt
base_dir = os.path.join(PROJECT_ROOT, "data", "busi")
val_list_path = os.path.join(base_dir, "busi_val.txt")

with open(val_list_path, "r") as f:
    case_names = [ln.strip() for ln in f.readlines() if ln.strip()]

if not case_names:
    raise RuntimeError(
        "Validation list is empty; cannot select a sample for inference."
    )

case_name = random.choice(case_names)

# 2) Load image and mask, preprocess to model input shape (1,3,256,256)
image_bgr, mask_arr, image_name = load_image_and_mask(base_dir, case_name)
image_np, mask_np = preprocess(image_bgr, mask_arr)

# image_np is C,H,W; expand batch dimension to get 1,C,H,W
image_np = np.expand_dims(image_np, axis=0).astype(np.float32)

# For metrics later, keep target as a torch tensor of shape 1,C,H,W
target_t = torch.from_numpy(mask_np).unsqueeze(0)  # 1,C,H,W

# Copy flattened input into the host buffer
if image_np.size != inputs[0]["host"].size:
    raise RuntimeError(
        f"Engine input buffer size {inputs[0]['host'].size} "
        f"does not match prepared image size {image_np.size}"
    )

np.copyto(inputs[0]["host"], image_np.ravel())

# Transfer input to device
cuda.memcpy_htod(inputs[0]["device"], inputs[0]["host"])

# update the address mapping explicitly
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    context.set_tensor_address(name, bindings[i])

# Run inference and time it
start_time = time.perf_counter()
context.execute_v2(bindings=bindings)
cuda.Context.synchronize()
end_time = time.perf_counter()
inference_time = end_time - start_time

# Transfer output back to host
cuda.memcpy_dtoh(outputs[0]["host"], outputs[0]["device"])

# Reshape output: assume batch size 1
output_name = "output" # Ensure this matches your ONNX output name
output_shape = engine.get_tensor_shape(output_name)
output_np = outputs[0]['host'].reshape(*output_shape)

# Compute loss and metrics similarly to the PyTorch inference helper
output_t = torch.from_numpy(output_np)  # 1,C,H,W

criterion = losses.BCEDiceLoss()
loss_val = float(criterion(output_t, target_t).item())
iou, dice, SE, PC, F1, SP, ACC = iou_score(output_t, target_t)

metrics = {
    "image_name": image_name,
    "engine_path": os.path.join("..", "checkpoint", "quantized", "cmunext_int8.plan"),
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
out_path = os.path.join(out_dir, "inference_quantized.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Quantized inference finished on '{image_name}'. Metrics saved to: {out_path}")
