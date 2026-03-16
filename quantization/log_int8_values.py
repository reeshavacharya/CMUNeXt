# This script will run inference on a single image and dump the raw [-128, 127] 
# values for every debug tensor into a folder.

import os
import random
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from quant_utils import PROJECT_ROOT, load_image_and_mask, preprocess

# --- 1. Setup & Load Engine ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
PLAN_PATH = "../checkpoint/quantized/cmunext_int8.plan"
DUMP_DIR = "tensor_dumps"
os.makedirs(DUMP_DIR, exist_ok=True)

with open(PLAN_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# --- 2. Identify Debug Tensors & Allocate Buffers ---
# We use TensorRT 10's I/O Tensor API
input_tensors = []
output_tensors = []
tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

print(f"Engine has {len(tensor_names)} total I/O tensors (including Debug Tensors)")

for name in tensor_names:
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    size = trt.volume(shape)
    
    # Allocate memory
    # Note: Debug tensors for INT8 layers will return trt.int8 as their dtype
    host_mem = cuda.pagelocked_empty(size, trt.nptype(dtype))
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    
    tensor_info = {
        "name": name,
        "host": host_mem,
        "device": device_mem,
        "shape": shape,
        "dtype": trt.nptype(dtype)
    }
    
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        input_tensors.append(tensor_info)
    else:
        output_tensors.append(tensor_info)

# --- 3. Prepare Dummy Input ---
# Replace with your actual preprocessed image logic if needed
# --------------------------------------------------------------------

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

input_data = image_np
np.copyto(input_tensors[0]["host"], input_data.ravel())

# --------------------------------------------------------------------


# --- 4. Execution ---
# Bind addresses
for t in input_tensors + output_tensors:
    context.set_tensor_address(t["name"], int(t["device"]))

# Copy inputs to GPU
for t in input_tensors:
    cuda.memcpy_htod(t["device"], t["host"])

for t in output_tensors:
    # Fill with 55 so we know if the GPU actually changed it
    t["host"].fill(55)
    cuda.memcpy_htod(t["device"], t["host"])

print("Running inference...")
context.execute_v2([]) # execute_v2 with addresses set via set_tensor_address
cuda.Context.synchronize()

# --- 5. Log .raw Files ---
print(f"Logging tensors to {DUMP_DIR}...")
for t in output_tensors:
    # Copy result back to host
    cuda.memcpy_dtoh(t["host"], t["device"])
    
    # Sanitize filename (remove slashes from layer names)
    safe_name = t["name"].replace("/", "_").replace(".", "_")
    file_path = os.path.join(DUMP_DIR, f"{safe_name}.raw")
    
    # Save raw bytes
    t["host"].tofile(file_path)
    
    # If it's an INT8 tensor, print a small sample of the integers
    if t["dtype"] == np.int8:
        sample = t["host"].view(np.int8)[:10]
        print(f"Saved INT8 tensor: {t['name']} | Sample values: {sample}")

print("Done.")