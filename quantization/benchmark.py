import json
import os
import time
from typing import Dict, List

import numpy as np
import tensorrt as trt
import torch
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (initialises CUDA context)

from quant_utils import (
	PROJECT_ROOT,
	get_device,
	load_image_and_mask,
	load_model,
	preprocess,
)
from utils.metrics import iou_score
import utils.losses as losses


VAL_LIST_PATH = os.path.join(PROJECT_ROOT, "data", "busi", "busi_val.txt")
BASE_DIR = os.path.join(PROJECT_ROOT, "data", "busi")

CLEAN_CHECKPOINT_PATH = os.path.join(
	PROJECT_ROOT, "checkpoint", "CMUNeXt_model_busi_train.pth"
)
TRT_ENGINE_PATH = os.path.join(
	PROJECT_ROOT, "checkpoint", "quantized", "cmunext_int8.path"
)

BENCHMARK_PER_CASE_PATH = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "benchmark_per_case.json"
)
BENCHMARK_AGG_PATH = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "benchmark.json"
)


METRIC_KEYS = [
	"loss",
	"iou",
	"dice",
	"F1",
	"accuracy",
	"sensitivity",
	"precision",
	"specificity",
	"inference_time_sec",
]


def load_val_cases(val_list_path: str) -> List[str]:
	"""Load all case names (without extension) from the validation list file."""

	if not os.path.exists(val_list_path):
		raise FileNotFoundError(f"Validation list not found: {val_list_path}")

	with open(val_list_path, "r") as f:
		cases = [ln.strip() for ln in f.readlines() if ln.strip()]

	if not cases:
		raise RuntimeError("Validation list is empty; cannot run benchmark.")

	return cases


def run_clean_inference_for_case(
	model: torch.nn.Module,
	criterion: torch.nn.Module,
	device: torch.device,
	base_dir: str,
	case_name: str,
) -> Dict[str, float]:
	"""Run PyTorch (clean) inference for a single case and compute metrics.

	This mirrors the logic in infer_clean.py but works on a specific case name
	instead of picking one at random.
	"""

	image_bgr, mask_arr, image_name = load_image_and_mask(base_dir, case_name)
	image_np, mask_np = preprocess(image_bgr, mask_arr)

	image_t = torch.from_numpy(image_np).unsqueeze(0).to(device)
	mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)

	# Time PyTorch inference on GPU
	torch.cuda.synchronize(device)
	start = time.perf_counter()
	with torch.no_grad():
		output = model(image_t)
	torch.cuda.synchronize(device)
	end = time.perf_counter()
	inference_time = end - start

	loss_val = criterion(output, mask_t).item()
	iou, dice, SE, PC, F1, SP, ACC = iou_score(output, mask_t)

	metrics = {
		"image_name": image_name,
		"checkpoint": CLEAN_CHECKPOINT_PATH,
		"loss": float(loss_val),
		"iou": float(iou),
		"dice": float(dice),
		"F1": float(F1),
		"accuracy": float(ACC),
		"sensitivity": float(SE),
		"precision": float(PC),
		"specificity": float(SP),
		"inference_time_sec": float(inference_time),
	}
	return metrics


def build_trt_engine(engine_path: str):
	"""Load a TensorRT engine from disk and allocate buffers.

	This follows the pattern used in infer_quantized.py, but refactored into a
	reusable helper suitable for repeated inference across the full dataset.
	"""

	if not os.path.exists(engine_path):
		raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

	logger = trt.Logger(trt.Logger.WARNING)
	with open(engine_path, "rb") as f:
		engine_data = f.read()

	runtime = trt.Runtime(logger)
	engine = runtime.deserialize_cuda_engine(engine_data)
	if engine is None:
		raise RuntimeError("Engine could not be deserialized. Check TensorRT/GPU setup.")

	inputs = []
	outputs = []
	bindings = []

	for i in range(engine.num_io_tensors):
		name = engine.get_tensor_name(i)
		size = trt.volume(engine.get_tensor_shape(name))
		dtype = trt.nptype(engine.get_tensor_dtype(name))

		# Allocate host and device buffers
		host_mem = cuda.pagelocked_empty(size, dtype)
		device_mem = cuda.mem_alloc(host_mem.nbytes)

		if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
			inputs.append({"name": name, "host": host_mem, "device": device_mem})
		else:
			outputs.append({"name": name, "host": host_mem, "device": device_mem})

		bindings.append(int(device_mem))

	context = engine.create_execution_context()
	return engine, context, inputs, outputs, bindings


def run_quantized_inference_for_case(
	engine,
	context,
	inputs,
	outputs,
	bindings,
	criterion: torch.nn.Module,
	base_dir: str,
	case_name: str,
) -> Dict[str, float]:
	"""Run TensorRT (quantized) inference for a single case and compute metrics.

	This mirrors the logic in infer_quantized.py but operates on a provided
	TensorRT engine/context and a specific case name.
	"""

	image_bgr, mask_arr, image_name = load_image_and_mask(base_dir, case_name)
	image_np, mask_np = preprocess(image_bgr, mask_arr)

	# image_np is C,H,W; expand to 1,C,H,W
	image_np = np.expand_dims(image_np, axis=0).astype(np.float32)

	# Target tensor for metrics: 1,C,H,W
	target_t = torch.from_numpy(mask_np).unsqueeze(0)

	# Copy flattened input into the host buffer (assume single input tensor)
	if image_np.size != inputs[0]["host"].size:
		raise RuntimeError(
			f"Engine input buffer size {inputs[0]['host'].size} "
			f"does not match prepared image size {image_np.size}"
		)

	np.copyto(inputs[0]["host"], image_np.ravel())
	cuda.memcpy_htod(inputs[0]["device"], inputs[0]["host"])

	# Ensure tensor addresses are set
	for i in range(engine.num_io_tensors):
		name = engine.get_tensor_name(i)
		context.set_tensor_address(name, bindings[i])

	start_time = time.perf_counter()
	context.execute_v2(bindings=bindings)
	cuda.Context.synchronize()
	end_time = time.perf_counter()
	inference_time = end_time - start_time

	# Transfer output back to host (assume single output tensor)
	cuda.memcpy_dtoh(outputs[0]["host"], outputs[0]["device"])

	output_name = outputs[0]["name"]
	output_shape = engine.get_tensor_shape(output_name)
	output_np = outputs[0]["host"].reshape(*output_shape)

	output_t = torch.from_numpy(output_np)

	loss_val = float(criterion(output_t, target_t).item())
	iou, dice, SE, PC, F1, SP, ACC = iou_score(output_t, target_t)

	metrics = {
		"image_name": image_name,
		"engine_path": TRT_ENGINE_PATH,
		"loss": float(loss_val),
		"iou": float(iou),
		"dice": float(dice),
		"F1": float(F1),
		"accuracy": float(ACC),
		"sensitivity": float(SE),
		"precision": float(PC),
		"specificity": float(SP),
		"inference_time_sec": float(inference_time),
	}
	return metrics


def aggregate_metrics(per_case: List[Dict]) -> Dict[str, Dict[str, float]]:
	"""Compute average metrics across cases for both clean and quantized.

	per_case is a list of dicts of the form::

	    {
	        "case": ...,
	        "clean": {...},
	        "quantized": {...},
	    }
	"""

	num_cases = len(per_case)
	if num_cases == 0:
		return {"num_cases": 0, "clean": {}, "quantized": {}}

	clean_sums = {k: 0.0 for k in METRIC_KEYS}
	quant_sums = {k: 0.0 for k in METRIC_KEYS}

	for item in per_case:
		clean = item["clean"]
		quant = item["quantized"]
		for k in METRIC_KEYS:
			clean_sums[k] += float(clean.get(k, 0.0))
			quant_sums[k] += float(quant.get(k, 0.0))

	clean_avg = {k: v / num_cases for k, v in clean_sums.items()}
	quant_avg = {k: v / num_cases for k, v in quant_sums.items()}

	return {
		"num_cases": num_cases,
		"clean": clean_avg,
		"quantized": quant_avg,
	}


def main() -> None:
	"""Run clean and quantized inference on the full validation set.

	For each case listed in data/busi/busi_val.txt, this script runs:

	- Clean (PyTorch) inference following infer_clean.py
	- Quantized (TensorRT) inference following infer_quantized.py

	It then saves:

	- benchmark_per_case.json: per-case metrics for both clean and quantized
	- benchmark.json: dataset-level average metrics for both clean and quantized
	"""

	cases = load_val_cases(VAL_LIST_PATH)
	print(f"Loaded {len(cases)} validation cases from {VAL_LIST_PATH}")

	# Clean (PyTorch) setup
	device = get_device()
	print(f"Using device for clean inference: {device}")
	model = load_model(CLEAN_CHECKPOINT_PATH, device)
	criterion_clean = losses.BCEDiceLoss().to(device)

	# Quantized (TensorRT) setup
	print(f"Loading TensorRT engine from {TRT_ENGINE_PATH}")
	engine, context, inputs, outputs, bindings = build_trt_engine(TRT_ENGINE_PATH)
	criterion_quant = losses.BCEDiceLoss()  # runs on CPU for TensorRT outputs

	per_case_results: List[Dict] = []

	for idx, case_name in enumerate(cases, start=1):
		print(f"[{idx}/{len(cases)}] Running benchmark for case: {case_name}")

		clean_metrics = run_clean_inference_for_case(
			model=model,
			criterion=criterion_clean,
			device=device,
			base_dir=BASE_DIR,
			case_name=case_name,
		)

		quant_metrics = run_quantized_inference_for_case(
			engine=engine,
			context=context,
			inputs=inputs,
			outputs=outputs,
			bindings=bindings,
			criterion=criterion_quant,
			base_dir=BASE_DIR,
			case_name=case_name,
		)

		per_case_results.append(
			{
				"case": case_name,
				"clean": clean_metrics,
				"quantized": quant_metrics,
			}
		)

	# Save per-case metrics
	os.makedirs(os.path.dirname(BENCHMARK_PER_CASE_PATH), exist_ok=True)
	with open(BENCHMARK_PER_CASE_PATH, "w") as f:
		json.dump(per_case_results, f, indent=4)
	print(f"Per-case benchmark metrics saved to: {BENCHMARK_PER_CASE_PATH}")

	# Save aggregated metrics
	agg = aggregate_metrics(per_case_results)
	with open(BENCHMARK_AGG_PATH, "w") as f:
		json.dump(agg, f, indent=4)
	print(f"Aggregate benchmark metrics saved to: {BENCHMARK_AGG_PATH}")


if __name__ == "__main__":
	main()

