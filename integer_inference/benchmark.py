"""Benchmark clean FP32 vs integer inference on malignant validation cases.

This script:
- Reads malignant cases from ../data/busi/busi_val.txt (names starting with
  "malignant").
- Runs clean FP32 inference (infer_clean.run_inference_once) on each case and
  averages the metrics.
- Runs integer inference (integer_inference.run_integer_inference) on the same
  cases and averages the metrics.
- Writes benchmark.json in this directory with the structure:

	{
		"floating-point": {<avg metrics>},
		"integer": {<avg metrics>}
	}

Metrics dictionary keys match the per-run outputs:

	{
		"loss": float,
		"iou": float,
		"dice": float,
		"F1": float,
		"accuracy": float,
		"sensitivity": float,
		"precision": float,
		"specificity": float,
		"inference_time_sec": float,
		...
	}

For the floating-point path we ignore the "checkpoint" and "image_name"
fields when aggregating.
"""

import json
import os
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from infer_clean import run_inference_once as run_clean_inference
from integer_inference import run_integer_inference


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


def _load_malignant_cases(val_list_path: str) -> List[str]:
	if not os.path.exists(val_list_path):
		raise FileNotFoundError(f"Validation list not found: {val_list_path}")

	with open(val_list_path, "r") as f:
		all_cases = [ln.strip() for ln in f.readlines() if ln.strip()]

	malignant = [c for c in all_cases if c.startswith("malignant")]
	if not malignant:
		raise RuntimeError("No malignant cases found in validation list.")
	return malignant


def _init_metric_accumulator() -> Dict[str, float]:
	return {k: 0.0 for k in METRIC_KEYS}


def _avg_metrics(total: Dict[str, float], count: int) -> Dict[str, float]:
	if count == 0:
		return {k: 0.0 for k in METRIC_KEYS}
	return {k: v / count for k, v in total.items()}


def _print_progress(step: int, total_steps: int) -> None:
	if total_steps <= 0:
		return
	percent = (step / total_steps) * 100.0
	msg = f"Progress: {percent:.1f}%"
	print(msg, end="\r", flush=True)


def benchmark() -> None:
	base_dir = os.path.join(PROJECT_ROOT, "data", "busi")
	val_list_path = os.path.join(base_dir, "busi_val.txt")

	cases = _load_malignant_cases(val_list_path)

	total_steps = len(cases) * 2
	step = 0

	# Floating-point (clean) inference benchmark
	fp_totals = _init_metric_accumulator()
	for case in cases:
		# This will write inference.json in the current directory
		run_clean_inference(case_name=case)

		# Read metrics from the JSON file infer_clean.py just wrote
		inf_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.json")
		with open(inf_json_path, "r") as f:
			m = json.load(f)

		for key in METRIC_KEYS:
			fp_totals[key] += float(m[key])

		step += 1
		_print_progress(step, total_steps)

	fp_avg = _avg_metrics(fp_totals, len(cases))

	# Integer inference benchmark
	int_totals = _init_metric_accumulator()
	for case in cases:
		m = run_integer_inference(case_name=case)
		for key in METRIC_KEYS:
			int_totals[key] += float(m[key])

		step += 1
		_print_progress(step, total_steps)

	int_avg = _avg_metrics(int_totals, len(cases))

	out = {
		"floating-point": fp_avg,
		"integer": int_avg,
	}

	out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.json")
	with open(out_path, "w") as f:
		json.dump(out, f, indent=4)


if __name__ == "__main__":
	benchmark()
