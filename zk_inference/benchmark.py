import json
import os
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantization.quant_utils import get_device, load_image_and_mask, load_model, preprocess
from utils.metrics import iou_score
import utils.losses as losses

from fixed_point_inference import run_fixed_point_inference
from record import record


VAL_LIST_PATH = os.path.join(PROJECT_ROOT, "data", "busi", "busi_val.txt")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoint", "CMUNeXt_model_busi_train.pth")
BASE_DIR = os.path.join(PROJECT_ROOT, "data", "busi")

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


def _load_val_cases() -> List[str]:
    if not os.path.exists(VAL_LIST_PATH):
        raise FileNotFoundError(f"Validation list not found: {VAL_LIST_PATH}")
    with open(VAL_LIST_PATH, "r") as f:
        cases = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not cases:
        raise RuntimeError("Validation file is empty; cannot benchmark.")
    return cases


def benchmark_clean(cases: List[str], progress_cb=None) -> Dict[str, float]:
    """Run FP32 CMUNeXt inference over all validation cases and average metrics.

    If progress_cb is provided, it is called once per case.
    """

    device = get_device()
    model = load_model(CHECKPOINT_PATH, device)
    criterion = losses.BCEDiceLoss().to(device)

    # Lazy imports to avoid circular imports at module load time
    import torch
    import time

    sums = {k: 0.0 for k in METRIC_KEYS}

    for case_name in cases:
        image_bgr, mask_arr, image_name = load_image_and_mask(BASE_DIR, case_name)
        image_np, mask_np = preprocess(image_bgr, mask_arr)

        image_t = torch.from_numpy(image_np).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model(image_t)
        torch.cuda.synchronize(device)
        end = time.perf_counter()

        inference_time = end - start

        loss_val = criterion(output, mask_t).item()
        iou, dice, SE, PC, F1, SP, ACC = iou_score(output, mask_t)

        per_case = {
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

        for k in METRIC_KEYS:
            sums[k] += per_case[k]

        if progress_cb is not None:
            progress_cb()

    n = float(len(cases))
    return {k: v / n for k, v in sums.items()}


def benchmark_fixed_point(cases: List[str], exponents: List[int], progress_cb=None) -> Dict[str, Dict[str, float]]:
    """Run fixed-point inference for each exponent over all validation cases.

    If progress_cb is provided, it is called once per case.
    """

    results: Dict[str, Dict[str, float]] = {}

    for exp in exponents:
        print(f"Running fixed-point benchmark for exp={exp}...")

        # Re-generate fixed-point parameters for this exponent
        record(exp=exp)

        sums = {k: 0.0 for k in METRIC_KEYS}

        for case_name in cases:
            metrics = run_fixed_point_inference(case_name=case_name, exp=exp)

            for k in METRIC_KEYS:
                sums[k] += float(metrics[k])

            if progress_cb is not None:
                progress_cb()

        n = float(len(cases))
        results[str(exp)] = {k: v / n for k, v in sums.items()}

    return results


def main() -> None:
    cases = _load_val_cases()

    # Overall progress tracking across clean + all fixed-point runs
    exponents = [4, 6, 8, 12, 16]
    total_steps = len(cases) * (1 + len(exponents))
    completed = 0

    def progress_cb() -> None:
        nonlocal completed
        completed += 1
        pct = (completed / total_steps) * 100.0
        # Single-line progress log updated in place
        print(f"\rProgress: {pct:6.2f}% ({completed}/{total_steps})", end="", flush=True)

    # Clean FP32 baseline
    clean_avg = benchmark_clean(cases, progress_cb=progress_cb)

    # Fixed-point for requested exponents
    fixed_point_avgs = benchmark_fixed_point(cases, exponents, progress_cb=progress_cb)

    benchmark_result = {
        "clean": clean_avg,
        "fixed_point": fixed_point_avgs,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(benchmark_result, f, indent=4)

    # Finish the progress line and print summary
    print()  # newline after progress
    print(f"Benchmark completed over {len(cases)} validation cases.")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()