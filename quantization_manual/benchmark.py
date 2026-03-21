import json
import os

from infer_quantized import main as run_quantized, VAL_LIST_PATH, MODEL_PATH


def benchmark() -> None:
    """Run manual quantized inference over all validation cases and average metrics.

    Writes benchmark.json in this directory with averaged metrics over the
    validation set.
    """

    # Load validation case list
    with open(VAL_LIST_PATH, "r") as f:
        cases = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not cases:
        raise RuntimeError("Validation list is empty; nothing to benchmark.")

    sum_metrics = None
    counted_cases = 0

    for case_name in cases:
        metrics = run_quantized(case_name=case_name)

        if sum_metrics is None:
            # Initialize accumulator for numeric fields
            sum_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    sum_metrics[k] = float(v)
        else:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k in sum_metrics:
                    sum_metrics[k] += float(v)

        counted_cases += 1

    # Compute averages
    avg_metrics = {k: v / counted_cases for k, v in sum_metrics.items()}

    result = {
        "num_cases": counted_cases,
        "checkpoint": MODEL_PATH,
        **avg_metrics,
    }

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Benchmarked {counted_cases} validation cases.")
    print(f"Averaged metrics written to: {out_path}")


if __name__ == "__main__":
    benchmark()
