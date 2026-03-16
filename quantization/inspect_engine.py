import numpy as np
import json
import os

# --- Configuration ---
RAW_FILE = "./tensor_dumps/output.raw" 
OUTPUT_JSON = "./int8_activation_log.json"
TENSOR_SHAPE = (1, 3, 256, 256) 

def log_int8_to_json(file_path, output_path, shape):
    # 1. Load the raw bytes
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    data = np.fromfile(file_path, dtype=np.int8)
    
    # 2. Calculate Distribution (Counts for each integer -128 to 127)
    # This creates a frequency map of the 256 possible values
    counts, bins = np.histogram(data, bins=256, range=(-128, 128))
    distribution = {int(bin_val): int(count) for bin_val, count in zip(range(-128, 128), counts)}

    # 3. Calculate Key Statistics
    total_elements = len(data)
    neg_clips = int(np.sum(data == -128))
    pos_clips = int(np.sum(data == 127))
    zeros = int(np.sum(data == 0))

    log_data = {
        "metadata": {
            "file_source": os.path.basename(file_path),
            "tensor_shape": list(shape),
            "total_elements": total_elements
        },
        "statistics": {
            "min": int(data.min()),
            "max": int(data.max()),
            "mean": float(data.mean()),
            "std_dev": float(data.std()),
            "zeros_count": zeros,
            "zeros_percentage": float(zeros / total_elements * 100)
        },
        "quantization_artifacts": {
            "lower_saturation_limit": -128,
            "upper_saturation_limit": 127,
            "lower_clipping_count": neg_clips,
            "upper_clipping_count": pos_clips,
            "clipping_percentage_total": float((neg_clips + pos_clips) / total_elements * 100)
        },
        "full_distribution": distribution
    }

    # 4. Save to JSON
    with open(output_path, "w") as f:
        json.dump(log_data, f, indent=4)
    
    print(f"Quantization log saved successfully to: {output_path}")

if __name__ == "__main__":
    log_int8_to_json(RAW_FILE, OUTPUT_JSON, TENSOR_SHAPE)