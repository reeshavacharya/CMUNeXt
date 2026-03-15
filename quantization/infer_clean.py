"""Single-image inference helper for CMUNeXt.

Implements the following steps:
- Load the model from ../checkpoint/CMUNeXt_model_busi_train.pth
- Pick a random line from ../data/busi/busi_val.txt (image name without extension)
- Load the corresponding image from ../data/busi/images/<name>.png
- Load its mask from ../data/busi/masks/0/<name>.png
- Run inference on GPU using the loaded model
- Compute metrics {dice, F1, accuracy, loss, inference_time, etc.}
- Save metrics to ./inference/inference.json
"""

import os
import sys
import json
import time
import random

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from quantization.quant_utils import get_device, load_image_and_mask, load_model, preprocess
from utils.metrics import iou_score
import utils.losses as losses



def choose_random_case(val_list_path: str) -> str:
	"""Pick a random image name from the validation list.

	The file at val_list_path contains one base name per line (e.g. "benign (1)").
	We will append ".png" when loading the actual image/mask files.
	"""

	if not os.path.exists(val_list_path):
		raise FileNotFoundError(f"Validation list not found: {val_list_path}")

	with open(val_list_path, "r") as f:
		lines = [ln.strip() for ln in f.readlines() if ln.strip()]

	if not lines:
		raise RuntimeError("Validation file is empty; cannot select a sample.")

	name = random.choice(lines)
	return name


def run_inference_once() -> None:
	"""Run inference on a single random validation image and save metrics."""

	device = get_device()

	base_dir = os.path.join(PROJECT_ROOT, "data", "busi")
	val_list_path = os.path.join(base_dir, "busi_val.txt")
	checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoint", "CMUNeXt_model_busi_train.pth")

	model = load_model(checkpoint_path, device)
	criterion = losses.BCEDiceLoss().to(device)

	# Pick random case and load corresponding image + mask
	case_name = choose_random_case(val_list_path)
	image_bgr, mask_arr, image_name = load_image_and_mask(base_dir, case_name)

	image_np, mask_np = preprocess(image_bgr, mask_arr)

	# Convert to tensors
	image_t = torch.from_numpy(image_np).unsqueeze(0).to(device)  # 1,C,H,W
	mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)    # 1,C,H,W

	# Run inference on GPU and time it
	torch.cuda.synchronize(device)
	start = time.perf_counter()
	with torch.no_grad():
		output = model(image_t)
	torch.cuda.synchronize(device)
	end = time.perf_counter()

	inference_time = end - start

	# Compute loss and metrics
	loss_val = criterion(output, mask_t).item()
	iou, dice, SE, PC, F1, SP, ACC = iou_score(output, mask_t)

	metrics = {
		"image_name": image_name,
		"checkpoint": checkpoint_path,
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

	out_dir = os.path.dirname(os.path.abspath(__file__))
	os.makedirs(out_dir, exist_ok=True)

	# Save metrics JSON
	out_path = os.path.join(out_dir, "inference.json")
	with open(out_path, "w") as f:
		json.dump(metrics, f, indent=4)

	# Save predicted mask as a grayscale PNG
	with torch.no_grad():
		prob = torch.sigmoid(output)
	pred_mask = (prob > 0.5).float().cpu().numpy()[0, 0]  # H, W
	pred_mask_img = (pred_mask * 255).astype(np.uint8)
	mask_path = os.path.join(out_dir, "inference.png")
	cv2.imwrite(mask_path, pred_mask_img)

	print(
		f"Inference finished on image '{image_name}'. Metrics saved to: {out_path}, "
		f"mask saved to: {mask_path}"
	)


if __name__ == "__main__":
	run_inference_once()
