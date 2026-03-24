import os
import sys

import cv2
import numpy as np
import torch
import albumentations as A

# Ensure project root (containing the `network` package) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from network.CMUNeXt import cmunext


def get_device() -> torch.device:
	"""Return CUDA device; raise if not available (inference must run on GPU)."""

	if not torch.cuda.is_available():
		raise RuntimeError("CUDA device is required for inference, but none is available.")
	return torch.device("cuda")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
	"""Instantiate CMUNeXt and load trained weights from checkpoint_path."""

	model = cmunext().to(device)
	model.eval()

	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	state_dict = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(state_dict)
	return model

def preprocess(image: np.ndarray, mask: np.ndarray):
	"""Apply the same preprocessing used during training/validation."""

	img_size = 256
	transform = A.Compose(
		[
			A.Resize(img_size, img_size),
			A.Normalize(),
		]
	)

	augmented = transform(image=image, mask=mask)
	image = augmented["image"]
	mask = augmented["mask"]

	image = image.astype("float32") / 255.0
	image = image.transpose(2, 0, 1)  # C, H, W

	mask = mask.astype("float32") / 255.0
	mask = mask.transpose(2, 0, 1)  # C, H, W

	return image, mask


def load_image_and_mask(base_dir: str, case_name: str):
	"""Load image and mask arrays for a given case name (without extension)."""

	image_name = case_name
	if not image_name.endswith(".png"):
		image_name = image_name + ".png"

	img_path = os.path.join(base_dir, "images", image_name)
	mask_path = os.path.join(base_dir, "masks", "0", image_name)

	image = cv2.imread(img_path)
	if image is None:
		raise FileNotFoundError(f"Image not found: {img_path}")

	mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
	if mask is None:
		raise FileNotFoundError(f"Mask not found: {mask_path}")

	mask = mask[..., None]  # H, W, 1
	return image, mask, image_name

