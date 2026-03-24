import os
import sys

import cv2
import numpy as np
import torch

# Ensure project root (containing the `network` package) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from network.CMUNeXt import cmunext


def get_device() -> torch.device:
	"""Return CUDA device if available, otherwise fall back to CPU.

	This makes helper code usable in CPU-only environments (e.g., for
	parameter recording) without requiring a working CUDA/NCCL stack.
	"""

	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


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
	"""Apply the same preprocessing used during training/validation.

	Primary path uses albumentations; if that stack is unavailable or
	broken (e.g., skimage/numpy mismatch), fall back to a simple
	OpenCV-based resize + normalization so integer_inference and other
	CPU-only utilities still work.
	"""

	img_size = 256

	try:
		from albumentations.augmentations import transforms
		from albumentations.core.composition import Compose
		from albumentations import Resize

		transform = Compose(
			[
				Resize(img_size, img_size),
				transforms.Normalize(),
			]
		)

		augmented = transform(image=image, mask=mask)
		image = augmented["image"]
		mask = augmented["mask"]

	except Exception:
		# Fallback: manual resize & normalize using OpenCV.
		image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
		mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
		# Ensure mask has a channel dimension (H, W, 1) like the
		# load_image_and_mask path above.
		if mask.ndim == 2:
			mask = mask[..., None]

	# Common post-processing
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

