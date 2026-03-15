"""Create a small calibration dataset from the BUSI training split.

Steps (as per comments):
- From ../data/busi/busi_train.txt, select the top N lines (default: 200)
- Create a folder ../data/busi/calibration
- Copy the corresponding images and masks into that folder

The resulting structure will be:

	data/busi/calibration/
		images/
			<name>.png
		masks/0/
			<name>.png

This mirrors the main dataset layout and is suitable for use as a
calibration dataset for quantization.
"""

import argparse
import os
import shutil


def build_paths() -> tuple[str, str, str]:
	"""Return (base_dir, train_list_path, calibration_dir)."""

	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	base_dir = os.path.join(project_root, "data", "busi")
	train_list_path = os.path.join(base_dir, "busi_train.txt")
	calibration_dir = os.path.join(base_dir, "calibration")
	return base_dir, train_list_path, calibration_dir


def read_top_k_names(train_list_path: str, k: int) -> list[str]:
	"""Read the first k non-empty lines from the training list.

	Each line is assumed to be an image base name (without extension),
	e.g. "benign (1)".
	"""

	if not os.path.exists(train_list_path):
		raise FileNotFoundError(f"Training list not found: {train_list_path}")

	with open(train_list_path, "r") as f:
		lines = [ln.strip() for ln in f.readlines() if ln.strip()]

	if not lines:
		raise RuntimeError("Training list is empty; cannot create calibration dataset.")

	return lines[:k]


def copy_images_and_masks(base_dir: str, calibration_dir: str, names: list[str]) -> None:
	"""Copy selected images and masks into the calibration directory."""

	images_src_dir = os.path.join(base_dir, "images")
	masks_src_dir = os.path.join(base_dir, "masks", "0")

	images_dst_dir = os.path.join(calibration_dir, "images")
	masks_dst_dir = os.path.join(calibration_dir, "masks", "0")

	os.makedirs(images_dst_dir, exist_ok=True)
	os.makedirs(masks_dst_dir, exist_ok=True)

	copied = 0
	for name in names:
		filename = name if name.endswith(".png") else name + ".png"

		src_img = os.path.join(images_src_dir, filename)
		src_mask = os.path.join(masks_src_dir, filename)

		if not os.path.exists(src_img):
			print(f"[WARN] Image not found, skipping: {src_img}")
			continue
		if not os.path.exists(src_mask):
			print(f"[WARN] Mask not found, skipping: {src_mask}")
			continue

		dst_img = os.path.join(images_dst_dir, filename)
		dst_mask = os.path.join(masks_dst_dir, filename)

		shutil.copy2(src_img, dst_img)
		shutil.copy2(src_mask, dst_mask)
		copied += 1

	print(f"Copied {copied} image/mask pairs into calibration dataset.")
	print(f"Calibration images dir: {images_dst_dir}")
	print(f"Calibration masks dir:  {masks_dst_dir}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Create BUSI calibration dataset from training split")
	parser.add_argument(
		"--num_samples",
		type=int,
		default=200,
		help="number of training samples to include in the calibration dataset (default: 200)",
	)
	args = parser.parse_args()

	base_dir, train_list_path, calibration_dir = build_paths()
	print(f"Base dataset directory: {base_dir}")
	print(f"Training list: {train_list_path}")
	print(f"Calibration directory: {calibration_dir}")

	names = read_top_k_names(train_list_path, args.num_samples)
	print(f"Selected {len(names)} names for calibration dataset.")

	copy_images_and_masks(base_dir, calibration_dir, names)


if __name__ == "__main__":
	main()


