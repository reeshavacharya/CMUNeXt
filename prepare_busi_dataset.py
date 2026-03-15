import os
import glob
import shutil


def prepare_busi_dataset(
    src_root: str = "./data/Dataset_BUSI_with_GT",
    dst_root: str = "./data/busi",
) -> None:
    """Reorganize the BUSI dataset into the layout expected by split.py and dataset.py.

    Expected final structure:
        dst_root/
            images/      # all images (benign, malignant, normal)
            masks/0/     # corresponding masks with the SAME base name as images

    For each image like "benign (1).png" we look for a mask file
    "benign (1)_mask*.png" in the source class folder and copy it as
    "dst_root/masks/0/benign (1).png".
    """

    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)

    images_dir = os.path.join(dst_root, "images")
    masks_dir = os.path.join(dst_root, "masks", "0")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    classes = ["benign", "malignant", "normal"]

    num_images = 0
    num_masks_missing = 0

    for cls in classes:
        cls_dir = os.path.join(src_root, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Class directory not found: {cls_dir}")
            continue

        # All PNGs in this class folder
        all_pngs = glob.glob(os.path.join(cls_dir, "*.png"))

        # Treat files WITHOUT "_mask" in the name as images
        image_files = [p for p in all_pngs if "_mask" not in os.path.basename(p)]

        print(f"Processing class '{cls}' with {len(image_files)} images...")

        for img_path in sorted(image_files):
            base = os.path.splitext(os.path.basename(img_path))[0]  # e.g. "benign (1)"

            # Destination image path (keep the original name)
            dst_img_path = os.path.join(images_dir, f"{base}.png")

            # Find corresponding mask(s): base + "_mask*.png"
            mask_pattern = os.path.join(cls_dir, f"{base}_mask*.png")
            mask_candidates = sorted(glob.glob(mask_pattern))

            if not mask_candidates:
                print(f"[WARN] No mask found for image '{img_path}'")
                num_masks_missing += 1
                continue

            # Use the first mask candidate
            mask_path = mask_candidates[0]

            # Destination mask path uses the SAME base name as the image
            dst_mask_path = os.path.join(masks_dir, f"{base}.png")

            # Copy image and mask; do not modify originals
            shutil.copy2(img_path, dst_img_path)
            shutil.copy2(mask_path, dst_mask_path)

            num_images += 1

    print("\nDone.")
    print(f"Total images prepared: {num_images}")
    if num_masks_missing > 0:
        print(f"Images without masks: {num_masks_missing} (see warnings above)")
    print(f"Images dir: {images_dir}")
    print(f"Masks dir:  {masks_dir}")


if __name__ == "__main__":
    prepare_busi_dataset()
