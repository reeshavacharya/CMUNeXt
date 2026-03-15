import os

import modelopt.torch.quantization as mtq
import onnx
import torch
from torch.utils.data import Dataset, DataLoader
from quant_utils import get_device, load_model, load_image_and_mask, preprocess

config = mtq.INT8_DEFAULT_CFG
QUANTIZED_ONNX_NAME = "CMUNeXt_model_busi_quantized.onnx"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_PATH = os.path.join(PROJECT_ROOT, "checkpoint", "quantized", QUANTIZED_ONNX_NAME)


class CalibrationDataset(Dataset):
    """Calibration samples built from images in ../data/busi/calibration/.

    Each item is a single preprocessed image tensor (matching training
    preprocessing) along with a dummy label placeholder.
    """

    def __init__(self, calibration_dir: str):
        self.base_dir = calibration_dir
        images_dir = os.path.join(calibration_dir, "images")
        self.names = []

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"Calibration images directory not found: {images_dir}"
            )

        for fname in os.listdir(images_dir):
            if fname.lower().endswith(".png"):
                self.names.append(os.path.splitext(fname)[0])

        if not self.names:
            raise RuntimeError(f"No calibration images found in {images_dir}")

        self.names.sort()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        case_name = self.names[idx]
        image_np, mask_np, _ = load_image_and_mask(self.base_dir, case_name)
        image_np, _ = preprocess(image_np, mask_np)

        # image_np is C,H,W; DataLoader will add batch dimension to create
        # inputs of shape [B,C,H,W] as expected by the model.
        image_t = torch.from_numpy(image_np)  # C,H,W
        # Second element is a dummy placeholder; mtq only uses inputs
        return image_t, 0


def quantize_model(model, calibration_data_loader):
    def forward_loop(m):
        for image, _ in calibration_data_loader:
            image = image.to(next(m.parameters()).device)
            with torch.no_grad():
                m(image)

    return mtq.quantize(model, config, forward_loop)


# load the pre-trained model from ../checkpoint/CMUNeXt_model_busi_train.pth
device = get_device()
checkpoint_path = os.path.join(
    PROJECT_ROOT, "checkpoint", "CMUNeXt_model_busi_train.pth"
)
model = load_model(checkpoint_path, device)

# calibration data is in ../data/busi/calibration/
calib_dir = os.path.join(PROJECT_ROOT, "data", "busi", "calibration")
calib_dataset = CalibrationDataset(calib_dir)
calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=False)

model = quantize_model(model, calib_loader)
model.eval()
# Dummy input: batch=1, 3-channel RGB, 256x256 (matches img_size in main.py)
dummy_input = torch.randn(1, 3, 256, 256).to(device)

# Export to ONNX
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        EXPORT_PATH,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
        do_constant_folding=True,
    )

print(f"Quantized model saved to: {EXPORT_PATH}")