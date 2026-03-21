import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from quant_utils import get_device, load_model, load_image_and_mask, preprocess
from torch.utils.data import Dataset, DataLoader

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoint", "CMUNeXt_model_busi_train.pth")
CALIB_DIR = os.path.join(PROJECT_ROOT, "data", "busi", "calibration")
OUTPUT_JSON = "calibration.json"

# Dictionary to store global absolute maximums
# Structure: { "layer_name": tensor_of_max_abs_per_channel }
act_stats = {}

def get_per_channel_max(tensor, is_weight=False):
    """
    Calculates the max absolute value per channel.
    Activations [B, C, H, W] -> Reduce (0, 2, 3) to get [C] scales.
    Weights [OC, IC, KH, KW] -> Reduce (1, 2, 3) to get [OC] scales.
    """
    t_abs = torch.abs(tensor).detach().cpu()
    
    if len(t_abs.shape) == 4:
        if is_weight:
            # For Weights: Keep Out_Channels (Dim 0)
            return torch.amax(t_abs, dim=(1, 2, 3))
        else:
            # For Activations: Keep Channels (Dim 1)
            return torch.amax(t_abs, dim=(0, 2, 3))
            
    elif len(t_abs.shape) == 2:
        # Linear Layers [Out, In] -> Reduce Dim 1 to get [Out]
        if is_weight:
            return torch.amax(t_abs, dim=1)
        else:
            return torch.amax(t_abs, dim=0)
    
    return torch.amax(t_abs)

def calibration_hook(name):
    def hook(module, input, output):
        m = get_per_channel_max(output, is_weight=False)
        if name not in act_stats:
            act_stats[name] = m
        else:
            act_stats[name] = torch.max(act_stats[name], m)
    return hook

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


def run_manual_calibration():
    device = get_device()
    model = load_model(CHECKPOINT_PATH, device)
    model.eval()

    # 1. Prepare Dataset
    calib_dataset = CalibrationDataset(CALIB_DIR)
    calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=False)

    # 2. Register Hooks for EVERY layer (No skipping)
    # We target every module that isn't a container (Sequential, CMUNeXtBlock, etc.)
    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(calibration_hook(name)))

    # 3. Calibration Pass
    print(f"Starting calibration on {len(calib_dataset)} images...")
    with torch.no_grad():
        for images, _ in tqdm(calib_loader):
            model(images.to(device))

    # 4. Process Results into Scale Points
    calibration_results = {}

    # A. Process Activations (from Hooks)
    for name, max_val_tensor in act_stats.items():
        max_t = max_val_tensor.tolist()
        # Scale = max(|T|) / 127
        if isinstance(max_t, list):
            scale = [v / 127.0 for v in max_t]
        else:
            scale = max_t / 127.0
            
        calibration_results[name] = {
            "type": "activation",
            "max_T": max_t,
            "scale_fp32": scale
        }

    # B. Process Weights (Static)
    print("Calculating weight scales...")
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            w_max = get_per_channel_max(module.weight.data, is_weight=True).tolist()
            if isinstance(w_max, list):
                w_scale = [v / 127.0 for v in w_max]
            else:
                w_scale = w_max / 127.0
                
            calibration_results[f"{name}.weight"] = {
                "type": "weight",
                "max_T": w_max,
                "scale_fp32": w_scale
            }

    # 5. Add scale for the raw input (the images themselves)
    # We can calculate this from the last batch processed or track it in the loop
    # For BUSI images preprocessed to [0, 1] or [-1, 1], this is crucial
    input_max = 0
    with torch.no_grad():
        for images, _ in calib_loader:
            m = torch.max(torch.abs(images)).item()
            input_max = max(input_max, m)
    
    calibration_results["input"] = {
        "type": "activation",
        "max_T": input_max,
        "scale_fp32": input_max / 127.0
    }

    # 6. Save to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(calibration_results, f, indent=4)
    
    print(f"Calibration complete. Results saved to {OUTPUT_JSON}")
    
    # Cleanup
    for h in handles:
        h.remove()

if __name__ == "__main__":
    run_manual_calibration()