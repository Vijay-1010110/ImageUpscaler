"""
ImageUpscaler — Kaggle Training Script
=======================================
Paste this into a Kaggle notebook (Python) with GPU T4 x2 enabled.

Setup:
1. Create new Kaggle Notebook
2. Settings → Accelerator → GPU T4 x2
3. Settings → Internet → ON
4. Add Dataset: search "DIV2K" or upload your own HR images
5. Paste this entire script into a code cell and run
"""

# ============================================================
# CELL 1: Setup — Clone repo and install dependencies
# ============================================================
import os

# Clone the repo
if not os.path.exists("/kaggle/working/ImageUpscaler"):
    os.system("git clone -b kaggle https://github.com/Vijay-1010110/ImageUpscaler.git /kaggle/working/ImageUpscaler")

os.chdir("/kaggle/working/ImageUpscaler")

# Install requirements
os.system("pip install -q -r requirements.txt")

print("✅ Setup complete!")

# ============================================================
# CELL 2: Prepare dataset
# ============================================================
import shutil

# Link DIV2K dataset from Kaggle datasets
# Update this path based on how you added the dataset
KAGGLE_DATASET_DIR = "/kaggle/input/div2k-dataset/DIV2K_train_HR"

# Alternative common paths on Kaggle:
ALT_PATHS = [
    "/kaggle/input/div2k-dataset/DIV2K_train_HR",
    "/kaggle/input/div2k/DIV2K_train_HR",
    "/kaggle/input/div2k-images/DIV2K_train_HR",
    "/kaggle/input/div2k-dataset/train",
]

hr_folder = "data/hr_images"
os.makedirs(hr_folder, exist_ok=True)

# Find the dataset
dataset_found = False
for path in [KAGGLE_DATASET_DIR] + ALT_PATHS:
    if os.path.exists(path):
        # Symlink images to avoid copying
        for img in os.listdir(path):
            src = os.path.join(path, img)
            dst = os.path.join(hr_folder, img)
            if not os.path.exists(dst):
                os.symlink(src, dst)
        dataset_found = True
        num_images = len(os.listdir(hr_folder))
        print(f"✅ Linked {num_images} images from {path}")
        break

if not dataset_found:
    print("❌ DIV2K dataset not found! Please add it as a Kaggle dataset.")
    print("   Go to: Add Data → Search 'DIV2K' → Add")
    print(f"   Searched paths: {[KAGGLE_DATASET_DIR] + ALT_PATHS}")

# ============================================================
# CELL 3: Check GPU setup
# ============================================================
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    vram = props.total_memory / (1024**3)
    print(f"  GPU {i}: {props.name} ({vram:.1f} GB)")

# ============================================================
# CELL 4: Train!
# ============================================================
os.system("python train.py --config configs/kaggle_config.yaml")

# ============================================================
# CELL 5: Copy best checkpoint to Kaggle output
# ============================================================
import re

checkpoint_dir = "checkpoints"
files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

if files:
    def epoch_num(fname):
        m = re.search(r'(\d+)', fname)
        return int(m.group(1)) if m else -1
    files.sort(key=epoch_num)
    latest = files[-1]

    output_dir = "/kaggle/working"
    src = os.path.join(checkpoint_dir, latest)
    dst = os.path.join(output_dir, f"best_checkpoint_{latest}")
    shutil.copy2(src, dst)
    print(f"✅ Saved to output: {dst}")
    print(f"   Download from Kaggle Output tab")
else:
    print("❌ No checkpoints found")
