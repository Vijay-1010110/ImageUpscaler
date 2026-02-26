# 🔬 ImageUpscaler

A clean, modular, and extensible **image super-resolution** training framework built with PyTorch. Designed for local development with easy migration to cloud platforms like Kaggle.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **Modular architecture** — swap models with a single config change
- **Mixed precision training** — faster training via PyTorch AMP
- **TensorBoard integration** — live loss & PSNR curves
- **Auto-resume** — automatically resumes from the latest checkpoint
- **Graceful interrupt** — `Ctrl+C` saves a checkpoint before exiting
- **Hardware-agnostic** — runs on CUDA GPU or CPU seamlessly
- **Kaggle-ready** — clean structure that ports directly to notebooks

---

## 📁 Project Structure

```
ImageUpscaler/
├── configs/
│   └── config.yaml          # All hyperparameters in one place
├── models/
│   ├── __init__.py
│   ├── residual_block.py     # Core residual block
│   ├── sr_residual.py        # Super-resolution network
│   └── model_factory.py      # Architecture switcher
├── data/
│   ├── dataset.py            # LR/HR patch pair generator
│   └── hr_images/            # Place training images here
├── engine/
│   ├── trainer.py            # Training loop with AMP + logging
│   └── checkpoint.py         # Save / auto-resume checkpoints
├── utils/
│   ├── metrics.py            # PSNR metric
│   ├── device.py             # Auto GPU/CPU detection
│   └── logger.py             # TensorBoard writer
├── train.py                  # Main training entry point
├── inference.py              # Upscale images with a trained model
├── requirements.txt          # Python dependencies
└── checkpoints/              # Saved model weights
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Vijay-1010110/ImageUpscaler.git
cd ImageUpscaler

# Create and activate virtual environment
python -m venv AI_lab
AI_lab\Scripts\activate        # Windows
# source AI_lab/bin/activate   # Linux/Mac

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Add Training Data

Download the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and place high-resolution images into:

```
data/hr_images/
```

> Images must be larger than `patch_size` (default: 96px). The dataset class automatically creates low-resolution patches on the fly.

### 3. Configure

Edit `configs/config.yaml` to tune hyperparameters:

```yaml
model_name: residual_sr
scale: 2                # Upscale factor (2x)
num_res_blocks: 4

patch_size: 96           # Training patch size
batch_size: 8
epochs: 50
learning_rate: 0.0001

mixed_precision: true    # FP16 training for speed
num_workers: 2
```

---

## 🏋️ Training

```bash
python train.py
```

**What happens:**
- Auto-detects GPU (CUDA) or falls back to CPU
- Auto-resumes from the latest checkpoint in `checkpoints/`
- Saves a checkpoint after every epoch
- Logs loss and PSNR to TensorBoard

**Monitor training live:**
```bash
tensorboard --logdir=runs
# Open http://localhost:6006
```

**Interrupt safely:** Press `Ctrl+C` — it will save a checkpoint before exiting.

---

## 🖼️ Inference (Upscale Images)

After training, upscale any image:

```bash
python inference.py --input path/to/low_res.png --output path/to/upscaled.png --checkpoint checkpoints/sr_epoch_49.pth
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | ✅ | Path to input low-resolution image |
| `--output` | ✅ | Path to save the upscaled result |
| `--checkpoint` | ✅ | Path to a trained `.pth` checkpoint |
| `--config` | ❌ | Config file (default: `configs/config.yaml`) |

**Example:**
```bash
# Upscale a single image using the best checkpoint
python inference.py --input test_images/photo.jpg --output results/photo_2x.png --checkpoint checkpoints/sr_epoch_49.pth
```

---

## 🔄 Switching Architectures

Adding a new model is simple:

1. Create your model class in `models/`
2. Register it in `models/model_factory.py`:

```python
from .your_model import YourModel

def build_model(config):
    if config["model_name"] == "residual_sr":
        return ResidualSR(...)
    elif config["model_name"] == "your_model":
        return YourModel(...)
```

3. Update `config.yaml`:

```yaml
model_name: your_model
```

---

## 📊 Metrics

| Metric | Description |
|--------|-------------|
| **L1 Loss** | Per-pixel absolute error (training objective) |
| **PSNR** | Peak Signal-to-Noise Ratio (evaluation quality) |

---

## 🗺️ Roadmap

- [x] Core training pipeline
- [x] Mixed precision (AMP)
- [x] TensorBoard visualization
- [x] Auto-resume from checkpoints
- [x] Graceful `Ctrl+C` save
- [x] Inference script
- [ ] SSIM metric
- [ ] Learning rate scheduler
- [ ] Sample image logging to TensorBoard
- [ ] ESRGAN / SwinIR architecture support
- [ ] Kaggle notebook adapter

---

## 📄 License

MIT License — free for personal and commercial use.
