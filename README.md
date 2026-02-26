# 🔬 ImageUpscaler

A clean, modular, and extensible **image super-resolution** training framework built with PyTorch. Designed for local development with easy migration to cloud platforms like Kaggle.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
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
└── checkpoints/              # Saved model weights
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv AI_lab
AI_lab\Scripts\activate        # Windows
# source AI_lab/bin/activate   # Linux/Mac

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tensorboard pyyaml tqdm pillow
```

### 2. Add Training Data

Place high-resolution images (e.g., [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)) into:

```
data/hr_images/
```

> Images should be larger than the `patch_size` (default: 96px). The dataset class automatically creates low-resolution patches on the fly.

### 3. Configure

Edit `configs/config.yaml`:

```yaml
model_name: residual_sr
scale: 2
num_res_blocks: 4

patch_size: 96
batch_size: 8
epochs: 50
learning_rate: 0.0001

mixed_precision: true
num_workers: 2
```

### 4. Train

```bash
python train.py
```

Training will:
- Auto-detect GPU/CPU
- Auto-resume from the latest checkpoint if one exists
- Save checkpoints after every epoch to `checkpoints/`
- Log metrics to TensorBoard under `runs/`

### 5. Monitor with TensorBoard

```bash
tensorboard --logdir=runs
```

Then open [http://localhost:6006](http://localhost:6006) to view live loss and PSNR curves.

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

## 🛑 Interrupting Training

Press `Ctrl+C` at any time. The current model state will be saved as:

```
checkpoints/interrupted_epoch_X.pth
```

Restart with `python train.py` and it will auto-resume from the latest checkpoint.

---

## 📊 Metrics

| Metric | Description |
|--------|-------------|
| **L1 Loss** | Per-pixel absolute error (training objective) |
| **PSNR** | Peak Signal-to-Noise Ratio (evaluation quality) |

---

## 🗺️ Roadmap

- [ ] Inference script for upscaling arbitrary images
- [ ] SSIM metric
- [ ] Learning rate scheduler
- [ ] Sample image logging to TensorBoard
- [ ] ESRGAN / SwinIR architecture support
- [ ] Kaggle notebook adapter

---

## 📄 License

MIT License — free for personal and commercial use.
