# 🚀 Training on Kaggle (T4 x2)

## Quick Start

### 1. Create Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **Settings** → Accelerator → **GPU T4 x2**
3. **Settings** → Internet → **ON**

### 2. Add DIV2K Dataset

1. Click **Add Data** (right sidebar)
2. Search **"DIV2K"**
3. Add a dataset that has `DIV2K_train_HR` folder (800 images)

### 3. Run Training

Paste the contents of `kaggle_train.py` into cells and run.

Or run directly:
```python
!git clone -b kaggle https://github.com/Vijay-1010110/ImageUpscaler.git
%cd ImageUpscaler
!pip install -q -r requirements.txt
!python train.py --config configs/kaggle_config.yaml
```

### 4. Download Checkpoint

After training, the best checkpoint is saved to `/kaggle/working/`.
Download it from the **Output** tab.

### 5. Use Locally

Copy the downloaded `.pth` file to your local `checkpoints/` folder:
```bash
python inference.py --input photo.jpg --output result.png --checkpoint checkpoints/sr_epoch_299.pth
```

Or use the web UI:
```bash
python app.py
# Open http://localhost:5000
```

## Kaggle Config vs Local Config

| Setting | Local (GTX 1650, 4GB) | Kaggle (T4 x2, 30GB) |
|---|---|---|
| Channels | 128 | 256 |
| Res Blocks | 16 | 32 |
| Batch Size | 32 | 64 |
| Patch Size | 192 | 256 |
| Epochs | 200 | 300 |
| Multi-GPU | No | Yes (DataParallel) |
| Model Size | ~4.7M | ~18.9M params |

## Troubleshooting

- **OOM Error**: Reduce `batch_size` to 32 in `kaggle_config.yaml`
- **Dataset not found**: Check the path printed in the error and update `KAGGLE_DATASET_DIR` in the script
- **Slow training**: Make sure GPU T4 x2 is selected (not CPU or P100)
