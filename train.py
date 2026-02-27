import yaml
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

from models.model_factory import build_model
from data.dataset import SRDataset
from utils.device import get_device
from engine.trainer import Trainer
from engine.checkpoint import save_checkpoint, load_latest_checkpoint
from utils.logger import create_writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()

    dataset = SRDataset(
        hr_folder="data/hr_images",
        patch_size=config["patch_size"],
        scale=config["scale"]
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    model = build_model(config).to(device)

    # Multi-GPU support
    if config.get("multi_gpu", False) and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model)
        print(f"Using {num_gpus} GPUs via DataParallel")
    else:
        print(f"Using single device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Loss function
    if config.get("use_perceptual_loss", False):
        from utils.losses import CombinedLoss
        criterion = CombinedLoss(
            perceptual_weight=config.get("perceptual_weight", 0.006)
        ).to(device)
        print("Using: L1 + VGG Perceptual Loss")
    else:
        criterion = nn.L1Loss()
        print("Using: L1 Loss only")

    # Print model info
    real_model = model.module if hasattr(model, "module") else model
    params = sum(p.numel() for p in real_model.parameters()) / 1e6
    print(f"Model: {config['model_name']} | {params:.2f}M parameters")

    writer = create_writer()

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        config["mixed_precision"],
        writer
    )

    start_epoch = load_latest_checkpoint(
        model, optimizer, "checkpoints"
    )

    try:
        for epoch in range(start_epoch, config["epochs"]):
            loss, avg_psnr = trainer.train_one_epoch(loader, epoch)

            print(f"Epoch {epoch} | Loss {loss:.4f} | PSNR {avg_psnr:.2f}")

            save_checkpoint(
                model,
                optimizer,
                epoch,
                f"checkpoints/sr_epoch_{epoch}.pth"
            )

            # Kaggle safety backup every 50 epochs
            if (epoch + 1) % 50 == 0:
                import shutil
                backup_dir = "/kaggle/working/backup"
                os.makedirs(backup_dir, exist_ok=True)
                src = f"checkpoints/sr_epoch_{epoch}.pth"
                dst = f"{backup_dir}/sr_epoch_{epoch}.pth"
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f"💾 Backup saved: {dst}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        save_checkpoint(
            model,
            optimizer,
            epoch,
            f"checkpoints/interrupted_epoch_{epoch}.pth"
        )


if __name__ == "__main__":
    main()