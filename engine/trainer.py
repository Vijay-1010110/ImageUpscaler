import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from utils.metrics import psnr
from utils.system_monitor import get_system_metrics


class Trainer:
    def __init__(self, model, optimizer, criterion,
                 device, mixed_precision, writer=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler('cuda', enabled=mixed_precision)
        self.writer = writer

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        total_psnr = 0

        for step, (lr, hr) in enumerate(tqdm(loader)):
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.mixed_precision):
                sr = self.model(lr)
                loss = self.criterion(sr, hr)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_psnr += psnr(sr, hr).item()

            if self.writer:
                global_step = epoch * len(loader) + step
                self.writer.add_scalar("Loss/step", loss.item(), global_step)

                # Log system metrics every 10 steps
                if step % 10 == 0:
                    sys_metrics = get_system_metrics()
                    for key, val in sys_metrics.items():
                        self.writer.add_scalar(key, val, global_step)

        avg_loss = total_loss / len(loader)
        avg_psnr = total_psnr / len(loader)

        if self.writer:
            self.writer.add_scalar("Loss/epoch", avg_loss, epoch)
            self.writer.add_scalar("PSNR/epoch", avg_psnr, epoch)

        return avg_loss, avg_psnr
