import torch


def psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    return 10 * torch.log10(1.0 / mse)
