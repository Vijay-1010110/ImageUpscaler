import torch.nn as nn
import torch.nn.functional as F
from .residual_block import ResidualBlock


class ResidualSR(nn.Module):
    def __init__(self, scale=2, num_res_blocks=4, channels=64):
        super().__init__()
        self.scale = scale

        self.entry = nn.Conv2d(3, channels, 3, 1, 1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        self.exit = nn.Conv2d(channels, 3 * (scale ** 2), 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        # Global skip: bicubic upscale of input
        bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # Feature extraction + residual learning
        feat = self.entry(x)
        res = self.res_blocks(feat)
        res = res + feat              # Feature-level skip
        res = self.exit(res)
        res = self.pixel_shuffle(res)

        return bicubic + res          # Image-level skip
