import torch.nn as nn
from .residual_block import ResidualBlock


class ResidualSR(nn.Module):
    def __init__(self, scale=2, num_res_blocks=4, channels=64):
        super().__init__()

        self.entry = nn.Conv2d(3, channels, 3, 1, 1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        self.exit = nn.Conv2d(channels, 3 * (scale ** 2), 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.entry(x)
        x = self.res_blocks(x)
        x = self.exit(x)
        x = self.pixel_shuffle(x)
        return x

