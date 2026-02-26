import torch.nn as nn
from .residual_block import ResidualBlock


class ResidualSR(nn.Module):
    def __init__(self, scale=2, num_res_blocks=4):
        super().__init__()

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        self.exit = nn.Conv2d(64, 3 * (scale ** 2), 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.entry(x)
        x = self.res_blocks(x)
        x = self.exit(x)
        x = self.pixel_shuffle(x)
        return x
