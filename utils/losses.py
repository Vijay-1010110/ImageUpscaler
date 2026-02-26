import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG19-based perceptual loss for sharper super-resolution."""

    def __init__(self):
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # Use features up to relu3_4 (layer 16) — good balance of detail + structure
        self.feature_extractor = nn.Sequential(
            *list(vgg.features[:16])
        ).eval()

        # Freeze VGG weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.l1 = nn.L1Loss()

    def forward(self, sr, hr):
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return self.l1(sr_features, hr_features)


class CombinedLoss(nn.Module):
    """L1 pixel loss + VGG perceptual loss."""

    def __init__(self, perceptual_weight=0.006):
        super().__init__()
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.perceptual_weight = perceptual_weight

    def forward(self, sr, hr):
        pixel = self.pixel_loss(sr, hr)
        perceptual = self.perceptual_loss(sr, hr)
        return pixel + self.perceptual_weight * perceptual
