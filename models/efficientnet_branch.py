"""
models/efficientnet_branch.py

Pretrained EfficientNet-B0 for the Azimuthal Projection branch.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetBranch(nn.Module):
    def __init__(self, n_classes: int = 2, freeze_backbone: bool = False):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        # EfficientNet-B0's classifier is Sequential(Dropout, Linear(1280, 1000))
        # Replace with identity to get the 1280-dim feature vector instead
        self.feature_dim = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x):
        features = self.backbone(x)       # (B, 1280)
        logits = self.classifier(features)
        return logits, features           # matches ConvMixer's (logits, features) convention


if __name__ == "__main__":
    model = EfficientNetBranch()
    dummy = torch.randn(4, 3, 224, 224)
    logits, features = model(dummy)
    print(f"Logits shape: {logits.shape}")     # expect (4, 2)
    print(f"Feature vector shape: {features.shape}")  # expect (4, 1280)