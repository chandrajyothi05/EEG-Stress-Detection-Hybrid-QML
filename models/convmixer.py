import torch
import torch.nn as nn


class ConvMixerBlock(nn.Module):
    """One ConvMixer block: depthwise conv (residual) + pointwise conv."""

    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim,
                      padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = x + self.depthwise(x)  # residual around depthwise only, per ConvMixer paper
        x = self.pointwise(x)
        return x


class ConvMixer(nn.Module):
    def __init__(
        self,
        in_channels: int = 19,
        embed_dim: int = 128,
        depth: int = 6,
        patch_size: tuple = (5, 25),
        kernel_size: int = 5,
        n_classes: int = 2,
    ):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim),
        )

        self.blocks = nn.Sequential(
            *[ConvMixerBlock(embed_dim, kernel_size) for _ in range(depth)]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)      # (B, 128, 8, 40)
        x = self.blocks(x)           # (B, 128, 8, 40)
        x = self.pool(x)             # (B, 128, 1, 1)
        x = x.flatten(1)             # (B, 128)  -- this is the "feature vector" for later fusion
        logits = self.classifier(x)  # (B, 2)
        return logits, x  # return both logits AND the feature vector (needed for Feature Fusion later)


if __name__ == "__main__":
    model = ConvMixer()
    dummy = torch.randn(4, 19, 40, 1000)  # batch of 4
    logits, features = model(dummy)
    print(f"Logits shape: {logits.shape}")     # expect (4, 2)
    print(f"Feature vector shape: {features.shape}")  # expect (4, 128)

