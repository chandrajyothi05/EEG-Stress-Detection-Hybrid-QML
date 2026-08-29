import torch
import torch.nn as nn


class ConvMixerBlock(nn.Module):
    """One ConvMixer block: depthwise conv (residual) + pointwise conv."""

    def __init__(self, dim: int, kernel_size: int = 5, drop_rate: float = 0.1):
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
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = x + self.depthwise(x)
        x = self.pointwise(x)
        x = self.dropout(x)
        return x


class ConvMixer(nn.Module):
    def __init__(
        self,
        in_channels: int = 19,
        embed_dim: int = 64,      # was 128 -- halved to reduce capacity relative to ~5k training samples
        depth: int = 3,           # was 6 -- halved for the same reason
        patch_size: tuple = (5, 25),
        kernel_size: int = 5,
        n_classes: int = 2,
        block_drop_rate: float = 0.1,
        classifier_drop_rate: float = 0.4,
    ):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim),
        )

        self.blocks = nn.Sequential(
            *[ConvMixerBlock(embed_dim, kernel_size, block_drop_rate) for _ in range(depth)]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_dropout = nn.Dropout(classifier_drop_rate)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)      # (B, 64, 8, 40)
        x = self.blocks(x)           # (B, 64, 8, 40)
        x = self.pool(x)             # (B, 64, 1, 1)
        x = x.flatten(1)             # (B, 64)  -- note: feature vector is now 64-dim, not 128
        logits = self.classifier(self.classifier_dropout(x))  # (B, 2)
        return logits, x


if __name__ == "__main__":
    model = ConvMixer()
    dummy = torch.randn(4, 19, 40, 1000)
    logits, features = model(dummy)
    print(f"Logits shape: {logits.shape}")            # expect (4, 2)
    print(f"Feature vector shape: {features.shape}")  # expect (4, 64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")