"""
models/feature_fusion.py

Feature Fusion + Feature Reduction stages, sitting between the two
parallel branches (ConvMixer: 64-dim, EfficientNet: 1280-dim) and the
downstream Bi-LSTM.

Fusion: project each branch's features into a shared 128-dim space,
then concatenate -> 256-dim. Symmetric projection (rather than projecting
1280 down to 64) so neither branch loses disproportionately more
information than the other.

Reduction: compress the 256-dim fused vector down to 128-dim before
it becomes one timestep of the Bi-LSTM's input sequence.
"""

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    def __init__(self, convmixer_dim: int = 64, efficientnet_dim: int = 1280, shared_dim: int = 128):
        super().__init__()
        self.proj_convmixer = nn.Linear(convmixer_dim, shared_dim)
        self.proj_efficientnet = nn.Linear(efficientnet_dim, shared_dim)
        self.fused_dim = shared_dim * 2  # 256

    def forward(self, feat_convmixer: torch.Tensor, feat_efficientnet: torch.Tensor) -> torch.Tensor:
        c = self.proj_convmixer(feat_convmixer)          # (B, 128)
        e = self.proj_efficientnet(feat_efficientnet)     # (B, 128)
        fused = torch.cat([c, e], dim=1)                  # (B, 256)
        return fused


class FeatureReduction(nn.Module):
    def __init__(self, in_dim: int = 256, out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionReductionBlock(nn.Module):
    """Convenience wrapper: fusion + reduction in one call, for use inside
    the extraction script or later end-to-end training."""

    def __init__(self, convmixer_dim: int = 64, efficientnet_dim: int = 1280,
                 shared_dim: int = 128, reduced_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.fusion = FeatureFusion(convmixer_dim, efficientnet_dim, shared_dim)
        self.reduction = FeatureReduction(self.fusion.fused_dim, reduced_dim, dropout)

    def forward(self, feat_convmixer: torch.Tensor, feat_efficientnet: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(feat_convmixer, feat_efficientnet)
        reduced = self.reduction(fused)
        return reduced


if __name__ == "__main__":
    block = FusionReductionBlock()
    dummy_c = torch.randn(8, 64)
    dummy_e = torch.randn(8, 1280)
    out = block(dummy_c, dummy_e)
    print(f"Output shape: {out.shape}")  # expect (8, 128)