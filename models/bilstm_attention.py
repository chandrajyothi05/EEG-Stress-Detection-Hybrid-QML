"""
models/bilstm_attention.py

Consumes per-timestep ConvMixer (64-dim) and EfficientNet (1280-dim)
features for a WINDOW_LEN-length sequence, fuses+reduces each timestep
via FusionReductionBlock (trained jointly, not precomputed), runs a
bidirectional LSTM over the sequence, then an additive-attention pooling
layer collapses the sequence into a single context vector for
classification.
"""

import torch
import torch.nn as nn

from models.feature_fusion import FusionReductionBlock


class AttentionPool(nn.Module):
    """Additive attention pooling over a sequence of hidden states.
    Learns a context vector; attention weights = softmax(tanh(W*h) . context)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h_seq: torch.Tensor):
        # h_seq: (B, T, hidden_dim)
        scores = self.context(torch.tanh(self.proj(h_seq)))   # (B, T, 1)
        weights = torch.softmax(scores, dim=1)                 # (B, T, 1)
        pooled = (h_seq * weights).sum(dim=1)                  # (B, hidden_dim)
        return pooled, weights.squeeze(-1)                     # (B, hidden_dim), (B, T)


class FusionBiLSTMAttention(nn.Module):
    def __init__(
        self,
        convmixer_dim: int = 64,
        efficientnet_dim: int = 1280,
        fusion_shared_dim: int = 128,
        fusion_reduced_dim: int = 128,
        lstm_hidden: int = 64,  # reverted from 32 -- combined with dropout tweak, gave worse results
        n_classes: int = 2,
        dropout: float = 0.3,   # reverted from 0.5
    ):
        super().__init__()
        self.fusion_reduction = FusionReductionBlock(
            convmixer_dim=convmixer_dim,
            efficientnet_dim=efficientnet_dim,
            shared_dim=fusion_shared_dim,
            reduced_dim=fusion_reduced_dim,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(
            input_size=fusion_reduced_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        self.attention = AttentionPool(lstm_out_dim)
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, n_classes)

    def forward(self, cm_seq: torch.Tensor, en_seq: torch.Tensor):
        # cm_seq: (B, T, 64), en_seq: (B, T, 1280)
        B, T, _ = cm_seq.shape

        # Apply fusion+reduction per timestep: flatten (B,T,*) -> (B*T,*)
        cm_flat = cm_seq.reshape(B * T, -1)
        en_flat = en_seq.reshape(B * T, -1)
        fused_flat = self.fusion_reduction(cm_flat, en_flat)      # (B*T, reduced_dim)
        fused_seq = fused_flat.reshape(B, T, -1)                  # (B, T, reduced_dim)

        lstm_out, _ = self.lstm(fused_seq)                        # (B, T, lstm_out_dim)
        pooled, attn_weights = self.attention(lstm_out)           # (B, lstm_out_dim), (B, T)

        logits = self.classifier(self.classifier_dropout(pooled))
        return logits, pooled, attn_weights


if __name__ == "__main__":
    model = FusionBiLSTMAttention()
    dummy_cm = torch.randn(4, 10, 64)
    dummy_en = torch.randn(4, 10, 1280)
    logits, pooled, attn_weights = model(dummy_cm, dummy_en)
    print(f"Logits shape: {logits.shape}")           # expect (4, 2)
    print(f"Pooled shape: {pooled.shape}")            # expect (4, 128)
    print(f"Attention weights shape: {attn_weights.shape}")  # expect (4, 10)
    print(f"Attention weights sum to 1 per sample: {attn_weights.sum(dim=1)}")