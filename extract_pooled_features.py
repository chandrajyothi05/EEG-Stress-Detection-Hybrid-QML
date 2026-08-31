"""
extract_pooled_features.py

Runs the trained, now-frozen FusionBiLSTMAttention (up through attention
pooling) over train/val/test sequences and saves the 128-dim pooled
vectors. The VQC stage trains on top of these -- the classical
fusion/reduction/LSTM/attention stack is treated as a fixed feature
extractor at this point, the same staged-training pattern used for
ConvMixer and EfficientNet earlier. This also matters for speed: quantum
circuit simulation is the bottleneck in the next stage, so we don't want
to re-run the LSTM every epoch on top of it.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.bilstm_attention import FusionBiLSTMAttention
from train_bilstm import SequenceDataset  # reuse the same Dataset class

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("data/processed/pooled_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = "bilstm_attention_best.pt"


@torch.no_grad()
def extract(model, loader):
    model.eval()
    pooled_all, y_all = [], []
    for cm, en, y in loader:
        cm, en = cm.to(DEVICE), en.to(DEVICE)
        _, pooled, _ = model(cm, en)
        pooled_all.append(pooled.cpu().numpy())
        y_all.append(y.numpy())
    return np.concatenate(pooled_all), np.concatenate(y_all)


def main():
    model = FusionBiLSTMAttention().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    for split in ("train", "val", "test"):
        ds = SequenceDataset(split)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        pooled, y = extract(model, loader)

        assert not np.isnan(pooled).any() and not np.isinf(pooled).any(), f"[{split}] NaN/Inf in pooled features"
        assert np.array_equal(y, ds.labels), f"[{split}] label order mismatch"

        print(f"[{split}] pooled features: {pooled.shape} | class balance = {np.bincount(y)}")

        np.save(OUT_DIR / f"pooled_{split}.npy", pooled)
        np.save(OUT_DIR / f"labels_{split}.npy", y)
        np.save(OUT_DIR / f"subjects_{split}.npy", ds.subjects)

    print(f"\nDone. Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()