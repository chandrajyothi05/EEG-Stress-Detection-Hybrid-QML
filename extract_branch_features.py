"""
extract_branch_features.py

Runs both trained, frozen branches (ConvMixer on FASLT features,
EfficientNet-B0 on azimuthal topomaps) over train/val/test and saves
their raw feature vectors, aligned by the shared global epoch index.

This does NOT apply FeatureFusion/FeatureReduction -- those are learnable
and get trained jointly with the Bi-LSTM stage next, not baked in here as
a static preprocessing step. This script only produces the two aligned
inputs that stage will consume.

Assumes:
  - models/convmixer.py defines ConvMixer (forward -> (logits, features))
  - models/efficientnet_branch.py defines EfficientNetBranch (same convention)
  - preprocessing/faslt_dataset.py defines FASLTDataset(split_name)
  - preprocessing/azimuthal_dataset.py defines AzimuthalDataset(split_name)
  - both datasets filter the same global index space via the same
    eeg_denoised_metadata.npz + subject_split.npz, so their .indices
    arrays should be identical for a given split
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.convmixer import ConvMixer
from models.efficientnet_branch import EfficientNetBranch
from preprocessing.faslt_dataset import FASLTDataset
from preprocessing.azimuthal_dataset import AzimuthalDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("data/processed/branch_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONVMIXER_CKPT = "convmixer_best.pt"
EFFICIENTNET_CKPT = "efficientnet_partial_unfreeze_best.pt"


@torch.no_grad()
def extract_convmixer_features(model, loader):
    model.eval()
    feats, ys = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        _, f = model(x)
        feats.append(f.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


@torch.no_grad()
def extract_efficientnet_features(model, loader):
    model.eval()
    feats, ys = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        _, f = model(x)
        feats.append(f.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def main():
    convmixer = ConvMixer().to(DEVICE)
    convmixer.load_state_dict(torch.load(CONVMIXER_CKPT, map_location=DEVICE))

    efficientnet = EfficientNetBranch().to(DEVICE)
    efficientnet.load_state_dict(torch.load(EFFICIENTNET_CKPT, map_location=DEVICE))

    for split in ("train", "val", "test"):
        faslt_ds = FASLTDataset(split)
        azimuthal_ds = AzimuthalDataset(split)

        # Alignment check: same subjects -> same global indices, in the
        # same order, for both datasets. If this fails, do not proceed --
        # every downstream fusion pairing would be wrong.
        assert np.array_equal(faslt_ds.indices, azimuthal_ds.indices), (
            f"[{split}] index mismatch between FASLTDataset and AzimuthalDataset -- "
            "check that both were built from the same subject_split.npz"
        )

        faslt_loader = DataLoader(faslt_ds, batch_size=32, shuffle=False, num_workers=0)
        azimuthal_loader = DataLoader(azimuthal_ds, batch_size=32, shuffle=False, num_workers=0)

        cm_feats, cm_y = extract_convmixer_features(convmixer, faslt_loader)
        en_feats, en_y = extract_efficientnet_features(efficientnet, azimuthal_loader)

        # Labels should also match exactly, sample-for-sample, since both
        # loaders iterate the same indices in the same (unshuffled) order.
        assert np.array_equal(cm_y, en_y), f"[{split}] label mismatch between branches"
        assert not np.isnan(cm_feats).any() and not np.isinf(cm_feats).any(), f"[{split}] NaN/Inf in ConvMixer features"
        assert not np.isnan(en_feats).any() and not np.isinf(en_feats).any(), f"[{split}] NaN/Inf in EfficientNet features"

        print(f"[{split}] ConvMixer features: {cm_feats.shape} | EfficientNet features: {en_feats.shape} "
              f"| labels: {cm_y.shape}, class balance = {np.bincount(cm_y)}")

        np.save(OUT_DIR / f"convmixer_features_{split}.npy", cm_feats)
        np.save(OUT_DIR / f"efficientnet_features_{split}.npy", en_feats)
        np.save(OUT_DIR / f"labels_{split}.npy", cm_y)
        np.save(OUT_DIR / f"global_indices_{split}.npy", faslt_ds.indices)

    print(f"\nDone. Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()