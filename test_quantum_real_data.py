"""
test_quantum_real_data.py

Loads the real pooled EEG features (train split) and runs one batch
through the full QuantumStressHead to verify:
  - shapes are correct end-to-end
  - latent values (post-sigmoid, pre-quantum-circuit) aren't saturating
    near 0 or pi, which would flatten gradients through the quantum layer
  - gradients actually reach the ansatz parameters (theta) on real data,
    not just dummy random data
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.quantum_head import QuantumStressHead


class PooledFeaturesDataset(Dataset):
    def __init__(self, features_path: str, labels_path: str):
        # mmap_mode keeps memory low; fine for float32 (N, 128) arrays this size
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)

        assert self.features.shape[0] == self.labels.shape[0], (
            f"Mismatched sample counts: "
            f"{self.features.shape[0]} features vs {self.labels.shape[0]} labels"
        )

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()
        y = torch.tensor(self.labels[idx]).long()
        return x, y


def main():
    train_ds = PooledFeaturesDataset(
        "data/processed/pooled_features/pooled_train.npy",
        "data/processed/pooled_features/labels_train.npy",
    )

    print(f"Train dataset size: {len(train_ds)}")

    # Small batch first, matching the dummy-data smoke test batch size
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    model = QuantumStressHead(in_dim=128)
    criterion = torch.nn.CrossEntropyLoss()

    features, labels = next(iter(loader))
    print(f"Batch features shape: {features.shape}")
    print(f"Batch labels: {labels.tolist()}")

    logits, quantum_out, latent = model(features)

    print(f"Logits shape: {logits.shape}")
    print(f"Quantum output shape: {quantum_out.shape}")
    print(f"Latent shape: {latent.shape}")
    print(
        f"Latent range: [{latent.min().item():.3f}, {latent.max().item():.3f}] "
        f"(expected within [0, pi] ~= [0, 3.1416])"
    )

    # Check how close latent values get to the saturation boundaries (0 and pi)
    # Values very close to either edge => sigmoid gradient near zero => vanishing
    # gradient signal into the quantum circuit.
    near_zero = (latent < 0.05).float().mean().item()
    near_pi = (latent > (torch.pi - 0.05)).float().mean().item()
    print(
        f"Fraction of latent values near 0: {near_zero:.3f}, "
        f"near pi: {near_pi:.3f} (want these low, ideally < 0.05)"
    )

    loss = criterion(logits, labels)
    print(f"Loss: {loss.item()}")

    loss.backward()

    print("Gradient check (real data):")
    for name, p in model.quantum_layer.named_parameters():
        grad_exists = p.grad is not None
        grad_mean = p.grad.abs().mean().item() if grad_exists else None
        print(f"  {name} grad_exists={grad_exists} grad_mean={grad_mean}")

    # Also check the classical projection layer got gradients
    proj_grad = model.quantum_projection.weight.grad
    print(
        f"quantum_projection.weight grad_exists={proj_grad is not None} "
        f"grad_mean={proj_grad.abs().mean().item() if proj_grad is not None else None}"
    )


if __name__ == "__main__":
    main()