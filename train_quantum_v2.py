"""
train_quantum_v2.py

Improvements over train_quantum_smoke.py:
  1. Prints class balance for train/val before training starts
  2. Adds weight_decay to the optimizer (L2 regularization)
  3. Tracks and saves the best checkpoint by val_loss (early-stopping style)
  4. Prints a confusion matrix each epoch, not just accuracy

This is still a smoke-test-scale script (few epochs, small model) --
the point is to see the *trend* clearly before committing to a longer run.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.quantum_head import QuantumStressHead
from test_quantum_real_data import PooledFeaturesDataset


def print_class_balance(name: str, labels_path: str):
    labels = np.load(labels_path)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    dist = ", ".join(
        f"class {u}: {c} ({c/total:.1%})" for u, c in zip(unique, counts)
    )
    print(f"{name} class balance ({total} samples): {dist}")


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, n_classes: int = 2):
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for p, l in zip(preds, labels):
        cm[l.item(), p.item()] += 1
    return cm


def evaluate(model, loader, criterion, device, n_classes=2):
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits, _, _ = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * features.size(0)
            total += features.size(0)

            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = (all_preds == all_labels).float().mean().item()
    cm = confusion_matrix(all_preds, all_labels, n_classes)

    return total_loss / total, acc, cm


def main():
    device = torch.device("cpu")

    train_features_path = "data/processed/pooled_features/pooled_train.npy"
    train_labels_path = "data/processed/pooled_features/labels_train.npy"
    val_features_path = "data/processed/pooled_features/pooled_val.npy"
    val_labels_path = "data/processed/pooled_features/labels_val.npy"

    # 1. Class balance check -- run this before anything else
    print_class_balance("Train", train_labels_path)
    print_class_balance("Val", val_labels_path)
    print()

    train_ds = PooledFeaturesDataset(train_features_path, train_labels_path)
    val_ds = PooledFeaturesDataset(val_features_path, val_labels_path)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = QuantumStressHead(in_dim=128).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # 2. Weight decay added for L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_epochs = 15
    best_val_loss = float("inf")
    best_epoch = -1
    checkpoint_path = "best_quantum_head.pt"

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _, _ = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            n_seen += features.size(0)

        train_loss = running_loss / n_seen
        val_loss, val_acc, cm = evaluate(model, val_loader, criterion, device)

        # 3. Save checkpoint whenever val_loss improves
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        marker = "  <-- best so far" if improved else ""
        print(
            f"Epoch {epoch}/{n_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}{marker}"
        )
        # 4. Confusion matrix each epoch
        print(f"  Confusion matrix (rows=true, cols=pred):\n{cm.numpy()}")

    print(
        f"\nBest checkpoint: epoch {best_epoch} "
        f"(val_loss={best_val_loss:.4f}), saved to {checkpoint_path}"
    )
    print(
        "To use the best model later:\n"
        "  model = QuantumStressHead(in_dim=128)\n"
        f"  model.load_state_dict(torch.load('{checkpoint_path}'))\n"
        "  model.eval()"
    )


if __name__ == "__main__":
    main()