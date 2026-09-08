"""
eval_checkpoint.py

Loads a saved checkpoint and reports val-set metrics using plain
CrossEntropyLoss (not focal loss), so results are comparable across
any run regardless of what loss function was used to train it.

Usage:
    python eval_checkpoint.py best_quantum_head.pt
"""

import sys
import torch
from torch.utils.data import DataLoader

from models.quantum_head import QuantumStressHead
from test_quantum_real_data import PooledFeaturesDataset


def confusion_matrix(preds, labels, n_classes=2):
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for p, l in zip(preds, labels):
        cm[l.item(), p.item()] += 1
    return cm


def precision_recall_f1_per_class(cm):
    n_classes = cm.shape[0]
    results = {}
    for c in range(n_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results[c] = (precision, recall, f1)
    return results


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "best_quantum_head.pt"

    device = torch.device("cpu")
    val_ds = PooledFeaturesDataset(
        "data/processed/pooled_features/pooled_val.npy",
        "data/processed/pooled_features/labels_val.npy",
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = QuantumStressHead(in_dim=128).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
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
    val_loss = total_loss / total
    val_acc = (all_preds == all_labels).float().mean().item()
    cm = confusion_matrix(all_preds, all_labels)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"val_loss (plain CE)={val_loss:.4f}  val_acc={val_acc:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred):\n{cm.numpy()}")

    prf = precision_recall_f1_per_class(cm)
    for c, (p, r, f1) in prf.items():
        name = "non-stress" if c == 0 else "stress"
        print(f"Class {c} ({name}): precision={p:.3f} recall={r:.3f} f1={f1:.3f}")


if __name__ == "__main__":
    main()