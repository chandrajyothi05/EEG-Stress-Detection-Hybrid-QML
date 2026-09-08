"""
train_quantum_v3.py

Changes from train_quantum_v2.py:
  - Loss: class-weighted Focal Loss (gamma=2), weights computed from TRAIN labels only
  - Early stopping: patience=4 on val_loss (stops training if no improvement
    for 4 consecutive epochs), instead of a fixed epoch count
  - Saves best_quantum_head.pt (overwrites the v2 checkpoint -- back it up
    first if you want to keep both, see note at bottom)
  - At the end, loads both the new checkpoint and the old baseline
    (best_quantum_head_baseline.pt, if present) and prints a side-by-side
    comparison: per-class precision/recall/F1, not just accuracy

Architecture is unchanged -- this run is purely about stabilizing training
dynamics on the existing VQC, not about changing the model.
"""

import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader

from models.quantum_head import QuantumStressHead
from test_quantum_real_data import PooledFeaturesDataset


class FocalLoss(torch.nn.Module):
    """
    Multi-class focal loss with per-class weights (alpha) and focusing
    parameter gamma. Reduces to weighted cross-entropy when gamma=0.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Gather the log-prob / prob / weight for the true class of each sample
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = self.class_weights[targets]

        loss = -alpha_t * (1 - p_t).pow(self.gamma) * log_p_t
        return loss.mean()


def compute_class_weights(labels_path: str, n_classes: int = 2) -> torch.Tensor:
    """
    Inverse-frequency class weights computed from TRAIN labels only,
    normalized so weights sum to n_classes (keeps loss scale comparable
    to unweighted cross-entropy).
    """
    labels = np.load(labels_path)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, n_classes: int = 2):
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for p, l in zip(preds, labels):
        cm[l.item(), p.item()] += 1
    return cm


def precision_recall_f1_per_class(cm: torch.Tensor):
    """cm: rows=true, cols=pred. Returns dict per class of (precision, recall, f1)."""
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


def print_report(label: str, val_loss: float, val_acc: float, cm: torch.Tensor):
    print(f"\n{label}")
    print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):\n{cm.numpy()}")
    prf = precision_recall_f1_per_class(cm)
    for c, (p, r, f1) in prf.items():
        name = "non-stress" if c == 0 else "stress"
        print(f"  Class {c} ({name}): precision={p:.3f} recall={r:.3f} f1={f1:.3f}")


def main(seed: int, checkpoint_path: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device("cpu")

    train_features_path = "data/processed/pooled_features/pooled_train.npy"
    train_labels_path = "data/processed/pooled_features/labels_train.npy"
    val_features_path = "data/processed/pooled_features/pooled_val.npy"
    val_labels_path = "data/processed/pooled_features/labels_val.npy"

    class_weights = compute_class_weights(train_labels_path, n_classes=2)
    print(f"Class weights (from train only): {class_weights.tolist()}")

    train_ds = PooledFeaturesDataset(train_features_path, train_labels_path)
    val_ds = PooledFeaturesDataset(val_features_path, val_labels_path)

    train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    generator=generator)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = QuantumStressHead(in_dim=128).to(device)
    criterion = FocalLoss(class_weights=class_weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    max_epochs = 50
    patience = 4
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    

    for epoch in range(1, max_epochs + 1):
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

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        marker = "  <-- best so far" if improved else ""
        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}{marker} "
            f"(no improvement: {epochs_without_improvement}/{patience})"
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(
        f"\nBest checkpoint: epoch {best_epoch} "
        f"(val_loss={best_val_loss:.4f}), saved to {checkpoint_path}"
    )
        # --- Final evaluation of the best checkpoint ---
    ce_criterion = torch.nn.CrossEntropyLoss()

    best_model = QuantumStressHead(in_dim=128).to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    final_val_loss, final_val_acc, final_cm = evaluate(
        best_model, val_loader, ce_criterion, device
    )

    # Stress class = class 1
    tn, fp, fn, tp = final_cm.ravel()

    stress_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    stress_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    stress_f1 = (
        2 * stress_precision * stress_recall /
        (stress_precision + stress_recall)
        if (stress_precision + stress_recall) > 0
        else 0.0
    )

    print(
        f"SUMMARY seed={seed} "
        f"best_epoch={best_epoch} "
        f"val_acc={final_val_acc:.4f} "
        f"stress_precision={stress_precision:.4f} "
        f"stress_recall={stress_recall:.4f} "
        f"stress_f1={stress_f1:.4f}"
    )
    

    # --- Comparison against baseline (v2 checkpoint), if present ---
    import os

    baseline_path = "best_quantum_head_baseline.pt"
    if os.path.exists(baseline_path):
        # Recompute val metrics for the NEW model using plain cross-entropy
        # so both models are scored on the same metric (not focal loss),
        # keeping the comparison apples-to-apples.
        ce_criterion = torch.nn.CrossEntropyLoss()

        new_model = QuantumStressHead(in_dim=128).to(device)
        new_model.load_state_dict(torch.load(checkpoint_path))

        old_model = QuantumStressHead(in_dim=128).to(device)
        old_model.load_state_dict(torch.load(baseline_path))

        new_val_loss, new_val_acc, new_cm = evaluate(
            new_model, val_loader, ce_criterion, device
        )
        old_val_loss, old_val_acc, old_cm = evaluate(
            old_model, val_loader, ce_criterion, device
        )

        print("\n" + "=" * 60)
        print("COMPARISON: baseline (v2, unweighted CE) vs new (v3, focal)")
        print("=" * 60)
        print_report("BASELINE (best_quantum_head_baseline.pt)", old_val_loss, old_val_acc, old_cm)
        print_report("NEW (best_quantum_head.pt)", new_val_loss, new_val_acc, new_cm)
    else:
        print(
            f"\nNo baseline found at '{baseline_path}'. "
            f"To enable comparison next time, copy your v2 checkpoint there:\n"
            f"  Copy-Item best_quantum_head.pt best_quantum_head_baseline.pt\n"
            f"(do this BEFORE running this script again, since it overwrites "
            f"best_quantum_head.pt)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=str, default="best_quantum_head.pt")
    args = parser.parse_args()

    main(
        seed=args.seed,
        checkpoint_path=args.checkpoint_out
    )