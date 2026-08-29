"""
train_efficientnet.py

Training loop for the Azimuthal Projection -> EfficientNet-B0 branch.
Mirrors train_convmixer.py's structure and lessons learned:
  - class weights / focal loss for the ~2.96:1 imbalance
  - subject-level majority-vote evaluation (not raw epoch accuracy)
  - num_workers=0 pattern isn't needed here since AzimuthalDataset loads
    the full array into RAM upfront (no memmap), but keeping it low is
    still safer on Windows.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing.azimuthal_dataset import AzimuthalDataset
from models.efficientnet_branch import EfficientNetBranch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def compute_class_weights(dataset: AzimuthalDataset) -> torch.Tensor:
    labels = dataset.y_all[dataset.indices]
    counts = np.bincount(labels)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits, _ = model(X)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

    return total_loss / total, correct / total


def subject_level_eval(model, dataset: AzimuthalDataset):
    """Majority-vote accuracy grouped by (subject_id, true_label) -- same
    diagnostic that revealed ConvMixer's rest-bias earlier."""
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    metadata = np.load("data/processed/eeg_denoised_metadata.npz", allow_pickle=True)
    subject_ids = metadata["subject_ids"][dataset.indices]
    true_labels = metadata["y"][dataset.indices]

    all_preds = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            logits, _ = model(X)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds)

    groups = {}
    for subj, true_y, pred in zip(subject_ids, true_labels, all_preds):
        key = (subj, true_y)
        groups.setdefault(key, []).append(pred)

    correct = 0
    for (subj, true_y), preds in groups.items():
        majority_pred = np.bincount(preds).argmax()
        is_correct = (majority_pred == true_y)
        correct += is_correct
        print(f"  {subj} (true={true_y}): majority_pred={majority_pred} "
              f"{'✓' if is_correct else '✗'} ({len(preds)} epochs)")

    print(f"Subject-level accuracy: {correct}/{len(groups)} = {correct/len(groups):.3f}")
    return correct, len(groups)


if __name__ == "__main__":
    train_ds = AzimuthalDataset("train")
    val_ds = AzimuthalDataset("val")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_ds).to(DEVICE)
    print(f"Class weights: {class_weights}")

    model = EfficientNetBranch(freeze_backbone=False).to(DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)  # starting directly with what worked for ConvMixer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)  # lower lr: pretrained backbone
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    n_epochs = 20
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 5

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{n_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "efficientnet_best.pt")
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print("\nRunning subject-level evaluation on best checkpoint...")
    model.load_state_dict(torch.load("efficientnet_best.pt"))
    subject_level_eval(model, val_ds)