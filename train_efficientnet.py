"""
train_efficientnet.py

Training loop for the Azimuthal Projection -> EfficientNet-B0 branch.

Run history so far:
  - freeze_backbone=True, uniform lr=1e-4 on model.parameters() -> full
    fine-tune, overfit (train 66%->91%, val degraded, best at epoch 1)
  - freeze_backbone=True (whole backbone frozen), lr=1e-3 -> underfit
    (subject-level 10/14 = 71.4%, errors skewed to missing stress)

THIS VERSION: partial unfreeze. Freeze the whole backbone, then re-enable
gradients on the last MBConv stage (features[-2]) and the final 1x1 conv
(features[-1]) -- features[-1] alone is just a pointwise conv, so it's
unfrozen together with the last real MBConv stage for it to mean anything.
Two LR groups: the unfrozen backbone tail trains slower than the
classifier head, since it still carries useful pretrained weights.
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


def unfreeze_last_stage(model: EfficientNetBranch):
    """model was constructed with freeze_backbone=True, so the whole
    backbone starts frozen. Re-enable gradients on the last MBConv stage
    (features[-2]) + final 1x1 conv (features[-1]); classifier head is
    always trainable regardless."""
    for param in model.backbone.features[-2].parameters():
        param.requires_grad = True
    for param in model.backbone.features[-1].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def get_param_groups(model: EfficientNetBranch):
    backbone_tail_params = (
        list(model.backbone.features[-2].parameters())
        + list(model.backbone.features[-1].parameters())
    )
    head_params = list(model.classifier.parameters())
    return [
        {"params": backbone_tail_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ]


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
    diagnostic used for ConvMixer and the previous two EfficientNet runs."""
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

    model = EfficientNetBranch(freeze_backbone=True).to(DEVICE)
    model = unfreeze_last_stage(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(get_param_groups(model), weight_decay=1e-2)
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
            torch.save(model.state_dict(), "efficientnet_partial_unfreeze_best.pt")
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print("\nRunning subject-level evaluation on best checkpoint...")
    model.load_state_dict(torch.load("efficientnet_partial_unfreeze_best.pt"))
    subject_level_eval(model, val_ds)