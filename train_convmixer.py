import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np

from preprocessing.faslt_dataset import FASLTDataset
from models.convmixer import ConvMixer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al. 2017): down-weights well-classified ("easy")
    examples via (1-p_t)^gamma, so gradient focuses on hard/misclassified
    examples -- directly targets the "model defaults to easy majority class"
    failure mode seen with plain weighted cross-entropy here.
    alpha (per-class weight) is reused from the existing inverse-frequency
    class_weights, so class imbalance is still accounted for on top of the
    hard-example focusing.
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = at * loss

        return loss.mean()


def compute_class_weights(dataset: FASLTDataset) -> torch.Tensor:
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


def evaluate_subject_level(model, dataset: FASLTDataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    all_preds, all_true, all_subjects = [], [], []
    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            X = X.to(DEVICE)
            logits, _ = model(X)
            preds = logits.argmax(1).cpu().numpy()
            start = i * loader.batch_size
            end = start + len(preds)
            batch_real_indices = dataset.indices[start:end]
            batch_subjects = dataset.subject_ids_all[batch_real_indices]

            all_preds.extend(preds)
            all_true.extend(y.numpy())
            all_subjects.extend(batch_subjects)

    groups = defaultdict(list)
    for pred, true, subj in zip(all_preds, all_true, all_subjects):
        groups[(subj, true)].append(pred)

    correct, total = 0, 0
    print("\n  Subject-level (majority-vote) breakdown:")
    for (subj, true_label), preds in sorted(groups.items()):
        majority_pred = int(round(np.mean(preds)))
        is_correct = majority_pred == true_label
        correct += is_correct
        total += 1
        print(f"    subject={subj} true_label={true_label} "
              f"n_epochs={len(preds)} epoch_acc={np.mean(np.array(preds) == true_label):.2f} "
              f"majority_pred={majority_pred} {'OK' if is_correct else 'WRONG'}")

    subject_acc = correct / total
    print(f"  Subject-level accuracy: {correct}/{total} = {subject_acc:.4f}\n")
    return subject_acc


if __name__ == "__main__":
    train_ds = FASLTDataset("train")
    val_ds = FASLTDataset("val")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_ds).to(DEVICE)
    print(f"Class weights (used as focal loss alpha): {class_weights}")

    model = ConvMixer().to(DEVICE)  # embed_dim=64, depth=3 (unchanged from previous run)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    n_epochs = 30
    patience = 5
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{n_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.2e}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "convmixer_best.pt")
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  -> no improvement for {patience} epochs, stopping early")
                break

    print(f"Training complete. Best epoch-level val_acc: {best_val_acc:.4f}")

    model.load_state_dict(torch.load("convmixer_best.pt"))
    evaluate_subject_level(model, val_ds)