"""
train_bilstm.py

Trains FusionBiLSTMAttention on the sequences built by build_sequences.py.
The FeatureFusion/FeatureReduction projections inside the model are
trained here, jointly with the LSTM and classifier -- not precomputed.

Mirrors conventions from the earlier branches: FocalLoss with class
weights, AdamW, ReduceLROnPlateau, early stopping, subject-level
majority-vote evaluation (grouped by (subject, true_label) exactly like
ConvMixer/EfficientNet, but voting over sequences instead of raw epochs).
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.bilstm_attention import FusionBiLSTMAttention

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_DIR = Path("data/processed/sequences")


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


class SequenceDataset(Dataset):
    def __init__(self, split: str):
        self.cm_seq = np.load(SEQ_DIR / f"convmixer_seq_{split}.npy")
        self.en_seq = np.load(SEQ_DIR / f"efficientnet_seq_{split}.npy")
        self.labels = np.load(SEQ_DIR / f"labels_seq_{split}.npy")
        self.subjects = np.load(SEQ_DIR / f"subjects_seq_{split}.npy")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cm = torch.from_numpy(self.cm_seq[idx].astype(np.float32))
        en = torch.from_numpy(self.en_seq[idx].astype(np.float32))
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return cm, en, y


def compute_class_weights(dataset: SequenceDataset) -> torch.Tensor:
    counts = np.bincount(dataset.labels)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for cm, en, y in loader:
            cm, en, y = cm.to(DEVICE), en.to(DEVICE), y.to(DEVICE)
            logits, _, _ = model(cm, en)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * cm.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += cm.size(0)

    return total_loss / total, correct / total


def subject_level_eval(model, dataset: SequenceDataset):
    """Majority-vote accuracy over sequences, grouped by (subject, true_label)
    -- same diagnostic used for every branch so far, now voting over
    sequences (each covering WINDOW_LEN epochs) instead of raw epochs."""
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds = []
    with torch.no_grad():
        for cm, en, y in loader:
            cm, en = cm.to(DEVICE), en.to(DEVICE)
            logits, _, _ = model(cm, en)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds)

    groups = {}
    for subj, true_y, pred in zip(dataset.subjects, dataset.labels, all_preds):
        key = (subj, true_y)
        groups.setdefault(key, []).append(pred)

    correct = 0
    for (subj, true_y), preds in groups.items():
        majority_pred = np.bincount(preds).argmax()
        is_correct = (majority_pred == true_y)
        correct += is_correct
        print(f"  {subj} (true={true_y}): majority_pred={majority_pred} "
              f"{'✓' if is_correct else '✗'} ({len(preds)} sequences)")

    print(f"Subject-level accuracy: {correct}/{len(groups)} = {correct/len(groups):.3f}")
    return correct, len(groups)


if __name__ == "__main__":
    train_ds = SequenceDataset("train")
    val_ds = SequenceDataset("val")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_ds).to(DEVICE)
    print(f"Class weights: {class_weights}")

    model = FusionBiLSTMAttention().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    n_epochs = 30
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
            torch.save(model.state_dict(), "bilstm_attention_best.pt")
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print("\nRunning subject-level evaluation on best checkpoint...")
    model.load_state_dict(torch.load("bilstm_attention_best.pt"))
    subject_level_eval(model, val_ds)