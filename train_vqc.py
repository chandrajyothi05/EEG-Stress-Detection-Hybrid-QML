"""
train_vqc.py

Trains QuantumStressHead on the pooled (128-dim) features extracted by
extract_pooled_features.py. The classical fusion/BiLSTM/attention stack
stays frozen (already extracted); only quantum_projection, the VQC
ansatz weights, and the post-quantum classifier train here.

Same conventions as every prior stage: FocalLoss + class weights, AdamW,
ReduceLROnPlateau, early stopping, subject-level majority-vote eval.

PERFORMANCE NOTE: quantum circuit simulation (even for 2 qubits) is much
slower per-sample than any classical layer so far. If an epoch is taking
far longer than the classical stages did, drop batch_size or n_epochs
before assuming something is broken -- this is expected to be the
slowest stage in the whole pipeline despite having the fewest params.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.quantum_head import QuantumStressHead

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POOLED_DIR = Path("data/processed/pooled_features")


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


class PooledDataset(Dataset):
    def __init__(self, split: str):
        self.pooled = np.load(POOLED_DIR / f"pooled_{split}.npy")
        self.labels = np.load(POOLED_DIR / f"labels_{split}.npy")
        self.subjects = np.load(POOLED_DIR / f"subjects_{split}.npy")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.pooled[idx].astype(np.float32))
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def compute_class_weights(dataset: PooledDataset) -> torch.Tensor:
    counts = np.bincount(dataset.labels)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _, _ = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def subject_level_eval(model, dataset: PooledDataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits, _, _ = model(x)
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
    train_ds = PooledDataset("train")
    val_ds = PooledDataset("val")

    # Small batch size: quantum simulation cost scales with samples-per-step,
    # and there's no benefit to large batches here since the ansatz has only
    # 4 trainable parameters total.
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_ds).to(DEVICE)
    print(f"Class weights: {class_weights}")

    model = QuantumStressHead().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,} (of which 4 are the VQC ansatz's own weights)")

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
            torch.save(model.state_dict(), "quantum_head_best.pt")
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print("\nRunning subject-level evaluation on best checkpoint...")
    model.load_state_dict(torch.load("quantum_head_best.pt"))
    subject_level_eval(model, val_ds)