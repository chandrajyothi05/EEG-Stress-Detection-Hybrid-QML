import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from preprocessing.faslt_dataset import FASLTDataset
from models.convmixer import ConvMixer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_class_weights(dataset: FASLTDataset) -> torch.Tensor:
    labels = dataset.y_all[dataset.indices]
    counts = np.bincount(labels)  # [n_class0, n_class1]
    weights = counts.sum() / (len(counts) * counts)  # inverse-frequency weighting
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


if __name__ == "__main__":
    train_ds = FASLTDataset("train")
    val_ds = FASLTDataset("val")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    class_weights = compute_class_weights(train_ds).to(DEVICE)
    print(f"Class weights: {class_weights}")

    model = ConvMixer().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 20
    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch}/{n_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "convmixer_best.pt")
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")