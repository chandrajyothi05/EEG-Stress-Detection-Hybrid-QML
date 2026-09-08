import torch
from torch.utils.data import DataLoader

from models.quantum_head import QuantumStressHead
from test_quantum_real_data import PooledFeaturesDataset


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits, _, _ = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += features.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cpu")  # quantum layer runs on CPU regardless

    train_ds = PooledFeaturesDataset(
        "data/processed/pooled_features/pooled_train.npy",
        "data/processed/pooled_features/labels_train.npy",
    )
    val_ds = PooledFeaturesDataset(
        "data/processed/pooled_features/pooled_val.npy",
        "data/processed/pooled_features/labels_val.npy",
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = QuantumStressHead(in_dim=128).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 5

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
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{n_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()