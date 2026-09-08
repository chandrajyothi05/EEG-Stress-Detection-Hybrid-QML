"""
evaluate_vqc_subject_level.py

Subject-level (majority-vote) evaluation for the trained QuantumStressHead,
mirroring evaluate_subject_level() in train_convmixer.py so all four
branches (ConvMixer, EfficientNet, Bi-LSTM+Attention, VQC) are comparable
on the same metric.
"""

from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.quantum_head import QuantumStressHead
from test_quantum_real_data import PooledFeaturesDataset

DEVICE = "cpu"  # VQC stage has been CPU-only throughout


def evaluate_vqc_subject_level(model, features_path, labels_path, subjects_path, batch_size=16):
    model.eval()
    ds = PooledFeaturesDataset(features_path, labels_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    subjects = np.load(subjects_path)
    assert len(subjects) == len(ds), (
        f"subjects array length {len(subjects)} != dataset length {len(ds)} "
        f"-- check {subjects_path} matches {features_path} row-for-row"
    )

    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            logits, _, _ = model(X)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    groups = defaultdict(list)
    for pred, true, subj in zip(all_preds, all_true, subjects):
        groups[(subj, true)].append(pred)

    correct, total = 0, 0
    print("\n  Subject-level (majority-vote) breakdown -- VQC:")
    for (subj, true_label), preds in sorted(groups.items()):
        majority_pred = int(round(np.mean(preds)))
        is_correct = majority_pred == true_label
        correct += is_correct
        total += 1
        print(f"    subject={subj} true_label={true_label} "
              f"n_seqs={len(preds)} epoch_acc={np.mean(np.array(preds) == true_label):.2f} "
              f"majority_pred={majority_pred} {'OK' if is_correct else 'WRONG'}")

    subject_acc = correct / total
    print(f"  VQC subject-level accuracy: {correct}/{total} = {subject_acc:.4f}\n")
    return subject_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_quantum_head.pt")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    model = QuantumStressHead(in_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))

    evaluate_vqc_subject_level(
        model,
        features_path=f"data/processed/pooled_features/pooled_{args.split}.npy",
        labels_path=f"data/processed/pooled_features/labels_{args.split}.npy",
        subjects_path=f"data/processed/pooled_features/subjects_{args.split}.npy",
    )