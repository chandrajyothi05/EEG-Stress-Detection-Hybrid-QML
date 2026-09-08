"""
explainability_analysis.py

Pulls two explainability signals for chosen case-study subjects, aligned
per-sequence:
  1. Bi-LSTM+Attention's attention weights over the 10-step sequence
     (where in time the model focused).
  2. The VQC's 2 Pauli-Z expectation values (what the quantum layer
     actually measured) for the same sequences.

Relies on extract_pooled_features.py having built pooled_{split}.npy /
labels_{split}.npy / subjects_{split}.npy with shuffle=False over
SequenceDataset(split) -- so row i in those files corresponds exactly to
row i of SequenceDataset(split). We assert this alignment before using it.

Case studies (edit CASE_STUDY_SUBJECTS to adjust):
  - Subject14 (val): misclassified by every architecture tried so far.
  - Subject12, 13, 21, 26, 31, 35 (test): shared failure group where
    every branch misses the stress block, ruled out as an epoch-count
    effect.
  - Subject08, Subject20 (val): consistently correct, used as contrast.
"""

from collections import defaultdict
import numpy as np
import torch

from models.bilstm_attention import FusionBiLSTMAttention
from models.quantum_head import QuantumStressHead
from train_bilstm import SequenceDataset

DEVICE = "cpu"

CASE_STUDY_SUBJECTS = {
    "val": {
        "hard": ["Subject14"],
        "easy": ["Subject08", "Subject20"],
    },
    "test": {
        "hard": ["Subject12", "Subject13", "Subject21", "Subject26", "Subject31", "Subject35"],
        "easy": [],  # fill in with any consistently-correct test subjects if you want a test-side contrast
    },
}


def load_split(split: str):
    seq_ds = SequenceDataset(split)

    pooled = np.load(f"data/processed/pooled_features/pooled_{split}.npy")
    labels_pooled = np.load(f"data/processed/pooled_features/labels_{split}.npy")
    subjects_pooled = np.load(f"data/processed/pooled_features/subjects_{split}.npy")

    assert len(pooled) == len(seq_ds), (
        f"[{split}] pooled length {len(pooled)} != SequenceDataset length {len(seq_ds)}"
    )
    assert np.array_equal(labels_pooled, seq_ds.labels), (
        f"[{split}] label order mismatch between pooled_{split}.npy and SequenceDataset -- "
        f"alignment assumption broken, do not trust per-sequence pairing below"
    )
    assert np.array_equal(subjects_pooled, seq_ds.subjects), (
        f"[{split}] subject order mismatch between subjects_{split}.npy and SequenceDataset"
    )

    return seq_ds, pooled


def run_models(seq_ds, pooled, bilstm_model, quantum_model):
    """Forward every sequence through both models, return per-sequence
    attention weights, quantum outputs, and predictions."""
    n = len(seq_ds)
    all_attn = np.zeros((n, 10), dtype=np.float32)
    all_quantum_out = np.zeros((n, 2), dtype=np.float32)
    all_bilstm_pred = np.zeros(n, dtype=np.int64)
    all_quantum_pred = np.zeros(n, dtype=np.int64)

    bilstm_model.eval()
    quantum_model.eval()

    batch_size = 32
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            cm = torch.stack([seq_ds[i][0] for i in range(start, end)]).to(DEVICE)
            en = torch.stack([seq_ds[i][1] for i in range(start, end)]).to(DEVICE)

            logits_b, _, attn_weights = bilstm_model(cm, en)
            all_attn[start:end] = attn_weights.cpu().numpy()
            all_bilstm_pred[start:end] = logits_b.argmax(1).cpu().numpy()

            pooled_batch = torch.from_numpy(pooled[start:end]).float().to(DEVICE)
            logits_q, quantum_out, _ = quantum_model(pooled_batch)
            all_quantum_out[start:end] = quantum_out.cpu().numpy()
            all_quantum_pred[start:end] = logits_q.argmax(1).cpu().numpy()

    return all_attn, all_quantum_out, all_bilstm_pred, all_quantum_pred


def summarize_subject(subj, seq_ds, attn, quantum_out, bilstm_pred, quantum_pred):
    idx_by_label = defaultdict(list)
    for i, (s, y) in enumerate(zip(seq_ds.subjects, seq_ds.labels)):
        if s == subj:
            idx_by_label[y].append(i)

    for true_label, idxs in sorted(idx_by_label.items()):
        idxs = np.array(idxs)
        mean_attn = attn[idxs].mean(axis=0)          # (10,) avg attention profile
        mean_quantum = quantum_out[idxs].mean(axis=0)  # (2,) avg expectation values
        bilstm_majority = int(round(bilstm_pred[idxs].mean()))
        quantum_majority = int(round(quantum_pred[idxs].mean()))

        label_name = "rest" if true_label == 0 else "stress"
        print(f"\n  {subj} -- true={label_name} ({len(idxs)} sequences)")
        print(f"    Bi-LSTM majority pred:  {'rest' if bilstm_majority == 0 else 'stress'}"
              f" {'OK' if bilstm_majority == true_label else 'WRONG'}")
        print(f"    VQC majority pred:      {'rest' if quantum_majority == 0 else 'stress'}"
              f" {'OK' if quantum_majority == true_label else 'WRONG'}")
        print(f"    Mean attention profile (10 timesteps): {np.round(mean_attn, 3).tolist()}")
        print(f"    Mean quantum expectation <Z0>,<Z1>:     {np.round(mean_quantum, 3).tolist()}")


def main():
    bilstm_model = FusionBiLSTMAttention().to(DEVICE)
    bilstm_model.load_state_dict(torch.load("bilstm_attention_best.pt", map_location=DEVICE))

    quantum_model = QuantumStressHead(in_dim=128).to(DEVICE)
    quantum_model.load_state_dict(torch.load("checkpoints_seed4.pt", map_location=DEVICE))

    for split, groups in CASE_STUDY_SUBJECTS.items():
        seq_ds, pooled = load_split(split)
        attn, quantum_out, bilstm_pred, quantum_pred = run_models(
            seq_ds, pooled, bilstm_model, quantum_model
        )

        for group_name, subjects in groups.items():
            if not subjects:
                continue
            print("\n" + "=" * 60)
            print(f"{split.upper()} split -- {group_name.upper()} subjects")
            print("=" * 60)
            for subj in subjects:
                summarize_subject(subj, seq_ds, attn, quantum_out, bilstm_pred, quantum_pred)


if __name__ == "__main__":
    main()