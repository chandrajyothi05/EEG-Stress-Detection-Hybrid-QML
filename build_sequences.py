"""
build_sequences.py

Groups each split's epochs into contiguous (subject, label) blocks and
slides a fixed-length window over each block to build sequences for the
Bi-LSTM stage.

ASSUMPTION (please sanity-check against the printed block lengths): rows
within data/processed/branch_features/*_{split}.npy are still in their
original array order (subject-contiguous, and within each subject,
rest-block-then-task-block, chronological within each block) since
nothing upstream of this script shuffles them -- only DataLoader(shuffle=
True) during training does, and this script doesn't use that.

Output per split, under data/processed/sequences/:
  convmixer_seq_{split}.npy    (num_seqs, WINDOW_LEN, 64)
  efficientnet_seq_{split}.npy (num_seqs, WINDOW_LEN, 1280)
  labels_seq_{split}.npy       (num_seqs,)
  subjects_seq_{split}.npy     (num_seqs,)  -- for subject-level eval later
"""

from pathlib import Path
import numpy as np

IN_DIR = Path("data/processed/branch_features")
OUT_DIR = Path("data/processed/sequences")
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = Path("data/processed/eeg_denoised_metadata.npz")

WINDOW_LEN = 10
STRIDE = 5  # reverted from 10 -- no-overlap experiment gave worse, noisier subject-level results (10/14 vs 12/14)


def contiguous_blocks(subject_ids: np.ndarray, labels: np.ndarray):
    """Yield (start, end) index ranges [start, end) where subject_id and
    label are both constant -- i.e. one rest block or one task block."""
    n = len(subject_ids)
    start = 0
    for i in range(1, n + 1):
        if i == n or subject_ids[i] != subject_ids[start] or labels[i] != labels[start]:
            yield start, i, subject_ids[start], labels[start]
            start = i


def build_split(split: str):
    cm_feats = np.load(IN_DIR / f"convmixer_features_{split}.npy")
    en_feats = np.load(IN_DIR / f"efficientnet_features_{split}.npy")
    labels = np.load(IN_DIR / f"labels_{split}.npy")
    global_idx = np.load(IN_DIR / f"global_indices_{split}.npy")

    metadata = np.load(META_PATH, allow_pickle=True)
    subject_ids = metadata["subject_ids"][global_idx]

    cm_seqs, en_seqs, seq_labels, seq_subjects = [], [], [], []
    n_blocks, n_skipped_short = 0, 0

    for start, end, subj, label in contiguous_blocks(subject_ids, labels):
        n_blocks += 1
        block_len = end - start
        print(f"  [{split}] {subj} label={label} block_len={block_len}")

        if block_len < WINDOW_LEN:
            n_skipped_short += 1
            continue

        for w_start in range(start, end - WINDOW_LEN + 1, STRIDE):
            w_end = w_start + WINDOW_LEN
            cm_seqs.append(cm_feats[w_start:w_end])
            en_seqs.append(en_feats[w_start:w_end])
            seq_labels.append(label)
            seq_subjects.append(subj)

    cm_seqs = np.stack(cm_seqs)          # (num_seqs, WINDOW_LEN, 64)
    en_seqs = np.stack(en_seqs)          # (num_seqs, WINDOW_LEN, 1280)
    seq_labels = np.array(seq_labels)
    seq_subjects = np.array(seq_subjects)

    print(f"[{split}] {n_blocks} blocks total, {n_skipped_short} skipped (shorter than WINDOW_LEN={WINDOW_LEN})")
    print(f"[{split}] built {len(seq_labels)} sequences | class balance = {np.bincount(seq_labels)}")

    np.save(OUT_DIR / f"convmixer_seq_{split}.npy", cm_seqs)
    np.save(OUT_DIR / f"efficientnet_seq_{split}.npy", en_seqs)
    np.save(OUT_DIR / f"labels_seq_{split}.npy", seq_labels)
    np.save(OUT_DIR / f"subjects_seq_{split}.npy", seq_subjects)


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        build_split(split)
        print()

    print(f"Done. Saved to {OUT_DIR}/")