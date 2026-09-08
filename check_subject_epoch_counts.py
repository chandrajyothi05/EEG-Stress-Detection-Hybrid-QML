"""
check_subject_epoch_counts.py

Quick sanity check: are the 6 test subjects that every branch
consistently misclassifies (Subject12, 13, 21, 26, 31, 35) unusual in
some simple way -- e.g. fewer epochs, more imbalanced rest/task split --
compared to the rest of the dataset?

Run this from the project root (where data/processed/ lives):
    python check_subject_epoch_counts.py
"""

import numpy as np
from collections import defaultdict

FLAGGED_SUBJECTS = {"Subject12", "Subject13", "Subject21", "Subject26", "Subject31", "Subject35"}

metadata = np.load("data/processed/eeg_denoised_metadata.npz", allow_pickle=True)
subject_ids = metadata["subject_ids"]
labels = metadata["y"]

# Count rest (0) and task/stress (1) epochs per subject
counts = defaultdict(lambda: [0, 0])
for subj, y in zip(subject_ids, labels):
    counts[subj][y] += 1

print(f"{'Subject':<12} {'Rest':>6} {'Task':>6} {'Total':>6}  Flagged?")
print("-" * 50)

all_totals, flagged_totals, other_totals = [], [], []
all_task_counts, flagged_task_counts, other_task_counts = [], [], []

for subj in sorted(counts.keys()):
    rest, task = counts[subj]
    total = rest + task
    is_flagged = subj in FLAGGED_SUBJECTS
    marker = "  <-- FLAGGED" if is_flagged else ""
    print(f"{subj:<12} {rest:>6} {task:>6} {total:>6}{marker}")

    all_totals.append(total)
    all_task_counts.append(task)
    if is_flagged:
        flagged_totals.append(total)
        flagged_task_counts.append(task)
    else:
        other_totals.append(total)
        other_task_counts.append(task)

print("\n--- Summary ---")
print(f"All subjects        : n={len(all_totals):2d}  mean_total={np.mean(all_totals):.1f}  mean_task_epochs={np.mean(all_task_counts):.1f}")
print(f"Flagged subjects     : n={len(flagged_totals):2d}  mean_total={np.mean(flagged_totals):.1f}  mean_task_epochs={np.mean(flagged_task_counts):.1f}")
print(f"Other subjects       : n={len(other_totals):2d}  mean_total={np.mean(other_totals):.1f}  mean_task_epochs={np.mean(other_task_counts):.1f}")

print(
    "\nInterpretation: if the flagged group's mean_total or mean_task_epochs "
    "is notably lower than 'Other subjects', these subjects may simply have "
    "less task data to learn/predict from. If the numbers look similar, "
    "epoch count is not the explanation and the difficulty is likely in the "
    "signal itself (harder to model), not in how much data each has."
)