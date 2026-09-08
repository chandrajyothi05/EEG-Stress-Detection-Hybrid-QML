"""
run_multi_seed.py

Runs train_quantum_v3.py across several seeds and aggregates the results,
to check whether the focal-loss improvement over baseline holds up
consistently or was a lucky single run.

Usage:
    python run_multi_seed.py
"""

import re
import subprocess
import sys

import numpy as np

SEEDS = [0, 1, 2, 3, 4]
SUMMARY_PATTERN = re.compile(
    r"SUMMARY seed=(\d+) best_epoch=(\d+) val_acc=([\d.]+) "
    r"stress_precision=([\d.]+) stress_recall=([\d.]+) stress_f1=([\d.]+)"
)


def main():
    results = []

    for seed in SEEDS:
        print(f"\n{'='*60}\nRunning seed={seed}\n{'='*60}")
        checkpoint_out = f"checkpoints_seed{seed}.pt"

        proc = subprocess.run(
            [
                sys.executable,
                "train_quantum_v3.py",
                "--seed",
                str(seed),
                "--checkpoint-out",
                checkpoint_out,
            ],
            capture_output=True,
            text=True,
        )

        print(proc.stdout)
        if proc.returncode != 0:
            print(f"Run for seed={seed} FAILED:\n{proc.stderr}")
            continue

        match = SUMMARY_PATTERN.search(proc.stdout)
        if match is None:
            print(f"Could not parse summary line for seed={seed}; skipping.")
            continue

        _, best_epoch, val_acc, stress_p, stress_r, stress_f1 = match.groups()
        results.append(
            {
                "seed": seed,
                "best_epoch": int(best_epoch),
                "val_acc": float(val_acc),
                "stress_precision": float(stress_p),
                "stress_recall": float(stress_r),
                "stress_f1": float(stress_f1),
            }
        )

    if not results:
        print("No successful runs to summarize.")
        return

    print(f"\n{'='*60}\nMULTI-SEED SUMMARY ({len(results)} runs)\n{'='*60}")
    print(f"{'seed':<6}{'epoch':<8}{'val_acc':<10}{'stress_p':<10}{'stress_r':<10}{'stress_f1':<10}")
    for r in results:
        print(
            f"{r['seed']:<6}{r['best_epoch']:<8}{r['val_acc']:<10.4f}"
            f"{r['stress_precision']:<10.4f}{r['stress_recall']:<10.4f}{r['stress_f1']:<10.4f}"
        )

    for key in ["val_acc", "stress_precision", "stress_recall", "stress_f1"]:
        values = np.array([r[key] for r in results])
        print(f"\n{key}: mean={values.mean():.4f}  std={values.std():.4f}  "
              f"min={values.min():.4f}  max={values.max():.4f}")

    print(
        "\nInterpretation guide:\n"
        "  - Low std (e.g. < 0.05) on stress_f1 => result is stable across "
        "initialization/shuffling, trust the baseline number.\n"
        "  - High std (e.g. > 0.10) => result is sensitive to seed; the "
        "single-run comparison against v2 isn't reliable evidence on its own."
    )


if __name__ == "__main__":
    main()