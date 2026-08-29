import numpy as np

FASLT_PATH = "data/processed/faslt_features.npy"
METADATA_PATH = "data/processed/eeg_denoised_metadata.npz"

# --------------------------------------------------
# Load
# --------------------------------------------------

faslt = np.load(FASLT_PATH, mmap_mode="r")
metadata = np.load(METADATA_PATH, allow_pickle=True)

y = metadata["y"]
subject_ids = metadata["subject_ids"]

# --------------------------------------------------
# Step 1: Shape, dtype and alignment
# --------------------------------------------------

print("========== SHAPE / ALIGNMENT ==========")

print(f"FASLT shape: {faslt.shape}")
print(f"FASLT dtype: {faslt.dtype}")
print(f"y shape: {y.shape}")
print(f"subject_ids shape: {subject_ids.shape}")

assert faslt.shape == (8604, 19, 40, 1000)
assert faslt.dtype == np.float32
assert faslt.shape[0] == y.shape[0] == subject_ids.shape[0]

print("Shape check: PASSED")
print("Dtype check: PASSED")
print("Alignment check: PASSED")

# --------------------------------------------------
# Step 2: Full NaN / Inf check
# --------------------------------------------------

print("\n========== NAN / INF CHECK ==========")

n_epochs = faslt.shape[0]
batch_size = 500

has_nan = False
has_inf = False

for start in range(0, n_epochs, batch_size):

    end = min(start + batch_size, n_epochs)

    batch = np.asarray(faslt[start:end])

    if np.isnan(batch).any():
        has_nan = True
        print(f"NaN found in epochs {start}-{end}")

    if np.isinf(batch).any():
        has_inf = True
        print(f"Inf found in epochs {start}-{end}")

    print(f"Checked epochs {start}-{end}")

print("\n========== RESULTS ==========")

print(f"NaN check: {'FAILED' if has_nan else 'PASSED'}")
print(f"Inf check: {'FAILED' if has_inf else 'PASSED'}")

print(f"Global min: {faslt.min():.4f}")
print(f"Global max: {faslt.max():.4f}")

print("\nVerification complete.")
