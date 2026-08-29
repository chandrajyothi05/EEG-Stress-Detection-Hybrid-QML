import numpy as np

denoised = np.load(
    "data/processed/eeg_denoised.npy",
    mmap_mode="r"
)

metadata = np.load(
    "data/processed/eeg_denoised_metadata.npz",
    allow_pickle=True
)

y = metadata["y"]
subject_ids = metadata["subject_ids"]

print(f"Denoised shape: {denoised.shape}")
print(f"Denoised dtype: {denoised.dtype}")
print(f"y shape: {y.shape}")
print(f"subject_ids shape: {subject_ids.shape}")

# Check that every epoch has a matching label and subject ID
assert denoised.shape[0] == y.shape[0] == subject_ids.shape[0]

print("Alignment check: PASSED")
print(f"Label counts: {np.bincount(y)}")