import numpy as np
import matplotlib.pyplot as plt

denoised = np.load("data/processed/eeg_denoised.npy", mmap_mode="r")
metadata = np.load("data/processed/eeg_denoised_metadata.npz", allow_pickle=True)
subject_ids = metadata["subject_ids"]
y = metadata["y"]

# Subject14 rest epochs (label 0)
mask_14_rest = (subject_ids == "Subject14") & (y == 0)
mask_other_rest = (subject_ids == "Subject00") & (y == 0)  # any subject that's classified correctly

idx_14 = np.where(mask_14_rest)[0][0]
idx_other = np.where(mask_other_rest)[0][0]

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(denoised[idx_14, 0])  # channel 0
axes[0].set_title("Subject14 rest, channel 0")
axes[1].plot(denoised[idx_other, 0])
axes[1].set_title("Subject00 rest, channel 0")
plt.tight_layout()
plt.savefig("subject14_check.png")