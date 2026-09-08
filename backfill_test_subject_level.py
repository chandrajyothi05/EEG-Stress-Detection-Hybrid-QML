"""
backfill_test_subject_level.py

Runs test-split subject-level (majority-vote) evaluation for the three
classical branches (ConvMixer, EfficientNet, Bi-LSTM+Attention), whose
subject-level numbers so far were only ever reported on val. This closes
the gap flagged when the VQC test-split number (11/16) turned out not to
be directly comparable to the others' val-split numbers -- gives a full
val+test subject-level table across all four branches for the report.

Reuses each branch's own subject_level_eval / evaluate_subject_level
function unchanged, just pointed at the "test" split instead of "val".
"""

import torch

from preprocessing.faslt_dataset import FASLTDataset
from preprocessing.azimuthal_dataset import AzimuthalDataset
from models.convmixer import ConvMixer
from models.efficientnet_branch import EfficientNetBranch
from models.bilstm_attention import FusionBiLSTMAttention

from train_convmixer import evaluate_subject_level as convmixer_eval
from train_efficientnet import subject_level_eval as efficientnet_eval
from train_bilstm import subject_level_eval as bilstm_eval, SequenceDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_convmixer():
    print("\n" + "=" * 60)
    print("ConvMixer -- test split")
    print("=" * 60)
    model = ConvMixer().to(DEVICE)
    model.load_state_dict(torch.load("convmixer_best.pt", map_location=DEVICE))
    test_ds = FASLTDataset("test")
    convmixer_eval(model, test_ds)


def run_efficientnet():
    print("\n" + "=" * 60)
    print("EfficientNet (partial unfreeze) -- test split")
    print("=" * 60)
    model = EfficientNetBranch(freeze_backbone=True).to(DEVICE)
    model.load_state_dict(torch.load("efficientnet_partial_unfreeze_best.pt", map_location=DEVICE))
    test_ds = AzimuthalDataset("test")
    efficientnet_eval(model, test_ds)


def run_bilstm():
    print("\n" + "=" * 60)
    print("Bi-LSTM + Attention -- test split")
    print("=" * 60)
    model = FusionBiLSTMAttention().to(DEVICE)
    model.load_state_dict(torch.load("bilstm_attention_best.pt", map_location=DEVICE))
    test_ds = SequenceDataset("test")
    bilstm_eval(model, test_ds)


if __name__ == "__main__":
    run_convmixer()
    run_efficientnet()
    run_bilstm()