
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTEGRATED_CSV = (
    PROJECT_ROOT / "features" / "ehr_ecg_integrated.csv"
)

ECG_ARRAY = (
    PROJECT_ROOT
    / "features"
    / "ecg_waveforms"
    / "ecg_waveforms.npy"
)


class EHRECGDataset(Dataset):
    """PyTorch dataset for paired EHR features and ECG recordings."""

    def __init__(self):
        if not INTEGRATED_CSV.exists():
            raise FileNotFoundError(
                f"Integrated CSV not found: {INTEGRATED_CSV}"
            )

        if not ECG_ARRAY.exists():
            raise FileNotFoundError(
                f"ECG waveform array not found: {ECG_ARRAY}"
            )

        self.metadata = pd.read_csv(INTEGRATED_CSV)
        self.ecg_waveforms = np.load(ECG_ARRAY, mmap_mode="r")

        if len(self.metadata) != len(self.ecg_waveforms):
            raise ValueError(
                "The metadata row count does not match "
                "the ECG waveform count."
            )

        if not np.array_equal(
            self.metadata["waveform_index"].to_numpy(),
            np.arange(len(self.ecg_waveforms)),
        ):
            raise ValueError("Waveform indices are not correctly aligned.")

        # Use only basic EHR input features for this initial dataset.
        self.ehr_columns = [
            "anchor_age",
            "admission_count",
            "gender",
        ]

        missing = [
            col for col in self.ehr_columns
            if col not in self.metadata.columns
        ]
        if missing:
            raise ValueError(f"Missing EHR columns: {missing}")

        self.ehr_features = self.metadata[self.ehr_columns].copy()

        self.ehr_features["gender"] = (
            self.ehr_features["gender"]
            .astype(str)
            .str.upper()
            .map({"F": 0.0, "M": 1.0})
        )

        self.ehr_features = self.ehr_features.astype(np.float32)

        if not np.isfinite(self.ehr_features.to_numpy()).all():
            raise ValueError(
                "EHR features contain missing or non-finite values. "
                "Check the source data before training."
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        ecg = np.array(
            self.ecg_waveforms[index],
            dtype=np.float32,
            copy=True,
        )

        ehr = self.ehr_features.iloc[index].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        return {
            "ecg": torch.from_numpy(ecg),
            "ehr": torch.from_numpy(ehr),
            "subject_id": int(self.metadata.iloc[index]["subject_id"]),
            "study_id": str(self.metadata.iloc[index]["study_id"]),
            "waveform_index": int(
                self.metadata.iloc[index]["waveform_index"]
            ),
        }


if __name__ == "__main__":
    dataset = EHRECGDataset()

    print("Dataset loaded successfully!")
    print("Number of ECG recordings:", len(dataset))
    print("Number of EHR features:", len(dataset.ehr_columns))

    sample = dataset[0]

    print("ECG tensor shape:", tuple(sample["ecg"].shape))
    print("EHR tensor shape:", tuple(sample["ehr"].shape))
    print("ECG values finite:", torch.isfinite(sample["ecg"]).all().item())
    print("EHR values finite:", torch.isfinite(sample["ehr"]).all().item())
    print("Sample patient ID:", sample["subject_id"])
    print("Sample study ID:", sample["study_id"])
