
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = PROJECT_ROOT / "features" / "labels" / "splits"
WAVEFORM_PATH = (
    PROJECT_ROOT / "features" / "ecg_waveforms" / "ecg_waveforms.npy"
)

EHR_COLUMNS = ["anchor_age", "gender"]


class EHR_ECGDataset(Dataset):
    def __init__(self, csv_path, waveform_path=WAVEFORM_PATH):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.waveforms = np.load(waveform_path, mmap_mode="r")

        required = {
            "waveform_index", "subject_id", "hadm_id",
            "label", *EHR_COLUMNS,
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        indices = self.df["waveform_index"].to_numpy(dtype=int)
        if np.any(indices < 0) or np.any(indices >= len(self.waveforms)):
            raise IndexError("A waveform_index is outside the waveform array.")

        if not set(self.df["label"].unique()).issubset({0, 1}):
            raise ValueError("Labels must be 0 or 1.")

        if self.df[EHR_COLUMNS].isna().any().any():
            raise ValueError("EHR features contain missing values.")

        # Encode gender as 0/1 and scale age to a roughly 0-1 range.
        gender = (
            self.df["gender"].astype(str).str.upper().map({"F": 0.0, "M": 1.0})
        )
        if gender.isna().any():
            raise ValueError("Unexpected or missing gender values.")

        age = pd.to_numeric(self.df["anchor_age"], errors="coerce")
        if age.isna().any():
            raise ValueError("anchor_age contains non-numeric values.")

        # Fixed scaling avoids fitting preprocessing on validation/test data.
        self.ehr_features = np.column_stack(
            [age.to_numpy(dtype=np.float32) / 100.0,
             gender.to_numpy(dtype=np.float32)]
        ).astype(np.float32)

        self.waveform_indices = indices
        self.labels = self.df["label"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        waveform = np.array(
            self.waveforms[self.waveform_indices[idx]],
            dtype=np.float32,
            copy=True,
        )
        ehr = self.ehr_features[idx]
        label = self.labels[idx]

        return {
            "ecg": torch.from_numpy(waveform),       # (12, 1000)
            "ehr": torch.from_numpy(ehr),             # (2,)
            "label": torch.tensor(label, dtype=torch.float32),
            "subject_id": int(self.df.iloc[idx]["subject_id"]),
            "hadm_id": int(self.df.iloc[idx]["hadm_id"]),
        }


def create_dataloaders(batch_size=16):
    train_ds = EHR_ECGDataset(LABEL_DIR / "train.csv")
    val_ds = EHR_ECGDataset(LABEL_DIR / "validation.csv")
    test_ds = EHR_ECGDataset(LABEL_DIR / "test.csv")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    batch = next(iter(train_loader))
    print("ECG batch shape:", tuple(batch["ecg"].shape))
    print("EHR batch shape:", tuple(batch["ehr"].shape))
    print("Label batch shape:", tuple(batch["label"].shape))
    print("Labels in sample batch:", batch["label"].tolist())
    print("ECG finite:", torch.isfinite(batch["ecg"]).all().item())
    print("EHR finite:", torch.isfinite(batch["ehr"]).all().item())
