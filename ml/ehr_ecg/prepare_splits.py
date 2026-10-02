
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTEGRATED_CSV = (
    PROJECT_ROOT / "features" / "ehr_ecg_integrated.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "features" / "splits"


def main():
    if not INTEGRATED_CSV.exists():
        raise FileNotFoundError(
            f"Integrated dataset not found: {INTEGRATED_CSV}"
        )

    df = pd.read_csv(INTEGRATED_CSV)

    required_columns = {"subject_id", "waveform_index"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["subject_id"].isna().any():
        raise ValueError("Some recordings have no subject_id.")

    if df["subject_id"].duplicated().all():
        print("Note: patients have multiple recordings; grouping by patient.")

    # Split unique patients first, not individual ECG recordings.
    patients = df["subject_id"].drop_duplicates().to_numpy()

    if len(patients) < 3:
        raise ValueError("At least 3 unique patients are needed.")

    # 70% train, 30% temporary (validation + test).
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42,
    )

    train_patient_idx, temp_patient_idx = next(
        first_split.split(patients, groups=patients)
    )

    train_patients = set(patients[train_patient_idx])
    temp_patients = patients[temp_patient_idx]

    # Split temporary patients equally into validation and test.
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    val_patient_idx, test_patient_idx = next(
        second_split.split(temp_patients, groups=temp_patients)
    )

    val_patients = set(temp_patients[val_patient_idx])
    test_patients = set(temp_patients[test_patient_idx])

    train_df = df[df["subject_id"].isin(train_patients)].copy()
    val_df = df[df["subject_id"].isin(val_patients)].copy()
    test_df = df[df["subject_id"].isin(test_patients)].copy()

    # Confirm no patient appears in more than one split.
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    # Confirm every recording is assigned exactly once.
    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows == len(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("Patient-level splitting completed!")
    print(f"Total recordings: {len(df)}")
    print(f"Total unique patients: {len(patients)}")

    for name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(
            f"{name}: {len(split_df)} recordings, "
            f"{split_df['subject_id'].nunique()} patients"
        )

    print("Patient overlap check: passed")
    print(f"Split files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
