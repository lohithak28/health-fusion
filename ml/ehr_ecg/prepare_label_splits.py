
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = PROJECT_ROOT / "features" / "labels" / "admission_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "features" / "labels" / "splits"

RANDOM_SEED = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def main():
    df = pd.read_csv(LABELS_PATH)

    required = {"subject_id", "hadm_id", "label", "waveform_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["subject_id"].isna().any() or df["label"].isna().any():
        raise ValueError("Patient IDs and labels must not be missing.")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0 or 1.")

    if not abs(TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE - 1.0) < 1e-9:
        raise ValueError("Split proportions must sum to 1.")

    # First separate training patients from the remaining patients.
    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
    )
    train_idx, remaining_idx = next(
        first_split.split(df, groups=df["subject_id"])
    )

    train = df.iloc[train_idx].copy()
    remaining = df.iloc[remaining_idx].copy()

    # Split remaining patients equally between validation and test.
    relative_test_size = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=RANDOM_SEED,
    )
    val_idx, test_idx = next(
        second_split.split(remaining, groups=remaining["subject_id"])
    )

    validation = remaining.iloc[val_idx].copy()
    test = remaining.iloc[test_idx].copy()

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # Check patient overlap.
    patient_sets = {
        name: set(part["subject_id"])
        for name, part in splits.items()
    }

    assert patient_sets["train"].isdisjoint(patient_sets["validation"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["validation"].isdisjoint(patient_sets["test"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, part in splits.items():
        part = part.sort_values(["subject_id", "hadm_id"])
        output_path = OUTPUT_DIR / f"{name}.csv"
        part.to_csv(output_path, index=False)

        print(f"\n{name.upper()}")
        print("Admissions:", len(part))
        print("Unique patients:", part["subject_id"].nunique())
        print("Label counts:")
        print(part["label"].value_counts().reindex([0, 1], fill_value=0))
        print("Positive rate:", f"{part['label'].mean():.2%}")
        print("Saved:", output_path)

    print("\nPatient overlap check: PASSED")


if __name__ == "__main__":
    main()
