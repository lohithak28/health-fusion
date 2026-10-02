
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT


def build_matched_table() -> pd.DataFrame:
    """
    Join the ECG waveform index with patient-level EHR features.

    Each row represents one ECG recording. The waveform_index
    column maps the row to the corresponding entry in ecg_waveforms.npy.
    """
    features_dir = PROJECT_ROOT / "features"
    ecg_dir = features_dir / "ecg_waveforms"

    ehr_file = features_dir / "ehr_patient_features.csv"
    ecg_index_file = ecg_dir / "ecg_waveform_index.csv"
    waveform_file = ecg_dir / "ecg_waveforms.npy"

    for file_path in (ehr_file, ecg_index_file, waveform_file):
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    ehr = pd.read_csv(ehr_file)
    ecg_index = pd.read_csv(ecg_index_file)
    waveforms = np.load(waveform_file, mmap_mode="r")

    if "subject_id" not in ehr.columns:
        raise ValueError("EHR table must contain subject_id.")

    if "subject_id" not in ecg_index.columns:
        raise ValueError("ECG index must contain subject_id.")

    if "waveform_index" not in ecg_index.columns:
        raise ValueError(
            "ECG index must contain waveform_index."
        )

    if ehr["subject_id"].duplicated().any():
        raise ValueError(
            "EHR features must contain one row per patient."
        )

    if len(ecg_index) != waveforms.shape[0]:
        raise ValueError(
            "ECG index row count does not match waveform count."
        )

    expected_indices = np.arange(len(ecg_index))
    actual_indices = ecg_index["waveform_index"].to_numpy()

    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            "Waveform indices are not aligned with array rows."
        )

    matched = ecg_index.merge(
        ehr,
        on="subject_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = matched["_merge"] != "both"

    if unmatched.any():
        missing_ids = (
            matched.loc[unmatched, "subject_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"Some ECG patients have no EHR features: {missing_ids}"
        )

    matched = matched.drop(columns="_merge")

    return matched


def main() -> None:
    matched = build_matched_table()

    output_file = (
        PROJECT_ROOT
        / "features"
        / "ehr_ecg_integrated.csv"
    )

    matched.to_csv(output_file, index=False)

    print(f"Integrated recording rows: {len(matched)}")
    print(
        "Unique patients:",
        matched["subject_id"].nunique(),
    )
    print(
        "Unique studies:",
        matched["study_id"].nunique(),
    )
    print(
        "Waveform indices aligned:",
        matched["waveform_index"].is_unique
        and matched["waveform_index"].min() == 0
        and matched["waveform_index"].max() == len(matched) - 1,
    )
    print(f"Saved integrated table to: {output_file}")
    print("Columns:", matched.columns.tolist())


if __name__ == "__main__":
    main()
