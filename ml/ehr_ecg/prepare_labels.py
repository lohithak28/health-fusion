
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "archive"

EHR_DIR = ARCHIVE_DIR / "mimic-iv-clinical-database-demo-2.2"
ECG_DIR = next(ARCHIVE_DIR.glob("mimic-iv-ecg-demo*"))

FEATURES_DIR = PROJECT_ROOT / "features"
OUTPUT_DIR = FEATURES_DIR / "labels"

ECG_INDEX_PATH = FEATURES_DIR / "ecg_waveforms" / "ecg_waveform_index.csv"
EHR_FEATURES_PATH = FEATURES_DIR / "ehr_patient_features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load ECG index, EHR features, and admission outcomes.
    ecg = pd.read_csv(ECG_INDEX_PATH)
    ehr = pd.read_csv(EHR_FEATURES_PATH)
    admissions_path = EHR_DIR / "hosp" / "admissions.csv"

    if not admissions_path.exists():
        admissions_path = EHR_DIR / "hosp" / "admissions.csv.gz"

    admissions = pd.read_csv(admissions_path)

    # Parse timestamps.
    ecg["ecg_time"] = pd.to_datetime(
        ecg["ecg_time"], errors="coerce"
    )
    admissions["admittime"] = pd.to_datetime(
        admissions["admittime"], errors="coerce"
    )
    admissions["dischtime"] = pd.to_datetime(
        admissions["dischtime"], errors="coerce"
    )
    admissions["deathtime"] = pd.to_datetime(
        admissions["deathtime"], errors="coerce"
    )

    # Keep only the admission fields needed for label construction.
    admission_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
    ]
    admission_data = admissions[admission_cols].copy()

    # Match ECGs to admissions for the same patient.
    matched = ecg.merge(
        admission_data,
        on="subject_id",
        how="inner",
        validate="many_to_many",
    )

    # Require a valid ECG timestamp within the admission interval.
    matched = matched[
        matched["ecg_time"].notna()
        & matched["admittime"].notna()
        & matched["dischtime"].notna()
        & (matched["ecg_time"] >= matched["admittime"])
        & (matched["ecg_time"] <= matched["dischtime"])
    ].copy()

    if matched.empty:
        raise RuntimeError(
            "No ECGs matched admission intervals. "
            "Check the ECG and admission timestamps."
        )

    # Exclude ECGs recorded after the recorded death time.
    # This does not by itself guarantee an early-warning prediction.
    matched = matched[
        matched["deathtime"].isna()
        | (matched["ecg_time"] <= matched["deathtime"])
    ].copy()

    # Select the earliest eligible ECG for each admission.
    matched = matched.sort_values(
        ["hadm_id", "ecg_time", "waveform_index"]
    )
    admission_examples = matched.drop_duplicates(
        subset=["hadm_id"], keep="first"
    ).copy()

    # Ensure labels are binary and present.
    admission_examples["hospital_expire_flag"] = pd.to_numeric(
        admission_examples["hospital_expire_flag"],
        errors="coerce",
    )
    admission_examples = admission_examples[
        admission_examples["hospital_expire_flag"].isin([0, 1])
    ].copy()
    admission_examples["label"] = (
        admission_examples["hospital_expire_flag"].astype("int64")
    )

    # Add patient-level EHR features.
    ehr_columns = [
        col for col in
        ["subject_id", "gender", "anchor_age", "admission_count"]
        if col in ehr.columns
    ]
    if "subject_id" not in ehr_columns:
        raise ValueError("EHR features must contain subject_id.")

    ehr_for_merge = ehr[ehr_columns].copy()

    if ehr_for_merge["subject_id"].duplicated().any():
        raise ValueError(
            "EHR features contain duplicate subject_id values."
        )

    admission_examples = admission_examples.merge(
        ehr_for_merge,
        on="subject_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    missing_ehr = int((admission_examples["_merge"] != "both").sum())
    admission_examples = admission_examples.drop(columns=["_merge"])

    if missing_ehr:
        print(
            f"Warning: {missing_ehr} admission examples have no EHR features."
        )

    # Save the admission-level examples.
    output_columns = [
        "waveform_index",
        "subject_id",
        "hadm_id",
        "study_id",
        "ecg_time",
        "admittime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
        "label",
    ] + [
        col for col in
        ["gender", "anchor_age", "admission_count"]
        if col in admission_examples.columns
    ]

    output_columns = [
        col for col in output_columns
        if col in admission_examples.columns
    ]

    output_path = OUTPUT_DIR / "admission_labels.csv"
    admission_examples[output_columns].to_csv(
        output_path, index=False
    )

    # Report class balance and patient counts.
    print("\nAdmission-level label preparation complete.")
    print("Matched ECG recordings before selection:", len(matched))
    print("Admission-level examples:", len(admission_examples))
    print(
        "Unique patients:",
        admission_examples["subject_id"].nunique(),
    )
    print(
        "Unique admissions:",
        admission_examples["hadm_id"].nunique(),
    )
    print("\nLabel counts (0 = survived, 1 = in-hospital death):")
    print(
        admission_examples["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print("\nPositive-label rate:")
    print(f"{admission_examples['label'].mean():.2%}")

    print("\nMissing values in selected EHR features:")
    feature_cols = [
        col for col in
        ["gender", "anchor_age", "admission_count"]
        if col in admission_examples.columns
    ]
    print(admission_examples[feature_cols].isna().sum().to_string())

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
