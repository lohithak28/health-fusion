
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "archive"

EHR_DIR = ARCHIVE_DIR / "mimic-iv-clinical-database-demo-2.2"
ECG_DIR = next(ARCHIVE_DIR.glob("mimic-iv-ecg-demo*"), None)


def inspect_csv(path, name, rows=5):
    print(f"\n{'=' * 65}")
    print(name)
    print(f"File: {path}")

    if not path.exists():
        print("FILE NOT FOUND")
        return None

    df = pd.read_csv(path, low_memory=False)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(rows).to_string(index=False))

    return df


def main():
    if ECG_DIR is None:
        raise FileNotFoundError("Could not find the ECG data folder.")

    print("ECG folder:", ECG_DIR)

    # ECG metadata: inspect what information is actually available.
    ecg = inspect_csv(
        ECG_DIR / "record_list.csv",
        "ECG RECORD LIST",
    )

    # EHR tables that may help define a clinical outcome.
    patients = inspect_csv(
        EHR_DIR / "hosp" / "patients.csv",
        "PATIENTS",
    )

    admissions = inspect_csv(
        EHR_DIR / "hosp" / "admissions.csv",
        "ADMISSIONS",
    )

    diagnoses = inspect_csv(
        EHR_DIR / "hosp" / "diagnoses_icd.csv",
        "DIAGNOSES ICD",
    )

    icustays = inspect_csv(
        EHR_DIR / "icu" / "icustays.csv",
        "ICU STAYS",
    )

    if diagnoses is not None:
        print("\nTOP DIAGNOSIS CODES")
        code_columns = [
            c for c in ["icd_version", "icd_code"]
            if c in diagnoses.columns
        ]

        if code_columns:
            print(
                diagnoses.groupby(code_columns, dropna=False)
                .size()
                .sort_values(ascending=False)
                .head(20)
                .to_string()
            )

        if "subject_id" in diagnoses.columns:
            print(
                "\nUnique patients with diagnosis rows:",
                diagnoses["subject_id"].nunique(),
            )

    if admissions is not None:
        if "hospital_expire_flag" in admissions.columns:
            print("\nHOSPITAL EXPIRATION FLAG COUNTS")
            print(
                admissions["hospital_expire_flag"]
                .value_counts(dropna=False)
                .sort_index()
                .to_string()
            )

        if "subject_id" in admissions.columns:
            print(
                "\nUnique patients with admissions:",
                admissions["subject_id"].nunique(),
            )

    if ecg is not None and "subject_id" in ecg.columns:
        ecg_patients = set(ecg["subject_id"].dropna().unique())
        print("\nECG patients:", len(ecg_patients))

        if diagnoses is not None and "subject_id" in diagnoses.columns:
            diagnosis_patients = set(
                diagnoses["subject_id"].dropna().unique()
            )
            print(
                "ECG patients with diagnosis records:",
                len(ecg_patients & diagnosis_patients),
            )

        if admissions is not None and "subject_id" in admissions.columns:
            admission_patients = set(
                admissions["subject_id"].dropna().unique()
            )
            print(
                "ECG patients with admission records:",
                len(ecg_patients & admission_patients),
            )


if __name__ == "__main__":
    main()
