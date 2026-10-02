
from pathlib import Path

import pandas as pd

from .config import EHR_HOSP_DIR, EHR_ICU_DIR


def load_table(file_path: Path) -> pd.DataFrame:
    """Load a MIMIC-IV CSV or CSV.GZ table."""
    if file_path.exists():
        return pd.read_csv(file_path, low_memory=False)

    gz_path = file_path.with_suffix(file_path.suffix + ".gz")

    if gz_path.exists():
        return pd.read_csv(gz_path, low_memory=False)

    raise FileNotFoundError(
        f"Table not found: {file_path} or {gz_path}"
    )


def load_patients() -> pd.DataFrame:
    """Load patient demographics from the hospital tables."""
    return load_table(EHR_HOSP_DIR / "patients.csv")


def load_admissions() -> pd.DataFrame:
    """Load hospital admission records."""
    return load_table(EHR_HOSP_DIR / "admissions.csv")


def load_diagnoses() -> pd.DataFrame:
    """Load diagnosis codes for hospital admissions."""
    return load_table(EHR_HOSP_DIR / "diagnoses_icd.csv")


def load_icu_stays() -> pd.DataFrame:
    """Load ICU stay records."""
    return load_table(EHR_ICU_DIR / "icustays.csv")


def build_patient_table() -> pd.DataFrame:
    """
    Build one row per patient with available demographic information
    and a count of hospital admissions.

    This is an initial baseline table, not a clinical prediction label.
    """
    patients = load_patients()
    admissions = load_admissions()

    admission_counts = (
        admissions.groupby("subject_id")
        .size()
        .rename("admission_count")
        .reset_index()
    )

    patient_table = patients.merge(
        admission_counts,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )

    patient_table["admission_count"] = (
        patient_table["admission_count"].fillna(0).astype("int64")
    )

    return patient_table


def main() -> None:
    patients = load_patients()
    admissions = load_admissions()
    diagnoses = load_diagnoses()
    icu_stays = load_icu_stays()

    print(f"Patients: {len(patients)}")
    print(f"Admissions: {len(admissions)}")
    print(f"Diagnosis rows: {len(diagnoses)}")
    print(f"ICU stays: {len(icu_stays)}")

    patient_table = build_patient_table()
    output_path = Path(__file__).resolve().parents[2] / "features"
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "ehr_patient_features.csv"
    patient_table.to_csv(output_file, index=False)

    print(f"Saved patient-level features to: {output_file}")


if __name__ == "__main__":
    main()
