
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "archive"

EHR_DIR = ARCHIVE_DIR / "mimic-iv-clinical-database-demo-2.2"
ECG_DIR = next(ARCHIVE_DIR.glob("mimic-iv-ecg-demo*"))

ecg = pd.read_csv(ECG_DIR / "record_list.csv")
admissions = pd.read_csv(EHR_DIR / "hosp" / "admissions.csv")

ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"], errors="coerce")
admissions["admittime"] = pd.to_datetime(
    admissions["admittime"], errors="coerce"
)
admissions["dischtime"] = pd.to_datetime(
    admissions["dischtime"], errors="coerce"
)
admissions["deathtime"] = pd.to_datetime(
    admissions["deathtime"], errors="coerce"
)

# Match ECG recordings to admissions for the same patient.
# Keep only ECGs recorded during the admission interval.
matched = ecg.merge(
    admissions[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "hospital_expire_flag",
        ]
    ],
    on="subject_id",
    how="left",
)

matched = matched[
    matched["ecg_time"].notna()
    & matched["admittime"].notna()
    & matched["dischtime"].notna()
    & (matched["ecg_time"] >= matched["admittime"])
    & (matched["ecg_time"] <= matched["dischtime"])
].copy()

print("ECG recordings:", len(ecg))
print("ECG recordings matched within admission:", len(matched))
print("Unique matched patients:", matched["subject_id"].nunique())
print("Unique matched admissions:", matched["hadm_id"].nunique())

if not matched.empty:
    print("\nAdmission outcome counts for matched ECG recordings:")
    print(
        matched.groupby("hadm_id")["hospital_expire_flag"]
        .first()
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    # Check whether ECGs occur before the recorded death time.
    matched["ecg_before_death"] = (
        matched["deathtime"].isna()
        | (matched["ecg_time"] <= matched["deathtime"])
    )

    print(
        "\nMatched ECGs recorded no later than death time:",
        int(matched["ecg_before_death"].sum()),
        "of",
        len(matched),
    )

    print(
        "\nPositive-outcome admissions with a matched ECG:",
        matched.loc[
            matched["hospital_expire_flag"] == 1, "hadm_id"
        ].nunique(),
    )

    print(
        "\nDistinct matched admission outcomes:"
    )
    print(
        matched.drop_duplicates("hadm_id")[
            ["hadm_id", "hospital_expire_flag"]
        ].groupby("hospital_expire_flag")["hadm_id"]
        .nunique()
        .to_string()
    )
else:
    print("No ECG recordings matched the admission intervals.")
