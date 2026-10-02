
from pathlib import Path

# Project root: health-fusion/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset directories
ARCHIVE_DIR = PROJECT_ROOT / "archive"

EHR_DATA_DIR = ARCHIVE_DIR / "mimic-iv-clinical-database-demo-2.2"
ECG_DATA_DIR = next(
    ARCHIVE_DIR.glob("mimic-iv-ecg-demo*"),
    ARCHIVE_DIR / "mimic-iv-ecg-demo",
)

# EHR tables
EHR_HOSP_DIR = EHR_DATA_DIR / "hosp"
EHR_ICU_DIR = EHR_DATA_DIR / "icu"

# ECG waveform files and metadata
ECG_FILES_DIR = ECG_DATA_DIR / "files"
ECG_RECORD_LIST = ECG_DATA_DIR / "record_list.csv"

# Reproducibility
RANDOM_SEED = 42

# ECG preprocessing
ECG_TARGET_SAMPLING_RATE = 100
ECG_DURATION_SECONDS = 10
