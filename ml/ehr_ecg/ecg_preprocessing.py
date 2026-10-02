
from pathlib import Path
from fractions import Fraction

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from .config import (
    ECG_FILES_DIR,
    ECG_RECORD_LIST,
    ECG_TARGET_SAMPLING_RATE,
    ECG_DURATION_SECONDS,
    PROJECT_ROOT,
)


def load_record_list() -> pd.DataFrame:
    """Load metadata for available ECG recordings."""
    if not ECG_RECORD_LIST.exists():
        raise FileNotFoundError(
            f"ECG record list not found: {ECG_RECORD_LIST}"
        )

    records = pd.read_csv(ECG_RECORD_LIST)

    required_columns = {
        "subject_id",
        "study_id",
        "file_name",
        "path",
    }
    missing = required_columns - set(records.columns)

    if missing:
        raise ValueError(
            f"ECG metadata is missing columns: {sorted(missing)}"
        )

    return records


def preprocess_ecg(record_path: Path) -> np.ndarray:
    """
    Read and resample one ECG, then truncate or zero-pad it.

    Handle non-finite values in each lead using interpolation.

    Returns:
        Float32 array shaped (12, target_length).
    """
    signal, fields = wfdb.rdsamp(str(record_path))

    original_rate = float(fields["fs"])

    if original_rate <= 0:
        raise ValueError(
            f"Invalid sampling rate: {original_rate}"
        )

    # Resample to the configured sampling rate.
    if original_rate != ECG_TARGET_SAMPLING_RATE:
        ratio = Fraction(
            ECG_TARGET_SAMPLING_RATE / original_rate
        ).limit_denominator(1000)

        signal = resample_poly(
            signal,
            ratio.numerator,
            ratio.denominator,
            axis=0,
        )

    target_length = (
        ECG_TARGET_SAMPLING_RATE * ECG_DURATION_SECONDS
    )

    # Truncate long signals or zero-pad short signals.
    if signal.shape[0] >= target_length:
        signal = signal[:target_length, :]
    else:
        padding = target_length - signal.shape[0]
        signal = np.pad(
            signal,
            ((0, padding), (0, 0)),
            mode="constant",
        )

    # Convert from (time_steps, leads) to (leads, time_steps).
    signal = signal.T.astype(np.float32)

    if signal.shape[0] != 12:
        raise ValueError(
            f"Expected 12 ECG leads, got {signal.shape[0]}"
        )

    # Handle NaN and infinite values in each ECG lead.
    for lead_idx in range(signal.shape[0]):
        lead = signal[lead_idx]
        valid = np.isfinite(lead)

        if not valid.any():
            # No valid samples in this lead.
            signal[lead_idx] = 0.0

        elif not valid.all():
            # Interpolate missing samples using valid samples.
            sample_indices = np.arange(len(lead))

            signal[lead_idx] = np.interp(
                sample_indices,
                sample_indices[valid],
                lead[valid],
            )

    # Confirm that all values are finite after cleaning.
    if not np.isfinite(signal).all():
        raise ValueError(
            "Non-finite values remain after cleaning"
        )

    return signal


def main() -> None:
    records = load_record_list()

    print(f"ECG metadata rows: {len(records)}")

    output_dir = PROJECT_ROOT / "features" / "ecg_waveforms"
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_signals = []
    successful_rows = []
    failed_rows = []

    for row_number, (_, row) in enumerate(
        records.iterrows(),
        start=1,
    ):
        relative_path = Path(str(row["path"]))
        record_path = ECG_FILES_DIR.parent / relative_path

        try:
            header_path = record_path.with_suffix(".hea")

            if not header_path.exists():
                raise FileNotFoundError(
                    f"ECG header not found: {header_path}"
                )

            signal = preprocess_ecg(record_path)

            # Keep metadata in the same order as the waveform array.
            successful_rows.append(row.to_dict())
            successful_signals.append(signal)

        except Exception as exc:
            failed_rows.append({
                "subject_id": row.get("subject_id"),
                "study_id": row.get("study_id"),
                "file_name": row.get("file_name"),
                "path": row.get("path"),
                "error": str(exc),
            })

        if (
            row_number % 50 == 0
            or row_number == len(records)
        ):
            print(
                f"Processed metadata rows: "
                f"{row_number}/{len(records)} | "
                f"Successful: {len(successful_signals)} | "
                f"Failed: {len(failed_rows)}"
            )

    # Save successfully processed recordings and their index.
    waveform_file = output_dir / "ecg_waveforms.npy"
    index_file = output_dir / "ecg_waveform_index.csv"
    failures_file = output_dir / "ecg_processing_errors.csv"

    if successful_signals:
        waveforms = np.stack(
            successful_signals,
            axis=0,
        )

        np.save(waveform_file, waveforms)

        index_df = pd.DataFrame(successful_rows)
        index_df.insert(
            0,
            "waveform_index",
            np.arange(len(index_df)),
        )
        index_df.to_csv(index_file, index=False)

        print(f"\nSaved waveform array: {waveform_file}")
        print(
            "Array shape (recordings, leads, time): "
            f"{waveforms.shape}"
        )
        print(f"Saved waveform index: {index_file}")

    else:
        print("No ECG recordings were successfully processed.")

    # Always refresh the error report, including when there are no failures.
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(
            failures_file,
            index=False,
        )
        print(f"Saved failure details: {failures_file}")

    elif failures_file.exists():
        failures_file.unlink()
        print("No failures remain; removed the old error report.")

    print("\nECG preprocessing complete.")
    print(f"Successful recordings: {len(successful_signals)}")
    print(f"Failed recordings: {len(failed_rows)}")


if __name__ == "__main__":
    main()
