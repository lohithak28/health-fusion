import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# Base upload directory relative to project root
DEFAULT_UPLOAD_DIR = Path("backend/uploads")
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB max limit

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_ECG_EXTENSIONS = {".npy", ".json", ".csv"}


def get_upload_dir() -> Path:
    """Returns the base upload directory, creating it if it doesn't exist."""
    dir_path = Path(os.environ.get("UPLOAD_STORAGE_DIR", DEFAULT_UPLOAD_DIR))
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path.resolve()


def sanitize_filename(filename: str) -> str:
    """Sanitizes user filename, removing path traversal characters and non-safe symbols."""
    # Strip any directory components
    clean_name = Path(filename).name
    # Remove potentially dangerous characters, allowing letters, digits, dots, dashes, underscores
    clean_name = re.sub(r"[^\w\.\-]", "_", clean_name)
    if not clean_name or clean_name.startswith("."):
        clean_name = f"upload_{uuid.uuid4().hex[:8]}"
    return clean_name


def generate_safe_filename(original_filename: str, prefix: str = "") -> str:
    """Generates a collision-resistant safe filename preserving the original extension."""
    clean_name = sanitize_filename(original_filename)
    ext = Path(clean_name).suffix.lower()
    unique_id = uuid.uuid4().hex[:12]
    base_stem = Path(clean_name).stem[:24]
    prefix_str = f"{prefix}_" if prefix else ""
    return f"{prefix_str}{base_stem}_{unique_id}{ext}"


def resolve_safe_path(stored_relative_path: str) -> Path:
    """
    Resolves a stored relative path and guarantees it resides within the upload root.
    Prevents path traversal attacks (e.g., ../).
    """
    base_dir = get_upload_dir()
    resolved = (base_dir / stored_relative_path).resolve()
    if not str(resolved).startswith(str(base_dir)):
        raise ValueError(f"Path traversal detected: {stored_relative_path}")
    return resolved


class StorageService:
    @staticmethod
    def save_cxr_file(
        patient_id: str,
        file_bytes: bytes,
        original_filename: str,
        prediction_id: Optional[str] = None,
        mime_type: Optional[str] = "image/jpeg",
    ) -> Dict[str, Any]:
        """
        Saves an uploaded CXR image to persistent storage:
        backend/uploads/<patient_id>/[<prediction_id>/]cxr/<safe_filename>
        """
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size exceeds maximum allowed limit ({MAX_FILE_SIZE_BYTES} bytes)")

        clean_ext = Path(original_filename).suffix.lower()
        if clean_ext not in ALLOWED_IMAGE_EXTENSIONS:
            # Default to .jpeg if unknown image extension
            clean_ext = ".jpeg"

        clean_patient_id = sanitize_filename(patient_id)
        sub_dir = Path(clean_patient_id)
        if prediction_id:
            sub_dir = sub_dir / sanitize_filename(str(prediction_id)) / "cxr"
        else:
            sub_dir = sub_dir / "cxr"

        target_dir = get_upload_dir() / sub_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        stored_filename = generate_safe_filename(original_filename, prefix="cxr")
        target_path = target_dir / stored_filename
        target_path.write_bytes(file_bytes)

        # Store relative path for database portability
        rel_path = str(target_path.relative_to(get_upload_dir()))

        return {
            "file_type": "CXR",
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "stored_path": rel_path,
            "absolute_path": str(target_path),
            "mime_type": mime_type,
            "file_size": len(file_bytes),
        }

    @staticmethod
    def save_ecg_file(
        patient_id: str,
        ecg_waveform: Union[np.ndarray, List[List[float]]],
        original_filename: str = "ecg_waveform.npy",
        prediction_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Saves a 12-lead x 1000-sample ECG waveform array to persistent storage:
        backend/uploads/<patient_id>/[<prediction_id>/]ecg/<safe_filename>.npy
        """
        arr = np.asarray(ecg_waveform, dtype=np.float32)
        if arr.shape != (12, 1000):
            raise ValueError(f"ECG array must have shape (12, 1000), got {arr.shape}")

        clean_patient_id = sanitize_filename(patient_id)
        sub_dir = Path(clean_patient_id)
        if prediction_id:
            sub_dir = sub_dir / sanitize_filename(str(prediction_id)) / "ecg"
        else:
            sub_dir = sub_dir / "ecg"

        target_dir = get_upload_dir() / sub_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        stored_filename = generate_safe_filename(original_filename, prefix="ecg")
        if not stored_filename.endswith(".npy"):
            stored_filename += ".npy"

        target_path = target_dir / stored_filename
        np.save(str(target_path), arr)

        file_size = target_path.stat().st_size
        rel_path = str(target_path.relative_to(get_upload_dir()))

        return {
            "file_type": "ECG",
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "stored_path": rel_path,
            "absolute_path": str(target_path),
            "mime_type": "application/x-npy",
            "file_size": file_size,
        }
