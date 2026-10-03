import io
import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
import numpy as np
from sqlalchemy.orm import Session

from backend.database.database import get_db, get_db_optional
from backend.database.models import Patient, Prediction, UploadedFile
from backend.schemas.patient import (
    PatientDetailResponse,
    PredictionHistoryItem,
    UploadedFileResponse,
)
from backend.services.patient_prediction_service import predict_patient
from backend.services.storage_service import (
    StorageService,
    generate_safe_filename,
    get_upload_dir,
)

router = APIRouter(prefix="/patients", tags=["Patients & History"])



@router.get("/{patient_id}", response_model=PatientDetailResponse)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    """Retrieves patient demographics and aggregated prediction/file counts."""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")

    return PatientDetailResponse(
        id=patient.id,
        patient_id=patient.patient_id,
        age=patient.age,
        gender=patient.gender,
        created_at=patient.created_at,
        total_predictions=len(patient.predictions),
        total_files=len(patient.uploaded_files),
    )


@router.get("/{patient_id}/predictions", response_model=List[PredictionHistoryItem])
def get_patient_predictions(
    patient_id: str,
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Retrieves previous multimodal predictions for a patient, ordered newest first.
    Includes associated file metadata without exposing arbitrary host filesystem paths.
    """
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")

    predictions = (
        db.query(Prediction)
        .filter(Prediction.patient_id == patient.id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )

    result = []
    for pred in predictions:
        file_items = [
            UploadedFileResponse(
                id=f.id,
                file_type=f.file_type,
                original_filename=f.original_filename,
                stored_path=f.stored_path,
                mime_type=f.mime_type,
                file_size=f.file_size,
                uploaded_at=f.uploaded_at,
                prediction_id=f.prediction_id,
            )
            for f in pred.uploaded_files
        ]

        result.append(
            PredictionHistoryItem(
                id=pred.id,
                patient_id=patient.patient_id,
                cxr_prediction=pred.cxr_prediction,
                cxr_probability=pred.cxr_probability,
                multimodal_prediction=pred.multimodal_prediction,
                multimodal_probability=pred.multimodal_probability,
                uncertainty=pred.uncertainty,
                created_at=pred.created_at,
                files=file_items,
            )
        )

    return result


@router.get("/{patient_id}/files", response_model=List[UploadedFileResponse])
def get_patient_files(
    patient_id: str,
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Retrieves uploaded clinical file metadata for a patient, ordered newest first."""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")

    files = (
        db.query(UploadedFile)
        .filter(UploadedFile.patient_id == patient.id)
        .order_by(UploadedFile.uploaded_at.desc())
        .limit(limit)
        .all()
    )

    return [
        UploadedFileResponse(
            id=f.id,
            file_type=f.file_type,
            original_filename=f.original_filename,
            stored_path=f.stored_path,
            mime_type=f.mime_type,
            file_size=f.file_size,
            uploaded_at=f.uploaded_at,
            prediction_id=f.prediction_id,
        )
        for f in files
    ]


@router.get("/demo/samples")
def get_demo_samples():
    """
    Returns available real sample files from the repository to assist presentation testing:
    - Sample CXR image metadata
    - Sample ECG waveforms count and indices
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    cxr_normal_dir = project_root / "archive" / "chest_xray" / "test" / "NORMAL"
    cxr_pneumonia_dir = project_root / "archive" / "chest_xray" / "test" / "PNEUMONIA"

    sample_images = []
    if cxr_normal_dir.exists():
        for p in list(cxr_normal_dir.glob("*.jpeg"))[:3]:
            sample_images.append({"name": p.name, "category": "NORMAL", "relative_path": str(p.relative_to(project_root))})
    if cxr_pneumonia_dir.exists():
        for p in list(cxr_pneumonia_dir.glob("*.jpeg"))[:3]:
            sample_images.append({"name": p.name, "category": "PNEUMONIA", "relative_path": str(p.relative_to(project_root))})

    ecg_path = project_root / "features" / "ecg_waveforms" / "ecg_waveforms.npy"
    has_ecg = ecg_path.exists()

    return {
        "sample_cxr_images": sample_images,
        "sample_ecg_available": has_ecg,
        "sample_ecg_indices": [0, 499, 601] if has_ecg else [],
    }


@router.get("/demo/image")
def get_demo_image(relative_path: str = Query(..., description="Relative path of sample image")):
    """Returns a sample CXR image directly for UI preview."""
    project_root = Path(__file__).resolve().parent.parent.parent
    target = (project_root / relative_path).resolve()
    if not target.exists() or not str(target).startswith(str(project_root)):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(target, media_type="image/jpeg")


@router.post("/analyze")
async def analyze_patient(
    patient_id: str = Form(..., description="Unique patient identifier"),
    age: float = Form(..., ge=0, le=120, description="Patient age (0 to 120)"),
    gender: str = Form(..., description="Gender 'M' or 'F'"),
    image: Optional[UploadFile] = File(None, description="Chest X-Ray image file (.jpeg/.png)"),
    sample_image_path: Optional[str] = Form(None, description="Optional path to existing repository CXR image"),
    ecg_file: Optional[UploadFile] = File(None, description="12-lead ECG file (.npy or .json)"),
    sample_ecg_index: Optional[int] = Form(None, description="Optional index of sample ECG from dataset"),
    db: Optional[Session] = Depends(get_db_optional),
):
    """
    Orchestrates complete patient-level multimodal clinical analysis:
    1. Validates Patient demographics.
    2. Ingests CXR image (uploaded or selected sample).
    3. Ingests 12-lead ECG waveform (uploaded .npy/.json or selected sample).
    4. Executes Member 1 ResNet-50 visual feature extraction (F_img [1, 2048]).
    5. Executes Member 3 Cross-Attentive Fusion.
    6. Persists Patient, Prediction, and UploadedFile records to PostgreSQL.
    """
    clean_gender = gender.strip().upper()
    if clean_gender not in ("M", "F"):
        raise HTTPException(status_code=422, detail="Gender must be 'M' or 'F'")

    project_root = Path(__file__).resolve().parent.parent.parent
    upload_root = get_upload_dir()

    # 1. Resolve CXR image
    image_temp_path = None
    if image is not None and image.filename:
        # User uploaded a file
        contents = await image.read()
        safe_name = generate_safe_filename(image.filename, prefix="temp_cxr")
        image_temp_path = upload_root / safe_name
        image_temp_path.write_bytes(contents)
    elif sample_image_path:
        # User picked a repository sample image
        resolved = (project_root / sample_image_path).resolve()
        if not resolved.exists() or not str(resolved).startswith(str(project_root)):
            raise HTTPException(status_code=400, detail="Invalid sample image path specified")
        image_temp_path = resolved
    else:
        raise HTTPException(status_code=422, detail="Chest X-Ray image is required (upload a file or select a sample)")

    # 2. Resolve ECG waveform [12, 1000]
    ecg_array = None
    if ecg_file is not None and ecg_file.filename:
        content = await ecg_file.read()
        fname = ecg_file.filename.lower()
        if fname.endswith(".npy"):
            try:
                ecg_array = np.load(io.BytesIO(content))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read .npy ECG array: {e}")
        elif fname.endswith(".json"):
            try:
                parsed = json.loads(content.decode("utf-8"))
                if isinstance(parsed, dict) and "ecg_waveform" in parsed:
                    ecg_array = np.array(parsed["ecg_waveform"], dtype=np.float32)
                else:
                    ecg_array = np.array(parsed, dtype=np.float32)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse JSON ECG: {e}")
        else:
            raise HTTPException(status_code=400, detail="ECG file must be .npy or .json")
    elif sample_ecg_index is not None:
        ecg_path = project_root / "features" / "ecg_waveforms" / "ecg_waveforms.npy"
        if not ecg_path.exists():
            raise HTTPException(status_code=500, detail="Sample ECG dataset file not found on server")
        all_ecg = np.load(ecg_path)
        if sample_ecg_index < 0 or sample_ecg_index >= len(all_ecg):
            raise HTTPException(status_code=400, detail=f"Sample ECG index out of bounds (0-{len(all_ecg)-1})")
        ecg_array = all_ecg[sample_ecg_index]
    else:
        raise HTTPException(status_code=422, detail="12-lead ECG input is required (upload .npy/.json or select a sample)")

    # Validate ECG shape [12, 1000]
    if ecg_array.shape != (12, 1000):
        raise HTTPException(
            status_code=422,
            detail=f"ECG array must have shape (12, 1000), got {list(ecg_array.shape)}",
        )

    # 3. Execute Multimodal Prediction and Database Persistence
    try:
        prediction_result = predict_patient(
            image_path=image_temp_path,
            age=age,
            gender=clean_gender,
            ecg_waveform=ecg_array,
            patient_id=patient_id.strip(),
            save_features=True,
            db=db,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal inference failed: {str(e)}")

    return prediction_result
