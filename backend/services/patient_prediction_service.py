from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from sqlalchemy.orm import Session
import torch

from backend.database.models import Patient, Prediction, UploadedFile
from backend.services.image_service import analyze_chest_xray
from backend.services.fusion_service import predict_multimodal
from backend.services.storage_service import StorageService


def predict_patient(
    image_path: Union[str, Path],
    age: Union[int, float],
    gender: str,
    ecg_waveform: Union[torch.Tensor, List[List[float]], Any],
    patient_id: str,
    save_features: bool = True,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Coordinates end-to-end patient-level multimodal clinical prediction.

    Pipeline Flow:
      1. CXR image path -> image_service.analyze_chest_xray() -> Member 1 ResNet-50 -> F_img [1, 2048]
      2. Age & Gender -> EHR representation
      3. 12-lead ECG waveform -> Sensor representation [1, 12, 1000]
      4. All modalities -> fusion_service.predict_multimodal() -> Member 3 Cross-Attention Fusion -> Prediction
      5. When db is provided: persists Patient, Prediction, and UploadedFile records in PostgreSQL.

    Returns clean clinical prediction results without exposing internal 2048-D feature vectors.
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"CXR image file not found at: {image_file}")

    # 1. Member 1: Chest X-ray inference and F_img extraction
    cxr_result = analyze_chest_xray(image_file)
    f_img = cxr_result["features"]  # torch.Tensor [1, 2048]

    # Optionally persist the 2048-D feature vector alongside image
    feature_path = None
    if save_features:
        feature_path = image_file.with_suffix(".pt")
        torch.save(f_img, feature_path)

    # 2. Member 3: Multimodal cross-attention fusion inference
    fusion_result = predict_multimodal(
        image_features=f_img,
        age=age,
        gender=gender,
        ecg_waveform=ecg_waveform,
    )

    db_prediction_id = None

    # 3. Database persistence when session is provided
    if db is not None:
        try:
            # Find or create patient
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
            if not patient:
                patient = Patient(
                    patient_id=patient_id,
                    age=float(age),
                    gender=str(gender).strip().upper(),
                )
                db.add(patient)
                db.flush()
            else:
                patient.age = float(age)
                patient.gender = str(gender).strip().upper()

            # Create prediction record
            pred_record = Prediction(
                patient_id=patient.id,
                cxr_prediction=cxr_result["prediction"],
                cxr_probability=round(cxr_result["probability"], 4),
                multimodal_prediction=fusion_result["prediction"],
                multimodal_probability=round(fusion_result["probability"], 4),
                uncertainty=round(1.0 - fusion_result["probability"], 4),
            )
            db.add(pred_record)
            db.flush()
            db_prediction_id = pred_record.id

            # Persist CXR file metadata and bytes
            cxr_meta = StorageService.save_cxr_file(
                patient_id=patient_id,
                file_bytes=image_file.read_bytes(),
                original_filename=image_file.name,
                prediction_id=str(pred_record.id),
            )
            db_cxr = UploadedFile(
                patient_id=patient.id,
                prediction_id=pred_record.id,
                file_type="CXR",
                original_filename=cxr_meta["original_filename"],
                stored_filename=cxr_meta["stored_filename"],
                stored_path=cxr_meta["stored_path"],
                mime_type=cxr_meta["mime_type"],
                file_size=cxr_meta["file_size"],
            )
            db.add(db_cxr)

            # Persist ECG file metadata and array
            ecg_meta = StorageService.save_ecg_file(
                patient_id=patient_id,
                ecg_waveform=ecg_waveform,
                original_filename="ecg_waveform.npy",
                prediction_id=str(pred_record.id),
            )
            db_ecg = UploadedFile(
                patient_id=patient.id,
                prediction_id=pred_record.id,
                file_type="ECG",
                original_filename=ecg_meta["original_filename"],
                stored_filename=ecg_meta["stored_filename"],
                stored_path=ecg_meta["stored_path"],
                mime_type=ecg_meta["mime_type"],
                file_size=ecg_meta["file_size"],
            )
            db.add(db_ecg)

            db.commit()
        except Exception:
            db.rollback()
            raise

    return {
        "patient_id": patient_id,
        "prediction_id": db_prediction_id,
        "cxr_prediction": cxr_result["prediction"],
        "cxr_probability": round(cxr_result["probability"], 4),
        "cxr_prob_normal": round(cxr_result["prob_normal"], 4),
        "cxr_prob_pneumonia": round(cxr_result["prob_pneumonia"], 4),
        "multimodal_prediction": fusion_result["prediction"],
        "multimodal_probability": round(fusion_result["probability"], 4),
        "multimodal_uncertainty": round(1.0 - fusion_result["probability"], 4),
        "feature_saved_to": str(feature_path) if feature_path else None,
    }
