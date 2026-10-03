from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db_optional
from backend.database.models import Patient, Prediction, UploadedFile
from backend.schemas.prediction import PredictionRequest, PredictionResponse
from backend.services.fusion_service import predict_multimodal
from backend.services.storage_service import StorageService


router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Optional[Session] = Depends(get_db_optional)):
    try:
        result = predict_multimodal(
            image_features=request.image_features,
            age=request.age,
            gender=request.gender,
            ecg_waveform=request.ecg_waveform,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Optional database persistence when database is available
    if db is not None:
        try:
            patient = db.query(Patient).filter(Patient.patient_id == request.patient_id).first()
            if not patient:
                patient = Patient(
                    patient_id=request.patient_id,
                    age=float(request.age),
                    gender=str(request.gender).strip().upper(),
                )
                db.add(patient)
                db.flush()
            else:
                patient.age = float(request.age)
                patient.gender = str(request.gender).strip().upper()

            pred_record = Prediction(
                patient_id=patient.id,
                cxr_prediction=None,
                cxr_probability=None,
                multimodal_prediction=result["prediction"],
                multimodal_probability=round(result["probability"], 4),
                uncertainty=round(1.0 - result["probability"], 4),
            )
            db.add(pred_record)
            db.flush()

            # Persist ECG waveform file
            ecg_meta = StorageService.save_ecg_file(
                patient_id=request.patient_id,
                ecg_waveform=request.ecg_waveform,
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

    return PredictionResponse(
        patient_id=request.patient_id,
        prediction=result["prediction"],
        probability=result["probability"],
        uncertainty=1.0 - result["probability"],
    )