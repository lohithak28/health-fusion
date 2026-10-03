from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class UploadedFileResponse(BaseModel):
    id: int
    file_type: str
    original_filename: str
    stored_path: str
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    uploaded_at: datetime
    prediction_id: Optional[int] = None

    class Config:
        from_attributes = True


class PredictionHistoryItem(BaseModel):
    id: int
    patient_id: str
    cxr_prediction: Optional[str] = None
    cxr_probability: Optional[float] = None
    multimodal_prediction: str
    multimodal_probability: float
    uncertainty: float
    created_at: datetime
    files: List[UploadedFileResponse] = []

    class Config:
        from_attributes = True


class PatientDetailResponse(BaseModel):
    id: int
    patient_id: str
    age: float
    gender: str
    created_at: datetime
    total_predictions: int = 0
    total_files: int = 0

    class Config:
        from_attributes = True
