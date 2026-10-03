from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    patient_id: str

    image_features: list[float] = Field(
        ...,
        min_length=2048,
        max_length=2048,
    )

    age: float = Field(..., ge=0, le=120)

    gender: str

    ecg_waveform: list[list[float]] = Field(
        ...,
        min_length=12,
        max_length=12,
    )


class PredictionResponse(BaseModel):
    patient_id: str
    prediction: str
    probability: float
    uncertainty: float