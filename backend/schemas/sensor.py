from typing import List
from pydantic import BaseModel, Field, field_validator


class SensorRequest(BaseModel):
    patient_id: str = Field(
        ...,
        min_length=1,
        description="Unique patient identifier",
        examples=["PATIENT_001"],
    )
    ecg_waveform: List[List[float]] = Field(
        ...,
        min_length=12,
        max_length=12,
        description="12-lead ECG waveform array formatted as [12, 1000] (12 leads, 1000 samples per lead at 100 Hz for 10 seconds)",
    )

    @field_validator("ecg_waveform")
    @classmethod
    def validate_ecg_dimensions(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 12:
            raise ValueError(f"ECG waveform must contain exactly 12 leads, found {len(v)}")
        for idx, lead in enumerate(v):
            if len(lead) != 1000:
                raise ValueError(
                    f"Lead {idx} must contain exactly 1000 samples, found {len(lead)}"
                )
        return v


class SensorResponse(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    number_of_leads: int = Field(..., description="Number of ECG leads validated (expected: 12)")
    samples_per_lead: int = Field(
        ..., description="Number of temporal samples per lead validated (expected: 1000)"
    )
    message: str = Field(..., description="Validation confirmation message")
