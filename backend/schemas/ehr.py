from typing import Literal
from pydantic import BaseModel, Field, field_validator


class EHRRequest(BaseModel):
    patient_id: str = Field(
        ...,
        min_length=1,
        description="Unique patient identifier",
        examples=["PATIENT_001"],
    )
    age: float = Field(
        ...,
        ge=0,
        le=120,
        description="Patient age in years (0 to 120)",
        examples=[54.0],
    )
    gender: Literal["M", "F", "m", "f"] = Field(
        ...,
        description="Patient gender representation ('M' or 'F')",
        examples=["M"],
    )

    @field_validator("gender")
    @classmethod
    def validate_and_normalize_gender(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in ("M", "F"):
            raise ValueError("Gender must be 'M' or 'F'")
        return upper


class EHRResponse(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    age: float = Field(..., description="Recorded age in years")
    gender: str = Field(..., description="Normalized gender ('M' or 'F')")
    normalized_age: float = Field(
        ...,
        description="Age normalized (age / 100.0) for the fusion model EHR branch [B, 2]",
    )
    encoded_gender: float = Field(
        ...,
        description="Gender numerically encoded (0.0 for M, 1.0 for F) for the fusion model",
    )
    message: str = Field(..., description="Status message confirming ingestion")
