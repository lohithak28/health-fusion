from fastapi import APIRouter

from backend.schemas.ehr import EHRRequest, EHRResponse


router = APIRouter(prefix="/ehr", tags=["EHR"])


@router.post(
    "",
    response_model=EHRResponse,
    include_in_schema=False,
)
@router.post(
    "/",
    response_model=EHRResponse,
    summary="Submit and validate clinical EHR data",
    description="Validates patient age and gender, formatting them into the representation expected by the HealthFusion EHR branch.",
)
def record_ehr(request: EHRRequest):
    norm_age = round(request.age / 100.0, 4)
    gender_encoded = 0.0 if request.gender == "M" else 1.0

    return EHRResponse(
        patient_id=request.patient_id,
        age=request.age,
        gender=request.gender,
        normalized_age=norm_age,
        encoded_gender=gender_encoded,
        message="EHR clinical data recorded and validated successfully",
    )
