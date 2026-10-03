from fastapi import APIRouter

from backend.schemas.sensor import SensorRequest, SensorResponse


router = APIRouter(prefix="/sensor", tags=["Sensor"])


@router.post(
    "",
    response_model=SensorResponse,
    include_in_schema=False,
)
@router.post(
    "/",
    response_model=SensorResponse,
    summary="Submit and validate 12-lead ECG sensor data",
    description=(
        "Validates 12-lead ECG time series data with exactly 1000 samples per lead "
        "(100 Hz, 10s) as expected by the HealthFusion multimodal sensor branch [12, 1000]."
    ),
)
def record_sensor_data(request: SensorRequest):
    number_of_leads = len(request.ecg_waveform)
    samples_per_lead = len(request.ecg_waveform[0])

    return SensorResponse(
        patient_id=request.patient_id,
        number_of_leads=number_of_leads,
        samples_per_lead=samples_per_lead,
        message="12-lead ECG sensor waveform recorded and validated successfully",
    )

