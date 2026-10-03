from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.prediction import router as prediction_router
from backend.api.image import router as image_router
from backend.api.ehr import router as ehr_router
from backend.api.sensor import router as sensor_router
from backend.api.patient import router as patient_router


app = FastAPI(
    title="HealthFusion API",
    description="Backend API for the HealthFusion-Transformer system",
    version="1.0.0",
)

# CORS configuration for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "HealthFusion API",
    }


app.include_router(prediction_router)
app.include_router(image_router)
app.include_router(ehr_router)
app.include_router(sensor_router)
app.include_router(patient_router)