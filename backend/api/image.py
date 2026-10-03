from pathlib import Path

from fastapi import APIRouter, File, UploadFile
import torch

from backend.services.image_service import analyze_chest_xray


router = APIRouter(prefix="/upload", tags=["Upload"])

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    contents = await file.read()
    file_path.write_bytes(contents)

    # Run Member 1 CXR inference & extract F_img [1, 2048]
    analysis = analyze_chest_xray(file_path)

    # Persist the 2048-D feature vector for downstream fusion
    feature_path = file_path.with_suffix(".pt")
    torch.save(analysis["features"], feature_path)

    return {
        "filename": file.filename,
        "saved_to": str(file_path),
        "prediction": analysis["prediction"],
        "class_index": analysis["class_index"],
        "probability": round(analysis["probability"], 4),
        "prob_normal": round(analysis["prob_normal"], 4),
        "prob_pneumonia": round(analysis["prob_pneumonia"], 4),
        "feature_saved_to": str(feature_path),
        "feature_shape": list(analysis["features"].shape),
        "message": "Image uploaded and analyzed successfully",
    }