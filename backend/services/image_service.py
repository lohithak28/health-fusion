from pathlib import Path

from ml.imaging.inference import predict_chest_xray


def analyze_chest_xray(image_path: Path):
    result = predict_chest_xray(image_path)

    return {
        "prediction": result["prediction"],
        "class_index": result["class_index"],
        "probability": result["probability"],
        "prob_normal": result["prob_normal"],
        "prob_pneumonia": result["prob_pneumonia"],
        "features": result["features"],
    }