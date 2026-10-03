from typing import Any, Dict, Union
import torch

from backend.services.prediction_service import device, model


def predict_multimodal(
    image_features: torch.Tensor,
    age: Union[int, float],
    gender: str,
    ecg_waveform: torch.Tensor,
) -> Dict[str, Any]:
    """
    Multimodal prediction service combining CXR image features, EHR variables, and ECG sensor data.

    Reuses the existing ThreeWayModel loaded in prediction_service.

    Inputs:
        image_features: torch.Tensor with shape [1, 2048] (or [2048])
        age: numeric age (0 to 120)
        gender: 'M' / 'F' (encoded as 0.0 for M, 1.0 for F)
        ecg_waveform: torch.Tensor with shape [1, 12, 1000] (or [12, 1000])

    Returns:
        Dict with 'prediction', 'probability', and 'raw_logit'
    """
    # 1. Validate and prepare image features [1, 2048]
    if not isinstance(image_features, torch.Tensor):
        image_features = torch.tensor(image_features, dtype=torch.float32)

    if image_features.dim() == 1:
        image_features = image_features.unsqueeze(0)

    image_tensor = image_features.to(dtype=torch.float32, device=device)

    if image_tensor.shape != torch.Size([1, 2048]):
        raise ValueError(
            f"Expected image_features shape [1, 2048], got {list(image_tensor.shape)}"
        )

    # 2. Validate and normalize EHR variables [1, 2]
    norm_age = float(age) / 100.0
    gender_str = str(gender).strip().upper()
    gender_encoded = 0.0 if gender_str == "M" else 1.0

    ehr_tensor = torch.tensor(
        [[norm_age, gender_encoded]],
        dtype=torch.float32,
        device=device,
    )

    # 3. Validate and prepare ECG waveform [1, 12, 1000]
    if not isinstance(ecg_waveform, torch.Tensor):
        ecg_waveform = torch.tensor(ecg_waveform, dtype=torch.float32)

    if ecg_waveform.dim() == 2:
        ecg_waveform = ecg_waveform.unsqueeze(0)

    ecg_tensor = ecg_waveform.to(dtype=torch.float32, device=device)

    if ecg_tensor.shape != torch.Size([1, 12, 1000]):
        raise ValueError(
            f"Expected ecg_waveform shape [1, 12, 1000], got {list(ecg_tensor.shape)}"
        )

    # 4. Model inference using existing ThreeWayModel
    with torch.no_grad():
        logits = model(image_tensor, ehr_tensor, ecg_tensor)
        logit_val = logits.item() if logits.numel() == 1 else logits.squeeze().item()
        probability = torch.sigmoid(torch.tensor(logit_val)).item()

    prediction = "Positive" if probability >= 0.5 else "Negative"

    return {
        "prediction": prediction,
        "probability": probability,
        "raw_logit": logit_val,
    }
