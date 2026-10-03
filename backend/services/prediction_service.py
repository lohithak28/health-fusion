from pathlib import Path

import torch
import torch.nn as nn

from ml.fusion.cross_attention_fusion import CrossAttentionFusion


class ThreeWayModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.ecg_encoder = nn.Sequential(
            nn.Conv1d(
                12,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.ReLU(),
            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.ehr_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
        )

        self.image_encoder = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
        )

        self.fusion = CrossAttentionFusion(
            image_dim=128,
            ehr_dim=64,
            ecg_dim=64,
            fusion_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.2,
        )

    def forward(self, image, ehr, ecg):
        image_features = self.image_encoder(image)

        ehr_features = self.ehr_encoder(ehr)

        ecg_features = self.ecg_encoder(ecg)
        ecg_features = ecg_features.squeeze(-1)

        return self.fusion(
            image_features,
            ehr_features,
            ecg_features,
        )


MODEL_PATH = Path(
    "features/models/member3_three_way_cross_attention.pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ThreeWayModel().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()


def predict_three_way(
    image_input,
    age: float,
    gender: str,
    ecg_input,
):
    """
    Runs inference through the trained ThreeWayModel using multimodal inputs:
    - image_input: Path to saved F_img .pt file, torch.Tensor [1, 2048] / [2048], or list of 2048 floats.
    - age: float (normalized by 100.0).
    - gender: str ('M' -> 0.0, else 1.0).
    - ecg_input: torch.Tensor, numpy array, or list of lists [12, 1000] or [1, 12, 1000].
    """
    # 1. Image features [1, 2048]
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image feature file not found: {path}")
        image_tensor = torch.load(path, map_location=device)
    elif isinstance(image_input, torch.Tensor):
        image_tensor = image_input.to(device)
    else:
        image_tensor = torch.tensor(image_input, dtype=torch.float32, device=device)

    if image_tensor.dim() == 1:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(dtype=torch.float32, device=device)

    if image_tensor.shape[-1] != 2048:
        raise ValueError(
            f"Expected image feature dimension 2048, got shape {list(image_tensor.shape)}"
        )

    # 2. EHR features [1, 2]
    norm_age = float(age) / 100.0
    gender_val = 0.0 if str(gender).strip().upper().startswith("M") else 1.0
    ehr_tensor = torch.tensor([[norm_age, gender_val]], dtype=torch.float32, device=device)

    # 3. ECG waveform [1, 12, 1000]
    if isinstance(ecg_input, torch.Tensor):
        ecg_tensor = ecg_input.to(device)
    elif hasattr(ecg_input, "__array__"):
        import numpy as np
        ecg_tensor = torch.from_numpy(np.asarray(ecg_input)).to(device)
    else:
        ecg_tensor = torch.tensor(ecg_input, dtype=torch.float32, device=device)

    if ecg_tensor.dim() == 2:
        ecg_tensor = ecg_tensor.unsqueeze(0)
    ecg_tensor = ecg_tensor.to(dtype=torch.float32, device=device)

    if ecg_tensor.shape[1] != 12:
        raise ValueError(f"Expected 12 ECG leads, got shape {list(ecg_tensor.shape)}")

    # 4. Model forward pass
    with torch.no_grad():
        logits = model(image_tensor, ehr_tensor, ecg_tensor)
        prob = torch.sigmoid(logits).item()

    pred = "Positive" if prob >= 0.5 else "Negative"

    return {
        "prediction": pred,
        "probability": prob,
        "uncertainty": 1.0 - prob,
    }