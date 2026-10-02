import torch

from fusion_model import MultimodalFusion


# Fake feature dimensions for testing
IMAGE_DIM = 2048
EHR_DIM = 64
ECG_DIM = 128


# Create the fusion model
model = MultimodalFusion(
    image_dim=IMAGE_DIM,
    ehr_dim=EHR_DIM,
    ecg_dim=ECG_DIM,
)


# Fake batch of 10 patients
image_features = torch.randn(10, IMAGE_DIM)
ehr_features = torch.randn(10, EHR_DIM)
ecg_features = torch.randn(10, ECG_DIM)


# Run the model
output = model(
    image_features,
    ehr_features,
    ecg_features,
)


print("Image features:", image_features.shape)
print("EHR features:", ehr_features.shape)
print("ECG features:", ecg_features.shape)
print("Model output:", output.shape)