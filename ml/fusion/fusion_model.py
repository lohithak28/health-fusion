import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """
    Baseline multimodal fusion model.

    Takes:
        image features
        EHR features
        ECG features

    Projects each modality into the same embedding space,
    concatenates them, and predicts the target.
    """

    def __init__(
        self,
        image_dim,
        ehr_dim,
        ecg_dim,
        fusion_dim=256,
        num_classes=2,
    ):
        super().__init__()

        # Convert each modality to the same dimension
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.ehr_projection = nn.Sequential(
            nn.Linear(ehr_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.ecg_projection = nn.Sequential(
            nn.Linear(ecg_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, image_features, ehr_features, ecg_features):

        # Project each modality
        image_embedding = self.image_projection(image_features)
        ehr_embedding = self.ehr_projection(ehr_features)
        ecg_embedding = self.ecg_projection(ecg_features)

        # Concatenate modalities
        fused = torch.cat(
            [
                image_embedding,
                ehr_embedding,
                ecg_embedding,
            ],
            dim=1,
        )

        # Prediction
        output = self.classifier(fused)

        return output