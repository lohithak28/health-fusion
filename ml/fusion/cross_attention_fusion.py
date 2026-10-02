import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion for image, EHR, and ECG modality tokens.

    Each modality is projected into the same embedding dimension.
    The modality tokens then interact through Transformer-style
    self-attention before classification.
    """

    def __init__(
        self,
        image_dim=128,
        ehr_dim=64,
        ecg_dim=64,
        fusion_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.image_projection = nn.Linear(
            image_dim,
            fusion_dim,
        )

        self.ehr_projection = nn.Linear(
            ehr_dim,
            fusion_dim,
        )

        self.ecg_projection = nn.Linear(
            ecg_dim,
            fusion_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(
            fusion_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        image_features,
        ehr_features,
        ecg_features,
    ):
        image_token = self.image_projection(
            image_features
        ).unsqueeze(1)

        ehr_token = self.ehr_projection(
            ehr_features
        ).unsqueeze(1)

        ecg_token = self.ecg_projection(
            ecg_features
        ).unsqueeze(1)

        tokens = torch.cat(
            [
                image_token,
                ehr_token,
                ecg_token,
            ],
            dim=1,
        )

        fused_tokens = self.transformer(
            tokens
        )

        fused = fused_tokens.mean(
            dim=1
        )

        fused = self.norm(fused)

        return self.classifier(
            fused
        ).squeeze(-1)