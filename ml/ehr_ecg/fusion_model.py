
import torch
import torch.nn as nn


class ECGEncoder(nn.Module):
    """Convert a 12-lead ECG into a sequence of learned tokens."""

    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ecg):
        # Input: (batch, 12, 1000)
        x = self.convolution(ecg)  # (batch, d_model, ~250)
        x = x.transpose(1, 2)      # (batch, ~250, d_model)
        x = self.transformer(x)
        return self.norm(x)


class EHREncoder(nn.Module):
    """Represent age and gender as two EHR tokens."""

    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()
        self.age_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.gender_embedding = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ehr):
        # Input: (batch, 2), age/100 and gender encoded as 0 or 1.
        age = self.age_encoder(ehr[:, 0:1]).unsqueeze(1)
        gender_ids = ehr[:, 1].long().clamp(0, 1)
        gender = self.gender_embedding(gender_ids).unsqueeze(1)

        tokens = torch.cat([age, gender], dim=1)
        return self.dropout(self.norm(tokens))


class CrossAttentionFusion(nn.Module):
    """ECG queries attend to EHR keys and values."""

    def __init__(self, d_model=64, nhead=4, dropout=0.1):
        super().__init__()

        self.ecg_norm = nn.LayerNorm(d_model)
        self.ehr_norm = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ecg_tokens, ehr_tokens):
        query = self.ecg_norm(ecg_tokens)
        key_value = self.ehr_norm(ehr_tokens)

        attended, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )

        x = ecg_tokens + self.dropout(attended)
        x = x + self.dropout(self.feed_forward(x))
        return self.output_norm(x)


class EHR_ECGFusionModel(nn.Module):
    """Binary in-hospital mortality prediction from ECG and EHR."""

    def __init__(self, d_model=64, nhead=4, dropout=0.1):
        super().__init__()

        self.ecg_encoder = ECGEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=2,
            dropout=dropout,
        )
        self.ehr_encoder = EHREncoder(
            d_model=d_model,
            dropout=dropout,
        )
        self.fusion = CrossAttentionFusion(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, ecg, ehr):
        ecg_tokens = self.ecg_encoder(ecg)
        ehr_tokens = self.ehr_encoder(ehr)

        fused_tokens = self.fusion(ecg_tokens, ehr_tokens)
        pooled = fused_tokens.mean(dim=1)

        # Raw logits; use with BCEWithLogitsLoss.
        return self.classifier(pooled).squeeze(-1)


if __name__ == "__main__":
    model = EHR_ECGFusionModel()
    ecg = torch.randn(4, 12, 1000)
    ehr = torch.randn(4, 2)
    ehr[:, 1] = torch.randint(0, 2, (4,)).float()

    logits = model(ecg, ehr)
    print("Output shape:", tuple(logits.shape))
    print("Output finite:", torch.isfinite(logits).all().item())
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))
