import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from ml.fusion.fusion_model import MultimodalFusion


SEED = 42
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ThreeWayDataset(Dataset):
    """
    Proof-of-concept three-modality dataset.

    IMPORTANT:
    Image features and MIMIC EHR/ECG samples are NOT patient matched.
    Image rows are assigned cyclically only to demonstrate the fusion
    architecture.
    """

    def __init__(self, labels_df, image_features, waveforms):
        self.df = labels_df.reset_index(drop=True)
        self.image_features = image_features
        self.waveforms = waveforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Synthetic image assignment for architecture testing.
        image_index = index % len(self.image_features)

        image = self.image_features[image_index]

        waveform_index = int(row["waveform_index"])
        ecg = self.waveforms[waveform_index]

        # Same EHR representation used by the existing EHR+ECG model.
        age = float(row["anchor_age"]) / 100.0
        gender = 0.0 if row["gender"] == "M" else 1.0

        ehr = np.array(
            [age, gender],
            dtype=np.float32,
        )

        label = float(row["label"])

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(ehr, dtype=torch.float32),
            torch.tensor(ecg, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


class ThreeWayModel(nn.Module):
    """
    Three-way HealthFusion prototype.

    Image: 2048-D
    EHR:   2-D
    ECG:   12 x 1000 waveform

    The ECG encoder produces a compact vector before the three
    modalities enter the fusion module.
    """

    def __init__(self):
        super().__init__()

        self.ecg_encoder = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
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

        self.fusion = MultimodalFusion(
            image_dim=128,
            ehr_dim=64,
            ecg_dim=64,
            fusion_dim=128,
            num_classes=1,
        )

    def forward(self, image, ehr, ecg):
        image_features = self.image_encoder(image)

        ehr_features = self.ehr_encoder(ehr)

        ecg_features = self.ecg_encoder(ecg)
        ecg_features = ecg_features.squeeze(-1)

        output = self.fusion(
            image_features,
            ehr_features,
            ecg_features,
        )

        return output.squeeze(-1)


def calculate_metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions,
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
    }

    if len(np.unique(labels)) == 2:
        metrics["auroc"] = roc_auc_score(
            labels,
            probabilities,
        )
    else:
        metrics["auroc"] = float("nan")

    return metrics


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    labels_path = "features/labels/admission_labels.csv"
    image_path = "features/image_features.pt"
    waveform_path = "features/ecg_waveforms/ecg_waveforms.npy"

    labels_df = pd.read_csv(labels_path)

    image_features = torch.load(
        image_path,
        map_location="cpu",
    )

    waveforms = np.load(waveform_path)

    print("Labels:", labels_df.shape)
    print("Image features:", image_features.shape)
    print("ECG waveforms:", waveforms.shape)

    dataset = ThreeWayDataset(
        labels_df,
        image_features,
        waveforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = ThreeWayModel().to(device)

    positive_count = int(labels_df["label"].sum())
    negative_count = len(labels_df) - positive_count

    pos_weight = torch.tensor(
        negative_count / max(positive_count, 1),
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("Positive examples:", positive_count)
    print("Negative examples:", negative_count)
    print("Positive-class weight:", pos_weight.item())

    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for image, ehr, ecg, labels in loader:
            image = image.to(device)
            ehr = ehr.to(device)
            ecg = ecg.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(
                image,
                ehr,
                ecg,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"| Loss: {average_loss:.4f}"
        )

    # Evaluation on the same cohort used for training.
    # These are NOT held-out test metrics.
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for image, ehr, ecg, labels in loader:
            image = image.to(device)
            ehr = ehr.to(device)
            ecg = ecg.to(device)

            logits = model(
                image,
                ehr,
                ecg,
            )

            probabilities = torch.sigmoid(logits)

            all_labels.extend(
                labels.numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    all_labels = np.array(
        all_labels,
        dtype=int,
    )

    all_probabilities = np.array(
        all_probabilities,
    )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
    )

    print()
    print("Three-way prototype metrics")
    print("----------------------------")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    output_path = (
        "features/models/"
        "member3_three_way_prototype.pt"
    )

    torch.save(
        model.state_dict(),
        output_path,
    )

    print()
    print("Saved model to:", output_path)


if __name__ == "__main__":
    main()