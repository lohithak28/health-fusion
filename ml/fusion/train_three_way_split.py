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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ml.fusion.cross_attention_fusion import CrossAttentionFusion


SEED = 42
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ThreeWayDataset(Dataset):
    def __init__(self, labels_df, image_features, waveforms):
        self.df = labels_df.reset_index(drop=True)
        self.image_features = image_features
        self.waveforms = waveforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Prototype-only image assignment.
        # The CXR and MIMIC samples are not patient matched.
        image_index = int(row["_image_index"])
        image = self.image_features[image_index]

        waveform_index = int(row["waveform_index"])
        ecg = self.waveforms[waveform_index]

        # EHR representation.
        age = float(row["anchor_age"]) / 100.0
        gender = 0.0 if row["gender"] == "M" else 1.0

        ehr = np.array(
            [age, gender],
            dtype=np.float32,
        )

        label = float(row["label"])

        return (
            image.float(),
            torch.tensor(ehr, dtype=torch.float32),
            torch.tensor(ecg, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


class ThreeWayModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ECG encoder
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

        # EHR encoder
        self.ehr_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
        )

        # Image feature encoder
        self.image_encoder = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
        )

        # Cross-attention multimodal fusion
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


def calculate_metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
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


def evaluate(model, loader, device):
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

    labels = np.array(
        all_labels,
        dtype=int,
    )

    probabilities = np.array(
        all_probabilities,
    )

    return calculate_metrics(
        labels,
        probabilities,
    )


def print_metrics(title, metrics):
    print()
    print(title)
    print("-" * len(title))

    for name, value in metrics.items():
        if np.isnan(value):
            print(f"{name}: N/A")
        else:
            print(f"{name}: {value:.4f}")


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    labels_df = pd.read_csv(
        "features/labels/admission_labels.csv"
    )

    image_features = torch.load(
        "features/image_features.pt",
        map_location="cpu",
    )

    waveforms = np.load(
        "features/ecg_waveforms/ecg_waveforms.npy"
    )

    # Deterministic prototype-only image assignment.
    # This does NOT represent real patient matching.
    labels_df["_image_index"] = (
        np.arange(len(labels_df))
        % len(image_features)
    )

    print("Total samples:", len(labels_df))
    print("Image features:", image_features.shape)
    print("ECG waveforms:", waveforms.shape)

    # 70% train / 15% validation / 15% test.
    # Stratification preserves the mortality class distribution.
    train_df, temp_df = train_test_split(
        labels_df,
        test_size=0.30,
        random_state=SEED,
        stratify=labels_df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    print()
    print("Split sizes:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    print()
    print("Split positives:")
    print("Train:", int(train_df["label"].sum()))
    print("Validation:", int(val_df["label"].sum()))
    print("Test:", int(test_df["label"].sum()))

    train_dataset = ThreeWayDataset(
        train_df,
        image_features,
        waveforms,
    )

    val_dataset = ThreeWayDataset(
        val_df,
        image_features,
        waveforms,
    )

    test_dataset = ThreeWayDataset(
        test_df,
        image_features,
        waveforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ThreeWayModel().to(device)

    positive_count = int(
        train_df["label"].sum()
    )

    negative_count = len(train_df) - positive_count

    pos_weight = torch.tensor(
        negative_count / max(positive_count, 1),
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print()
    print(
        "Training positive-class weight:",
        pos_weight.item(),
    )

    best_val_auroc = -float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0

        for image, ehr, ecg, labels in train_loader:
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

        average_loss = (
            total_loss / len(train_loader)
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        val_auroc = val_metrics["auroc"]

        if not np.isnan(val_auroc):
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in model.state_dict().items()
                }

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"| Loss: {average_loss:.4f} "
            f"| Val AUROC: {val_auroc:.4f}"
        )

    # Restore the model with the best validation AUROC.
    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    final_test_metrics = evaluate(
        model,
        test_loader,
        device,
    )

    print_metrics(
        "Final Validation Metrics",
        final_val_metrics,
    )

    print_metrics(
        "Final Test Metrics",
        final_test_metrics,
    )

    output_path = (
        "features/models/"
        "member3_three_way_cross_attention.pt"
    )

    torch.save(
        model.state_dict(),
        output_path,
    )

    print()
    print("Saved model to:", output_path)


if __name__ == "__main__":
    main()