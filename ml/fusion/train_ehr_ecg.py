import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from ml.ehr_ecg.fusion_model import EHR_ECGFusionModel


SEED = 42
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EHR_ECGDataset(Dataset):
    def __init__(self, labels_df, waveforms):
        self.df = labels_df.reset_index(drop=True)
        self.waveforms = waveforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        waveform_index = int(row["waveform_index"])
        ecg = self.waveforms[waveform_index]

        age = float(row["anchor_age"]) / 100.0

        gender = 0.0 if row["gender"] == "M" else 1.0

        ehr = np.array(
            [age, gender],
            dtype=np.float32,
        )

        label = float(row["label"])

        return (
            torch.tensor(ecg, dtype=torch.float32),
            torch.tensor(ehr, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


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

    labels_path = (
        "features/labels/admission_labels.csv"
    )

    waveform_path = (
        "features/ecg_waveforms/ecg_waveforms.npy"
    )

    labels_df = pd.read_csv(labels_path)
    waveforms = np.load(waveform_path)

    print("Labels:", labels_df.shape)
    print("Waveforms:", waveforms.shape)

    dataset = EHR_ECGDataset(
        labels_df,
        waveforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = EHR_ECGFusionModel().to(device)

    positive_count = int(
        labels_df["label"].sum()
    )

    negative_count = len(labels_df) - positive_count

    pos_weight = torch.tensor(
        negative_count / max(positive_count, 1),
        dtype=torch.float32,
        device=device,
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
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

        for ecg, ehr, labels in loader:

            ecg = ecg.to(device)
            ehr = ehr.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(
                ecg,
                ehr,
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
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"| Loss: {average_loss:.4f}"
        )

    # Evaluation on the same cohort for this initial pipeline test.
    # This is NOT a held-out test result.
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for ecg, ehr, labels in loader:

            ecg = ecg.to(device)
            ehr = ehr.to(device)

            logits = model(
                ecg,
                ehr,
            )

            probabilities = torch.sigmoid(
                logits
            )

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
        all_probabilities
    )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
    )

    print("\nInitial pipeline metrics")
    print("------------------------")

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    output_path = (
        "features/models/member3_ehr_ecg_baseline.pt"
    )

    torch.save(
        model.state_dict(),
        output_path,
    )

    print(
        "\nSaved model to:",
        output_path,
    )


if __name__ == "__main__":
    main()