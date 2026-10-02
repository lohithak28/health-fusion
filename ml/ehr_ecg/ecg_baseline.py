
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "features" / "labels" / "splits"
WAVEFORM_PATH = (
    PROJECT_ROOT / "features" / "ecg_waveforms" / "ecg_waveforms.npy"
)
MODEL_DIR = PROJECT_ROOT / "features" / "models"

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3


# --------------------------------------------------
# ECG-only dataset
# --------------------------------------------------

class ECGOnlyDataset(Dataset):
    """Load ECG waveforms and labels from a split CSV."""

    def __init__(self, split):
        csv_path = SPLIT_DIR / f"{split}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {csv_path}"
            )

        if not WAVEFORM_PATH.exists():
            raise FileNotFoundError(
                f"ECG waveform file not found: {WAVEFORM_PATH}"
            )

        self.df = pd.read_csv(csv_path)

        required_columns = {"waveform_index", "label"}
        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {sorted(missing)}"
            )

        self.waveform_indices = self.df[
            "waveform_index"
        ].to_numpy(dtype=np.int64)

        self.labels = self.df["label"].to_numpy(dtype=np.float32)

        # Memory-map the waveform array to avoid loading a second
        # full copy into RAM.
        self.waveforms = np.load(
            WAVEFORM_PATH,
            mmap_mode="r",
        )

        if len(self.waveform_indices) > 0:
            if (
                self.waveform_indices.min() < 0
                or self.waveform_indices.max() >= len(self.waveforms)
            ):
                raise IndexError(
                    f"Invalid waveform_index in {csv_path}"
                )

        if not np.isfinite(self.labels).all():
            raise ValueError(f"Non-finite labels found in {csv_path}")

        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError(f"Labels must be 0 or 1 in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        waveform_index = self.waveform_indices[index]

        ecg = np.array(
            self.waveforms[waveform_index],
            dtype=np.float32,
            copy=True,
        )

        if not np.isfinite(ecg).all():
            raise ValueError(
                f"Non-finite ECG values at waveform index {waveform_index}"
            )

        return (
            torch.from_numpy(ecg),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


def make_loader(split, shuffle=False):
    dataset = ECGOnlyDataset(split)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


# --------------------------------------------------
# ECG-only model: 1D CNN
# --------------------------------------------------

class ECGOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, ecg):
        features = self.encoder(ecg)
        return self.classifier(features).squeeze(-1)


# --------------------------------------------------
# Training and evaluation
# --------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.set_grad_enabled(training):
        for ecg, labels in loader:
            ecg = ecg.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits = model(ecg)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            all_labels.extend(
                labels.detach().cpu().numpy().astype(int)
            )
            all_probs.extend(
                torch.sigmoid(logits).detach().cpu().numpy()
            )

    average_loss = total_loss / len(loader.dataset)

    return (
        average_loss,
        np.asarray(all_labels, dtype=int),
        np.asarray(all_probs, dtype=float),
    )


def report_metrics(name, labels, probs):
    predictions = (probs >= 0.5).astype(int)

    print(f"\n{name} results")
    print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(labels, predictions):.4f}"
    )
    print(
        "Precision: "
        f"{precision_score(labels, predictions, zero_division=0):.4f}"
    )
    print(
        "Recall: "
        f"{recall_score(labels, predictions, zero_division=0):.4f}"
    )
    print(
        "F1: "
        f"{f1_score(labels, predictions, zero_division=0):.4f}"
    )
    print(
        "Confusion matrix (rows=true, columns=predicted):\n",
        confusion_matrix(labels, predictions, labels=[0, 1]),
    )

    if len(np.unique(labels)) == 2:
        print(f"AUROC: {roc_auc_score(labels, probs):.4f}")
    else:
        print("AUROC: undefined (only one class in this split)")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("validation")
    test_loader = make_loader("test")

    # Calculate positive class weight from training data only.
    train_labels = train_loader.dataset.labels.astype(int)

    negatives = int((train_labels == 0).sum())
    positives = int((train_labels == 1).sum())

    if positives == 0 or negatives == 0:
        raise ValueError(
            "Training split must contain both classes."
        )

    pos_weight = torch.tensor(
        [negatives / positives],
        dtype=torch.float32,
        device=DEVICE,
    )

    print(
        f"Training label counts: "
        f"[negative={negatives}, positive={positives}]"
    )
    print(f"Positive class weight: {pos_weight.item():.4f}")
    print(f"Train/validation/test sizes: "
          f"{len(train_loader.dataset)}/"
          f"{len(val_loader.dataset)}/"
          f"{len(test_loader.dataset)}")

    model = ECGOnlyModel().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid checkpoint."
        )

    model.load_state_dict(best_state)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Evaluate the selected checkpoint.
    _, val_labels, val_probs = run_epoch(
        model,
        val_loader,
        criterion,
    )

    _, test_labels, test_probs = run_epoch(
        model,
        test_loader,
        criterion,
    )

    report_metrics(
        "ECG-only validation",
        val_labels,
        val_probs,
    )

    report_metrics(
        "ECG-only test",
        test_labels,
        test_probs,
    )

    # Save the selected model checkpoint.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "ecg_only_baseline.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_validation_loss": best_val_loss,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    print(f"\nSaved ECG-only checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
