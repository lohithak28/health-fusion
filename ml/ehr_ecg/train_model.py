
from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from labelled_dataset import create_dataloaders
from fusion_model import EHR_ECGFusionModel


SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "features" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "ehr_ecg_fusion.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in loader:
            ecg = batch["ecg"].to(device)
            ehr = batch["ehr"].to(device)
            labels = batch["label"].to(device)

            logits = model(ecg, ehr)
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return np.asarray(all_labels), np.asarray(all_probs)


def calculate_metrics(labels, probs, threshold=0.5):
    predictions = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ),
    }

    if len(np.unique(labels)) == 2:
        metrics["auroc"] = roc_auc_score(labels, probs)
    else:
        metrics["auroc"] = float("nan")

    return metrics


def print_metrics(name, labels, probs):
    metrics = calculate_metrics(labels, probs)

    print(f"\n{name} metrics (threshold = 0.50)")
    for key, value in metrics.items():
        print(f"{key}:")
        print(value)

    print("Label counts:", np.bincount(labels.astype(int), minlength=2))


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    # Calculate positive weight from training labels only.
    train_labels = train_loader.dataset.labels
    positives = int((train_labels == 1).sum())
    negatives = int((train_labels == 0).sum())

    if positives == 0:
        raise ValueError("Training split has no positive examples.")

    pos_weight = torch.tensor(
        [negatives / positives], dtype=torch.float32, device=device
    )

    print(f"Training examples: {len(train_labels)}")
    print(f"Training deaths: {positives}")
    print(f"Training non-deaths: {negatives}")
    print(f"Positive class weight: {pos_weight.item():.3f}")

    model = EHR_ECGFusionModel().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            ecg = batch["ecg"].to(device)
            ehr = batch["ehr"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(ecg, ehr)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(labels)
            seen += len(labels)

        train_loss = running_loss / seen

        # Validation loss uses the same training-derived class weight.
        model.eval()
        val_loss_sum = 0.0
        val_seen = 0

        with torch.no_grad():
            for batch in val_loader:
                ecg = batch["ecg"].to(device)
                ehr = batch["ehr"].to(device)
                labels = batch["label"].to(device)

                logits = model(ecg, ehr)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * len(labels)
                val_seen += len(labels)

        val_loss = val_loss_sum / val_seen

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "validation_loss": best_val_loss,
            "seed": SEED,
        },
        MODEL_PATH,
    )
    print(f"\nBest epoch: {best_epoch}")
    print("Saved model:", MODEL_PATH)

    # Choose the checkpoint using validation loss only.
    val_labels, val_probs = collect_predictions(model, val_loader, device)
    print_metrics("Validation", val_labels, val_probs)

    # Evaluate test data only after model selection is complete.
    test_labels, test_probs = collect_predictions(model, test_loader, device)
    print_metrics("Test", test_labels, test_probs)

    # Majority-class baseline: always predict non-death.
    baseline_probs = np.zeros_like(test_labels, dtype=float)
    print_metrics("Test majority-class baseline", test_labels, baseline_probs)


if __name__ == "__main__":
    main()
