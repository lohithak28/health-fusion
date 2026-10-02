
from pathlib import Path

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
from torch.utils.data import DataLoader, TensorDataset

from labelled_dataset import EHR_ECGDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "features" / "labels" / "splits"
SEED = 42
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 16


class EHRBaseline(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
        )

    def forward(self, ehr):
        return self.network(ehr).squeeze(-1)


def load_data(split):
    ds = EHR_ECGDataset(SPLIT_DIR / f"{split}.csv")
    ehr = torch.tensor(ds.ehr_features, dtype=torch.float32)
    labels = torch.tensor(ds.labels, dtype=torch.float32)
    return TensorDataset(ehr, labels)


def predict(model, loader, device):
    model.eval()
    labels_all, probs_all = [], []

    with torch.no_grad():
        for ehr, labels in loader:
            logits = model(ehr.to(device))
            probs = torch.sigmoid(logits)
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())

    return np.asarray(labels_all), np.asarray(probs_all)


def report(name, labels, probs):
    pred = (probs >= 0.5).astype(int)
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(labels, pred), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(labels, pred), 4))
    print("Precision:", round(precision_score(labels, pred, zero_division=0), 4))
    print("Sensitivity/recall:", round(recall_score(labels, pred, zero_division=0), 4))
    print("F1:", round(f1_score(labels, pred, zero_division=0), 4))
    print("Confusion matrix:\n", confusion_matrix(labels, pred, labels=[0, 1]))
    if len(np.unique(labels)) == 2:
        print("AUROC:", round(roc_auc_score(labels, probs), 4))
    else:
        print("AUROC: undefined (only one class in this split)")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = load_data("train")
    val_ds = load_data("validation")
    test_ds = load_data("test")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    train_labels = train_ds.tensors[1].numpy()
    positives = int((train_labels == 1).sum())
    negatives = int((train_labels == 0).sum())

    model = EHRBaseline().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [negatives / positives], dtype=torch.float32, device=device
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_n = 0.0, 0

        for ehr, labels in train_loader:
            ehr, labels = ehr.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ehr), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_n += len(labels)

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for ehr, labels in val_loader:
                ehr, labels = ehr.to(device), labels.to(device)
                loss = criterion(model(ehr), labels)
                val_loss_sum += loss.item() * len(labels)
                val_n += len(labels)

        val_loss = val_loss_sum / val_n
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {total_loss / total_n:.4f} | "
            f"val loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No best model checkpoint was produced.")

    model.load_state_dict(best_state)

    # Validation and test evaluation after model selection.
    val_labels, val_probs = predict(model, val_loader, device)
    test_labels, test_probs = predict(model, test_loader, device)
    report("EHR-only validation results", val_labels, val_probs)
    report("EHR-only test results", test_labels, test_probs)

    output_dir = PROJECT_ROOT / "features" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        output_dir / "ehr_only_baseline.pt",
    )
    print("\nSaved EHR-only model checkpoint.")


if __name__ == "__main__":
    main()
