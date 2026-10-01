import argparse
import json
from pathlib import Path
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.imaging.config import ImagingConfig, config as default_config
from ml.imaging.dataset import ChestXRayDataset
from ml.imaging.model import HealthFusionResNet50
from ml.imaging.preprocessing import get_eval_transforms
from ml.imaging.utils import get_device, load_checkpoint


@torch.no_grad()
def evaluate_model(
    cfg: ImagingConfig = default_config,
    checkpoint_path: Path = None,
) -> Dict[str, Any]:
    """
    Evaluates the trained ResNet-50 on the official test split.
    Generates:
      - metrics.json
      - classification_report.txt
      - confusion_matrix.png
      - sample_predictions.csv
    """
    print("\n" + "=" * 70)
    print("HEALTHFUSION-TRANSFORMER: CHEST X-RAY EVALUATION (TEST SET)")
    print("=" * 70)

    device = torch.device(cfg.device)
    chk_path = Path(checkpoint_path) if checkpoint_path else cfg.best_model_path

    # 1. Dataset & DataLoader
    eval_transform = get_eval_transforms(
        image_size=cfg.image_size,
        mean=cfg.norm_mean,
        std=cfg.norm_std,
    )
    test_dataset = ChestXRayDataset(
        root_dir=cfg.test_dir,
        transform=eval_transform,
        class_to_idx=cfg.class_to_idx,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    print(f"Test Split Path: {cfg.test_dir}")
    print(f"Total Test Samples: {len(test_dataset)}")
    print(f"Class Counts: {test_dataset.get_class_counts()}")

    # 2. Load Model
    model = HealthFusionResNet50(
        num_classes=cfg.num_classes,
        pretrained=True,
        dropout_rate=cfg.dropout_rate,
    )

    if chk_path.exists():
        print(f"Loading trained model checkpoint: {chk_path}")
        load_checkpoint(model, chk_path, device)
    else:
        print(f"[Warning] Checkpoint '{chk_path}' not found! Evaluating ImageNet baseline initialization.")

    model.to(device)
    model.eval()

    # 3. Predict on Test Set
    all_targets = []
    all_preds = []
    all_probs = []
    sample_records = []

    global_idx = 0
    for images, targets in tqdm(test_loader, desc="Evaluating Test Set"):
        images = images.to(device)
        targets = targets.to(device)

        logits, features = model(images, return_features=True)
        probs = F.softmax(logits, dim=1)

        batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
        batch_targets = targets.cpu().numpy()
        batch_probs = probs.cpu().numpy()

        all_preds.extend(batch_preds)
        all_targets.extend(batch_targets)
        all_probs.extend(batch_probs)

        # Record samples for inspection
        for i in range(len(targets)):
            sample_path = test_dataset.get_sample_path(global_idx)
            pred_class = cfg.idx_to_class[batch_preds[i]]
            true_class = cfg.idx_to_class[batch_targets[i]]
            prob_pneumonia = float(batch_probs[i][1])
            confidence = float(batch_probs[i][batch_preds[i]])

            sample_records.append({
                "sample_idx": global_idx,
                "filename": sample_path.name,
                "filepath": str(sample_path),
                "true_label": int(batch_targets[i]),
                "true_class": true_class,
                "predicted_label": int(batch_preds[i]),
                "predicted_class": pred_class,
                "prob_pneumonia": prob_pneumonia,
                "confidence": confidence,
                "is_correct": bool(batch_preds[i] == batch_targets[i]),
            })
            global_idx += 1

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 4. Compute Metrics
    acc = float(accuracy_score(all_targets, all_preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    try:
        auc = float(roc_auc_score(all_targets, all_probs[:, 1]))
    except Exception:
        auc = None

    class_report_str = classification_report(
        all_targets,
        all_preds,
        target_names=cfg.class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(all_targets, all_preds)

    # 5. Save Outputs
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    # Classification report text
    report_path = cfg.outputs_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("HEALTHFUSION-TRANSFORMER: CHEST X-RAY EVALUATION REPORT\n")
        f.write("Task: Normal (0) vs Pneumonia (1) Classification\n")
        f.write("Model Backbone: ResNet-50 (2048-D F_img representation)\n")
        f.write("=" * 60 + "\n\n")
        f.write(class_report_str)
        f.write(f"\nOverall Test Accuracy: {acc * 100:.2f}%\n")
        if auc is not None:
            f.write(f"ROC AUC Score: {auc:.4f}\n")
    print(f"[Evaluation] Classification report saved to: {report_path}")

    # Metrics JSON
    metrics_path = cfg.outputs_dir / "metrics.json"
    metrics_data = {
        "dataset_split": "test",
        "total_test_samples": len(test_dataset),
        "accuracy": acc,
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classes": {
            "NORMAL": {
                "precision": float(prec_per_class[0]),
                "recall": float(rec_per_class[0]),
                "f1": float(f1_per_class[0]),
                "support": int(support_per_class[0]),
            },
            "PNEUMONIA": {
                "precision": float(prec_per_class[1]),
                "recall": float(rec_per_class[1]),
                "f1": float(f1_per_class[1]),
                "support": int(support_per_class[1]),
            },
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[Evaluation] Metrics JSON saved to: {metrics_path}")

    # Confusion Matrix Plot
    cm_path = cfg.outputs_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg.class_names)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)
    ax.set_title("HealthFusion ResNet-50: Test Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Confusion matrix plot saved to: {cm_path}")

    # Sample Predictions CSV
    sample_df = pd.DataFrame(sample_records)
    csv_path = cfg.outputs_dir / "sample_predictions.csv"
    sample_df.to_csv(csv_path, index=False)
    print(f"[Evaluation] Sample predictions CSV saved to: {csv_path}")

    print("\n" + "=" * 60)
    print("TEST EVALUATION SUMMARY:")
    print("=" * 60)
    print(class_report_str)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Macro F1 : {f1_macro:.4f}")
    if auc is not None:
        print(f"ROC AUC  : {auc:.4f}")
    print("=" * 60 + "\n")

    return metrics_data


def main():
    parser = argparse.ArgumentParser(description="HealthFusion-Transformer: Evaluate Imaging Branch")
    parser.add_argument("--data-dir", type=str, default=None, help="Root path of chest_xray dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_resnet50.pth")
    args = parser.parse_args()

    cfg = ImagingConfig()
    if args.data_dir:
        cfg.dataset_root = Path(args.data_dir)

    evaluate_model(cfg, checkpoint_path=Path(args.checkpoint) if args.checkpoint else None)


if __name__ == "__main__":
    main()
