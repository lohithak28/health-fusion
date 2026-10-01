import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from ml.imaging.config import ImagingConfig, config as default_config
from ml.imaging.dataset import get_dataloaders
from ml.imaging.model import HealthFusionResNet50
from ml.imaging.utils import set_seed, get_device, save_checkpoint, plot_training_history


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_desc: str = "Training",
    max_batches: int = None,
) -> Tuple[float, float, float, float, float]:
    """
    Train model for one epoch.
    Returns: (epoch_loss, accuracy, precision, recall, f1)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    pbar = tqdm(dataloader, desc=epoch_desc, leave=False)
    for batch_idx, (images, targets) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(images, return_features=True)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(total_samples, 1)
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    return epoch_loss, acc, precision, recall, f1


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
) -> Tuple[float, float, float, float, float]:
    """
    Evaluate model on validation split.
    Returns: (epoch_loss, accuracy, precision, recall, f1)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(dataloader, desc=desc, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        logits, _ = model(images, return_features=True)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    return epoch_loss, acc, precision, recall, f1


def train_pipeline(
    cfg: ImagingConfig = default_config,
    max_train_batches: int = None,
) -> Dict[str, Any]:
    """
    Executes full staged training and validation pipeline:
      - Stage 1: Freeze backbone, train classification head
      - Stage 2: Unfreeze higher layers (layer4) and fine-tune
      - Model checkpointing on best validation F1 / loss
      - Training curve generation
    """
    print("\n" + "=" * 70)
    print("HEALTHFUSION-TRANSFORMER: CHEST X-RAY IMAGING BRANCH TRAINING")
    print("=" * 70)

    # 1. Reproducibility & Device
    set_seed(cfg.random_seed)
    device = torch.device(cfg.device)
    print(f"Random Seed           : {cfg.random_seed}")
    print(f"Compute Device        : {device}")
    print(f"Dataset Root          : {cfg.dataset_root}")
    print(f"Batch Size            : {cfg.batch_size}")
    print(f"Stage 1 Epochs (Head) : {cfg.stage1_epochs} (LR={cfg.stage1_lr})")
    print(f"Stage 2 Epochs (Tune) : {cfg.stage2_epochs} (Backbone LR={cfg.stage2_backbone_lr}, Head LR={cfg.stage2_classifier_lr})")
    if max_train_batches:
        print(f"Batches per Epoch Lim : {max_train_batches}")

    # 2. DataLoaders & Class Weights
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(cfg)

    # 3. Model Initialization
    model = HealthFusionResNet50(
        num_classes=cfg.num_classes,
        pretrained=True,
        dropout_rate=cfg.dropout_rate,
    ).to(device)

    # 4. Criterion with Class Weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Tracking History
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    start_time = time.time()

    # =========================================================================
    # STAGE 1: Train Classification Head (Backbone Frozen)
    # =========================================================================
    print("\n--- STAGE 1: Training Classification Head (Backbone Frozen) ---")
    model.freeze_backbone()
    optimizer_stage1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.stage1_lr,
        weight_decay=cfg.weight_decay,
    )

    total_epochs = cfg.stage1_epochs + cfg.stage2_epochs
    current_epoch = 0

    for ep in range(1, cfg.stage1_epochs + 1):
        current_epoch += 1
        print(f"\n[Stage 1] Epoch {ep}/{cfg.stage1_epochs} (Total: {current_epoch}/{total_epochs})")
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer_stage1,
            device,
            epoch_desc=f"Stage 1 Epoch {ep}",
            max_batches=max_train_batches,
        )
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_epoch(
            model, val_loader, criterion, device, desc="Validation"
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"  Train -> Loss: {tr_loss:.4f} | Acc: {tr_acc * 100:.2f}% | Prec: {tr_prec:.4f} | Rec: {tr_rec:.4f} | F1: {tr_f1:.4f}\n"
            f"  Val   -> Loss: {val_loss:.4f} | Acc: {val_acc * 100:.2f}% | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}"
        )

        # Checkpoint if best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = current_epoch
            save_checkpoint(
                model,
                cfg.best_model_path,
                epoch=current_epoch,
                val_metrics={"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1},
                optimizer=optimizer_stage1,
            )
            # Save feature extractor weights
            torch.save(model.state_dict(), cfg.feature_extractor_path)

    # =========================================================================
    # STAGE 2: Deep Fine-Tuning (Unfreeze Layer4 + Head)
    # =========================================================================
    if cfg.stage2_epochs > 0:
        print("\n--- STAGE 2: Fine-Tuning ResNet-50 Layer4 and Classification Head ---")
        model.unfreeze_stage2()
        param_groups = model.get_stage2_param_groups(
            backbone_lr=cfg.stage2_backbone_lr,
            classifier_lr=cfg.stage2_classifier_lr,
            weight_decay=cfg.weight_decay,
        )
        optimizer_stage2 = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_stage2, T_max=cfg.stage2_epochs, eta_min=1e-6
        )

        for ep in range(1, cfg.stage2_epochs + 1):
            current_epoch += 1
            print(f"\n[Stage 2] Epoch {ep}/{cfg.stage2_epochs} (Total: {current_epoch}/{total_epochs})")
            tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer_stage2,
                device,
                epoch_desc=f"Stage 2 Epoch {ep}",
                max_batches=max_train_batches,
            )
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_epoch(
                model, val_loader, criterion, device, desc="Validation"
            )
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["train_f1"].append(tr_f1)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            print(
                f"  Train -> Loss: {tr_loss:.4f} | Acc: {tr_acc * 100:.2f}% | Prec: {tr_prec:.4f} | Rec: {tr_rec:.4f} | F1: {tr_f1:.4f}\n"
                f"  Val   -> Loss: {val_loss:.4f} | Acc: {val_acc * 100:.2f}% | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}"
            )

            # Checkpoint if best
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = current_epoch
                save_checkpoint(
                    model,
                    cfg.best_model_path,
                    epoch=current_epoch,
                    val_metrics={"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1},
                    optimizer=optimizer_stage2,
                )
                torch.save(model.state_dict(), cfg.feature_extractor_path)

    total_time = time.time() - start_time
    print(f"\n[Training Complete] Total elapsed time: {total_time / 60:.2f} minutes")
    print(f"Best Validation Macro F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Plot training history curves
    plot_path = cfg.outputs_dir / "training_history.png"
    plot_training_history(history, plot_path)

    # Save training metadata
    train_metadata = {
        "dataset_root": str(cfg.dataset_root),
        "device": str(device),
        "random_seed": cfg.random_seed,
        "batch_size": cfg.batch_size,
        "stage1_epochs": cfg.stage1_epochs,
        "stage2_epochs": cfg.stage2_epochs,
        "stage1_lr": cfg.stage1_lr,
        "stage2_backbone_lr": cfg.stage2_backbone_lr,
        "stage2_classifier_lr": cfg.stage2_classifier_lr,
        "weight_decay": cfg.weight_decay,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "total_time_seconds": total_time,
        "history": history,
    }
    meta_path = cfg.outputs_dir / "train_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(train_metadata, f, indent=4)
    print(f"[Metadata] Training metadata saved to: {meta_path}")

    return train_metadata


def main():
    parser = argparse.ArgumentParser(description="HealthFusion-Transformer: Train ResNet-50 Imaging Branch")
    parser.add_argument("--data-dir", type=str, default=None, help="Root path of chest_xray dataset")
    parser.add_argument("--stage1-epochs", type=int, default=None, help="Stage 1 training epochs")
    parser.add_argument("--stage2-epochs", type=int, default=None, help="Stage 2 training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit number of batches per epoch (for quick testing)")
    args = parser.parse_args()

    cfg = ImagingConfig()
    if args.data_dir:
        cfg.dataset_root = Path(args.data_dir)
    if args.stage1_epochs is not None:
        cfg.stage1_epochs = args.stage1_epochs
    if args.stage2_epochs is not None:
        cfg.stage2_epochs = args.stage2_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.device:
        cfg.device = args.device

    train_pipeline(cfg, max_train_batches=args.max_train_batches)


if __name__ == "__main__":
    main()
