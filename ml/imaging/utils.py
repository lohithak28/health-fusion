import os
import random
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across python, numpy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect and return available compute device with informative logging."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[Device] CUDA not available. Running on CPU.")
    return dev


def save_checkpoint(
    model: torch.nn.Module,
    filepath: Path,
    epoch: int = None,
    val_metrics: Dict[str, Any] = None,
    optimizer: torch.optim.Optimizer = None,
) -> None:
    """Save model checkpoint safely."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_metrics": val_metrics or {},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, filepath)
    print(f"[Checkpoint] Successfully saved checkpoint to: {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    filepath: Path,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, Any]:
    """Load model checkpoint safely."""
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        return {"model_state_dict": checkpoint}


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Path,
) -> None:
    """
    Plot and save training/validation loss, accuracy, and F1 curves.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("HealthFusion-Transformer: ResNet-50 Training History", fontsize=14, fontweight="bold")

    # 1. Loss
    axes[0].plot(epochs, history.get("train_loss", []), "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history.get("val_loss", []), "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Accuracy
    axes[1].plot(epochs, history.get("train_acc", []), "o-", label="Train Acc", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, history.get("val_acc", []), "s--", label="Val Acc", color="#d62728", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Macro F1
    axes[2].plot(epochs, history.get("train_f1", []), "o-", label="Train F1", color="#9467bd", linewidth=2)
    axes[2].plot(epochs, history.get("val_f1", []), "s--", label="Val F1", color="#8c564b", linewidth=2)
    axes[2].set_title("Macro F1-Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Training history plot saved to: {save_path}")
