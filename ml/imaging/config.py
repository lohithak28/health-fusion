import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List
import torch


def get_project_root() -> Path:
    """Dynamically determine the project root directory from this file's location."""
    # ml/imaging/config.py -> parent (imaging) -> parent (ml) -> parent (project root)
    return Path(__file__).resolve().parent.parent.parent


def default_dataset_path() -> Path:
    """
    Dynamically determine dataset path without hardcoding absolute paths.

    Resolution precedence:
    1. CHEST_XRAY_DATA_DIR environment variable (if set and exists)
    2. PROJECT_ROOT / "archive" / "chest_xray" (preferred project-relative location)
    3. PROJECT_ROOT / "data" / "chest_xray" (fallback if in data/)
    4. Path.home() / "Downloads" / "archive" / "chest_xray" (legacy download fallback)
    """
    # 1. Check environment variable override
    env_path = os.environ.get("CHEST_XRAY_DATA_DIR")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # 2. Preferred default: PROJECT_ROOT / "archive" / "chest_xray"
    project_root = get_project_root()
    preferred = project_root / "archive" / "chest_xray"
    if preferred.exists():
        return preferred

    # 3. Check local project data directory fallback
    local_data = project_root / "data" / "chest_xray"
    if local_data.exists():
        return local_data

    # 4. Check user downloads folder fallback
    user_downloads = Path.home() / "Downloads" / "archive" / "chest_xray"
    if user_downloads.exists():
        return user_downloads

    # Default to preferred project-relative location
    return preferred


@dataclass
class ImagingConfig:
    # --- Paths ---
    project_root: Path = field(default_factory=get_project_root)
    dataset_root: Path = field(default_factory=default_dataset_path)

    @property
    def train_dir(self) -> Path:
        return self.dataset_root / "train"

    @property
    def val_dir(self) -> Path:
        return self.dataset_root / "val"

    @property
    def test_dir(self) -> Path:
        return self.dataset_root / "test"

    @property
    def models_dir(self) -> Path:
        path = self.project_root / "models" / "imaging"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def outputs_dir(self) -> Path:
        path = self.project_root / "outputs" / "imaging"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def features_dir(self) -> Path:
        path = self.project_root / "features"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Checkpoint filenames
    best_model_filename: str = "best_resnet50.pth"
    feature_extractor_filename: str = "image_feature_extractor.pth"

    @property
    def best_model_path(self) -> Path:
        return self.models_dir / self.best_model_filename

    @property
    def feature_extractor_path(self) -> Path:
        return self.models_dir / self.feature_extractor_filename

    # --- Classes & Labels ---
    class_names: List[str] = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])
    num_classes: int = 2
    class_to_idx: dict = field(
        default_factory=lambda: {"NORMAL": 0, "PNEUMONIA": 1}
    )
    idx_to_class: dict = field(
        default_factory=lambda: {0: "NORMAL", 1: "PNEUMONIA"}
    )

    # --- Preprocessing & Model Specifications ---
    image_size: Tuple[int, int] = (224, 224)
    # Standard ImageNet normalization parameters
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Multimodal feature vector dimension (F_img)
    feature_dim: int = 2048

    # --- Hyperparameters & Training Settings ---
    batch_size: int = 32
    num_workers: int = 0  # 0 is safest and most reliable on Windows
    stage1_epochs: int = 3  # Feature head training with frozen backbone
    stage2_epochs: int = 5  # Fine-tuning unfrozen higher layers
    stage1_lr: float = 1e-3  # Learning rate for classifier head in stage 1
    stage2_backbone_lr: float = 5e-5  # Slower rate for backbone fine-tuning
    stage2_classifier_lr: float = 2e-4  # Classifier rate in stage 2
    weight_decay: float = 1e-4
    dropout_rate: float = 0.2
    early_stopping_patience: int = 4

    # Seed & Device
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Singleton instance for quick access
config = ImagingConfig()
