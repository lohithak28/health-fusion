from pathlib import Path
from typing import Dict, Tuple, Optional, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from ml.imaging.config import ImagingConfig, config as default_config
from ml.imaging.preprocessing import get_train_transforms, get_eval_transforms


class ChestXRayDataset(Dataset):
    """
    Robust Dataset wrapper for Chest X-Ray images.
    Filters out hidden files, verifies integrity, and provides sample metadata.
    """
    def __init__(
        self,
        root_dir: Path,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {self.root_dir}\n"
                f"Please ensure the chest X-ray dataset is extracted and path is configured."
            )

        self.transform = transform
        self.class_to_idx = class_to_idx or {"NORMAL": 0, "PNEUMONIA": 1}
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        valid_extensions = {".jpeg", ".jpg", ".png", ".bmp"}
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder '{class_name}' not found under split directory: {self.root_dir}"
                )
            
            found_files = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_extensions and not f.name.startswith(".")
            ]
            if len(found_files) == 0:
                raise ValueError(f"No valid image files found in class folder: {class_dir}")

            for f in sorted(found_files):
                self.samples.append((f, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            raise RuntimeError(f"Corrupted or unreadable image at {img_path}: {str(e)}")

    def get_sample_path(self, idx: int) -> Path:
        return self.samples[idx][0]

    def get_class_counts(self) -> Dict[str, int]:
        counts = {cls: 0 for cls in self.class_to_idx.keys()}
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        for _, label in self.samples:
            cls_name = idx_to_class[label]
            counts[cls_name] += 1
        return counts


def calculate_class_weights(counts: Dict[str, int], device: torch.device) -> torch.Tensor:
    """
    Calculate inverse frequency class weights for CrossEntropyLoss:
    w_c = total_samples / (num_classes * count_c)
    
    This penalizes the loss for misclassifying the minority class (NORMAL) proportionally,
    addressing significant class imbalance without introducing synthetic artifacts or
    distorting batch statistics through forced oversampling.
    """
    total = sum(counts.values())
    num_classes = len(counts)
    # Sort by class index 0, 1 (NORMAL, PNEUMONIA)
    weights = [
        total / (num_classes * counts["NORMAL"]),
        total / (num_classes * counts["PNEUMONIA"]),
    ]
    tensor_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    return tensor_weights


def get_dataloaders(
    cfg: ImagingConfig = default_config,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Instantiates training, validation, and test datasets and DataLoaders.
    Also prints dataset statistics and returns computed class weights.
    """
    if not cfg.dataset_root.exists():
        raise FileNotFoundError(
            f"[Dataset Error] Configured dataset root does not exist: {cfg.dataset_root}\n"
            f"Please set the CHEST_XRAY_DATA_DIR environment variable or update ImagingConfig.dataset_root."
        )

    # Transforms
    train_transform = get_train_transforms(cfg.image_size, cfg.norm_mean, cfg.norm_std)
    eval_transform = get_eval_transforms(cfg.image_size, cfg.norm_mean, cfg.norm_std)

    # Datasets
    train_dataset = ChestXRayDataset(cfg.train_dir, transform=train_transform, class_to_idx=cfg.class_to_idx)
    val_dataset = ChestXRayDataset(cfg.val_dir, transform=eval_transform, class_to_idx=cfg.class_to_idx)
    test_dataset = ChestXRayDataset(cfg.test_dir, transform=eval_transform, class_to_idx=cfg.class_to_idx)

    # Distribution inspection
    train_counts = train_dataset.get_class_counts()
    val_counts = val_dataset.get_class_counts()
    test_counts = test_dataset.get_class_counts()

    print("\n" + "=" * 60)
    print("CHEST X-RAY DATASET SPLIT SUMMARY:")
    print("=" * 60)
    print(f" Train Split : {train_counts} (Total: {len(train_dataset)})")
    print(f" Val Split   : {val_counts} (Total: {len(val_dataset)})")
    print(f" Test Split  : {test_counts} (Total: {len(test_dataset)})")
    print("=" * 60)

    # Calculate class weighting for imbalanced loss
    device = torch.device(cfg.device)
    class_weights = calculate_class_weights(train_counts, device)
    print(f"[Class Imbalance Strategy] Using Class-Weighted CrossEntropyLoss:")
    print(f" NORMAL (Class 0) weight    : {class_weights[0].item():.4f}")
    print(f" PNEUMONIA (Class 1) weight : {class_weights[1].item():.4f}")
    print("=" * 60 + "\n")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_weights
