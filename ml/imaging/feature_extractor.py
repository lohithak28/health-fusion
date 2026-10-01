import argparse
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.imaging.config import ImagingConfig, config as default_config
from ml.imaging.model import HealthFusionResNet50
from ml.imaging.preprocessing import preprocess_single_image, get_eval_transforms
from ml.imaging.dataset import ChestXRayDataset
from ml.imaging.utils import get_device, load_checkpoint


class ImageFeatureExtractor:
    """
    Dedicated Feature Extraction Interface for HealthFusion-Transformer.
    Extracts the 2048-dimensional intermediate feature representation F_img
    from chest X-ray images, ready for multimodal fusion.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: ImagingConfig = default_config,
    ):
        self.cfg = cfg
        self.device = torch.device(device) if device else get_device()
        self.model = HealthFusionResNet50(
            num_classes=self.cfg.num_classes,
            pretrained=True,
            dropout_rate=self.cfg.dropout_rate,
        )

        chk_path = Path(checkpoint_path) if checkpoint_path else self.cfg.best_model_path
        if chk_path.exists():
            print(f"[FeatureExtractor] Loading trained weights from: {chk_path}")
            load_checkpoint(self.model, chk_path, self.device)
        else:
            print(
                f"[FeatureExtractor] Checkpoint not found at {chk_path}. "
                f"Using ImageNet-pretrained ResNet-50 weights."
            )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_from_path(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Extract F_img [1, 2048] directly from an image file path.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Chest X-ray image does not exist: {path}")

        tensor = preprocess_single_image(
            str(path),
            image_size=self.cfg.image_size,
            mean=self.cfg.norm_mean,
            std=self.cfg.norm_std,
        ).to(self.device)

        f_img = self.model.extract_features(tensor)
        return f_img.detach().cpu()  # [1, 2048]

    @torch.no_grad()
    def extract_from_pil(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Extract F_img [1, 2048] from a PIL Image instance.
        """
        tensor = preprocess_single_image(
            pil_image,
            image_size=self.cfg.image_size,
            mean=self.cfg.norm_mean,
            std=self.cfg.norm_std,
        ).to(self.device)

        f_img = self.model.extract_features(tensor)
        return f_img.detach().cpu()  # [1, 2048]

    @torch.no_grad()
    def extract_from_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract F_img [B, 2048] from a preprocessed batch tensor [B, 3, 224, 224].
        """
        batch_tensor = batch_tensor.to(self.device)
        f_img = self.model.extract_features(batch_tensor)
        return f_img.detach().cpu()

    @torch.no_grad()
    def extract_dataset(
        self,
        dataset_split_dir: Path,
        split_name: str = "test",
        batch_size: int = 32,
    ) -> Tuple[torch.Tensor, pd.DataFrame]:
        """
        Extract features for an entire split directory, returning:
        - feature_tensor: [N, 2048]
        - metadata_df: DataFrame with filename, label, class_name, split, feature_index
        """
        eval_transform = get_eval_transforms(
            image_size=self.cfg.image_size,
            mean=self.cfg.norm_mean,
            std=self.cfg.norm_std,
        )
        dataset = ChestXRayDataset(
            root_dir=dataset_split_dir,
            transform=eval_transform,
            class_to_idx=self.cfg.class_to_idx,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        all_features = []
        metadata_records = []
        idx_to_class = self.cfg.idx_to_class
        global_idx = 0

        print(f"[FeatureExtractor] Extracting features from {split_name} split ({len(dataset)} images)...")
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=f"Extracting {split_name}")):
            images = images.to(self.device)
            feats = self.model.extract_features(images).cpu()
            all_features.append(feats)

            for i in range(len(targets)):
                sample_path = dataset.get_sample_path(global_idx)
                label_val = targets[i].item()
                metadata_records.append({
                    "feature_index": global_idx,
                    "filename": sample_path.name,
                    "filepath": str(sample_path),
                    "label": label_val,
                    "class_name": idx_to_class[label_val],
                    "split": split_name,
                })
                global_idx += 1

        feature_matrix = torch.cat(all_features, dim=0)  # [N, 2048]
        metadata_df = pd.DataFrame(metadata_records)

        return feature_matrix, metadata_df


def extract_and_save_dataset_features(
    dataset_dir: Optional[Path] = None,
    split: str = "test",
    output_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    cfg: ImagingConfig = default_config,
) -> None:
    """
    CLI/script entrypoint to extract and save features to disk for downstream fusion.
    Saves:
      - <output_dir>/image_features.pt (Tensor of shape [N, 2048])
      - <output_dir>/image_feature_metadata.csv
    """
    root = Path(dataset_dir) if dataset_dir else cfg.dataset_root
    out_dir = Path(output_dir) if output_dir else cfg.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = ImageFeatureExtractor(checkpoint_path=checkpoint_path, cfg=cfg)

    if split == "all":
        splits_to_process = ["train", "val", "test"]
    else:
        splits_to_process = [split]

    for s in splits_to_process:
        s_dir = root / s
        if not s_dir.exists():
            print(f"[FeatureExtractor] Warning: Split dir not found: {s_dir}. Skipping.")
            continue

        feats, meta = extractor.extract_dataset(s_dir, split_name=s, batch_size=cfg.batch_size)

        pt_file = out_dir / f"image_features_{s}.pt" if split == "all" else out_dir / "image_features.pt"
        csv_file = out_dir / f"image_feature_metadata_{s}.csv" if split == "all" else out_dir / "image_feature_metadata.csv"

        torch.save(feats, pt_file)
        meta.to_csv(csv_file, index=False)

        print("\n" + "=" * 60)
        print(f"FEATURE EXTRACTION SUMMARY ({s.upper()} SPLIT):")
        print("=" * 60)
        print(f" Saved Feature Tensor : {pt_file}")
        print(f" Feature Matrix Shape : {feats.shape} (Expected: [N, 2048])")
        print(f" Feature Tensor Dtype : {feats.dtype}")
        print(f" Metadata CSV File    : {csv_file}")
        print(f" Number of Samples    : {len(meta)}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="HealthFusion-Transformer: Chest X-Ray Feature Extractor")
    parser.add_argument("--data-dir", type=str, default=None, help="Root path of chest_xray dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="Dataset split to extract")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save feature files")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_resnet50.pth")
    parser.add_argument("--image", type=str, default=None, help="Extract 2048-D feature for a single image file")
    args = parser.parse_args()

    cfg = ImagingConfig()
    if args.data_dir:
        cfg.dataset_root = Path(args.data_dir)

    if args.image:
        extractor = ImageFeatureExtractor(checkpoint_path=args.checkpoint, cfg=cfg)
        feat = extractor.extract_from_path(args.image)
        print(f"\nExtracted single image feature vector:")
        print(f" Input Image   : {args.image}")
        print(f" Feature Shape : {feat.shape} (Expected: [1, 2048])")
        print(f" Feature Dtype : {feat.dtype}")
        print(f" First 5 values: {feat[0, :5].tolist()}")
    else:
        extract_and_save_dataset_features(
            dataset_dir=cfg.dataset_root,
            split=args.split,
            output_dir=Path(args.output_dir) if args.output_dir else cfg.features_dir,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
