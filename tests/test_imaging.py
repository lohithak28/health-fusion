import unittest
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from ml.imaging.config import ImagingConfig
from ml.imaging.preprocessing import (
    ConvertToRGB,
    get_train_transforms,
    get_eval_transforms,
    preprocess_single_image,
)
from ml.imaging.model import HealthFusionResNet50
from ml.imaging.feature_extractor import ImageFeatureExtractor
from ml.imaging.dataset import ChestXRayDataset


class TestImagingPipeline(unittest.TestCase):
    """
    Validation test suite for HealthFusion-Transformer Medical Imaging Branch.
    """

    def setUp(self):
        self.cfg = ImagingConfig()
        self.device = torch.device("cpu")  # Run test suite deterministically on CPU

    def test_1_rgb_conversion_and_preprocessing(self):
        """Test preprocessing handles 1-channel grayscale conversion and produces [1, 3, 224, 224]."""
        # Create a mock grayscale chest X-ray
        gray_img = Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8), mode="L")
        converter = ConvertToRGB()
        rgb_img = converter(gray_img)
        self.assertEqual(rgb_img.mode, "RGB")

        # Test preprocessing pipeline
        tensor = preprocess_single_image(rgb_img, image_size=(224, 224))
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_2_model_forward_shapes(self):
        """Test model forward pass returns logits [B, 2] and features F_img [B, 2048]."""
        model = HealthFusionResNet50(num_classes=2, pretrained=False).to(self.device)
        model.eval()

        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)

        logits, features = model(dummy_input, return_features=True)

        self.assertEqual(logits.shape, (batch_size, 2), "Logits shape must be [B, 2]")
        self.assertEqual(features.shape, (batch_size, 2048), "Feature vector F_img shape must be [B, 2048]")
        self.assertEqual(features.dtype, torch.float32)

    def test_3_feature_extractor_interface(self):
        """Test dedicated extract_features method satisfies the [B, 2048] contract."""
        model = HealthFusionResNet50(num_classes=2, pretrained=False).to(self.device)
        model.eval()

        # Single image
        single_input = torch.randn(1, 3, 224, 224).to(self.device)
        f_img = model.extract_features(single_input)
        self.assertEqual(f_img.shape, (1, 2048), "Single image feature must be [1, 2048]")

        # Batch of 8 images
        batch_input = torch.randn(8, 3, 224, 224).to(self.device)
        f_batch = model.extract_features(batch_input)
        self.assertEqual(f_batch.shape, (8, 2048), "Batch feature must be [8, 2048]")

    def test_4_single_image_extraction_with_pil(self):
        """Test ImageFeatureExtractor with a PIL Image."""
        extractor = ImageFeatureExtractor(device="cpu", cfg=self.cfg)
        mock_img = Image.new("RGB", (256, 256), color="white")
        feat = extractor.extract_from_pil(mock_img)
        self.assertEqual(feat.shape, (1, 2048))

    def test_5_missing_image_raises_error(self):
        """Test that missing image file raises FileNotFoundError or ValueError."""
        extractor = ImageFeatureExtractor(device="cpu", cfg=self.cfg)
        non_existent_file = Path("non_existent_chest_xray_12345.jpg")
        with self.assertRaises(FileNotFoundError):
            extractor.extract_from_path(non_existent_file)

    def test_6_staged_finetuning_param_freezing(self):
        """Verify layer freezing and unfreezing behavior across stages."""
        model = HealthFusionResNet50(num_classes=2, pretrained=False)

        # Stage 1: Freeze backbone
        model.freeze_backbone()
        for name, param in model.named_parameters():
            if "classifier" in name:
                self.assertTrue(param.requires_grad, f"{name} should be trainable in stage 1")
            else:
                self.assertFalse(param.requires_grad, f"{name} should be frozen in stage 1")

        # Stage 2: Unfreeze layer4 + classifier
        model.unfreeze_stage2()
        for name, param in model.named_parameters():
            if "layer4" in name or "classifier" in name:
                self.assertTrue(param.requires_grad, f"{name} should be trainable in stage 2")
            else:
                self.assertFalse(param.requires_grad, f"{name} should be frozen in stage 2")

    def test_7_dataset_loading_if_present(self):
        """Test dataset loading if the dataset directory is present."""
        if self.cfg.dataset_root.exists() and self.cfg.val_dir.exists():
            val_dataset = ChestXRayDataset(
                self.cfg.val_dir,
                transform=get_eval_transforms(),
                class_to_idx=self.cfg.class_to_idx,
            )
            self.assertGreater(len(val_dataset), 0)
            img, label = val_dataset[0]
            self.assertEqual(img.shape, (3, 224, 224))
            self.assertIn(label, [0, 1])

    def test_8_project_root_relative_dataset_paths(self):
        """Verify that dataset path is resolved relative to project_root / archive / chest_xray and all 6 folders exist."""
        expected_dataset = self.cfg.project_root / "archive" / "chest_xray"
        self.assertEqual(self.cfg.dataset_root.resolve(), expected_dataset.resolve())

        required_subdirs = [
            self.cfg.train_dir / "NORMAL",
            self.cfg.train_dir / "PNEUMONIA",
            self.cfg.val_dir / "NORMAL",
            self.cfg.val_dir / "PNEUMONIA",
            self.cfg.test_dir / "NORMAL",
            self.cfg.test_dir / "PNEUMONIA",
        ]
        for subdir in required_subdirs:
            self.assertTrue(subdir.exists(), f"Missing required dataset directory: {subdir}")
            self.assertTrue(subdir.is_dir(), f"Not a directory: {subdir}")

    def test_9_real_sample_inference_and_feature_extraction(self):
        """Perform end-to-end inference and F_img extraction using an image from the relative dataset."""
        sample_img_dir = self.cfg.test_dir / "NORMAL"
        img_files = list(sample_img_dir.glob("*.jpeg")) + list(sample_img_dir.glob("*.jpg"))
        self.assertGreater(len(img_files), 0, "No sample images found in test/NORMAL")

        sample_path = img_files[0]
        extractor = ImageFeatureExtractor(device="cpu", cfg=self.cfg)
        f_img = extractor.extract_from_path(sample_path)

        self.assertEqual(f_img.shape, (1, 2048), "F_img shape must be [1, 2048]")
        self.assertEqual(f_img.dtype, torch.float32, "F_img dtype must be torch.float32")
        self.assertFalse(torch.isnan(f_img).any(), "F_img contains NaN values")


if __name__ == "__main__":
    unittest.main()
