"""
HealthFusion-Transformer: Medical Imaging Branch (Chest X-Ray)

Extracts 2048-dimensional deep visual features F_img using ResNet-50
for downstream multimodal cross-attentive fusion.
"""

from ml.imaging.config import ImagingConfig
from ml.imaging.model import HealthFusionResNet50
from ml.imaging.feature_extractor import ImageFeatureExtractor

__all__ = ["ImagingConfig", "HealthFusionResNet50", "ImageFeatureExtractor"]
