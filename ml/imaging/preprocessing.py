from pathlib import Path
from typing import Tuple, Union
from PIL import Image
import torch
from torchvision import transforms


class ConvertToRGB:
    """
    Ensures input image is converted to 3-channel RGB.
    Essential for chest X-rays which are predominantly stored as single-channel grayscale (mode 'L').
    ResNet-50 expects 3 input channels.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Construct clinically conservative training augmentation transforms for chest X-rays.
    Avoids aggressive distortions (excessive zoom, elastic deformation, shear, high contrast inversion)
    to preserve delicate lung markings, consolidation patterns, and anatomical structures.
    """
    return transforms.Compose([
        ConvertToRGB(),
        transforms.Resize(image_size),
        # Subtle rotation up to +/- 5 degrees to mimic slight patient posture variations
        transforms.RandomRotation(degrees=5),
        # Subtle horizontal flip (chest X-rays preserve bilateral anatomy)
        transforms.RandomHorizontalFlip(p=0.5),
        # Very gentle brightness and contrast variation (+/- 5%) to simulate X-ray tube exposure variation
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Deterministic validation, testing, and inference transform pipeline.
    Ensures zero stochastic data augmentation during evaluation.
    """
    return transforms.Compose([
        ConvertToRGB(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def preprocess_single_image(
    image: Union[str, Path, Image.Image],
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Helper to preprocess an individual image file path or PIL Image instance.
    Returns tensor with shape [1, 3, 224, 224].
    """
    if isinstance(image, (str, Path)):
        try:
            image = Image.open(str(image))
        except Exception as e:
            raise ValueError(f"Failed to open image file '{image}': {str(e)}")

    transform = get_eval_transforms(image_size=image_size, mean=mean, std=std)
    tensor = transform(image)  # [3, 224, 224]
    return tensor.unsqueeze(0)  # [1, 3, 224, 224]
