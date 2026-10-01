import argparse
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image
import torch
import torch.nn.functional as F

from ml.imaging.config import ImagingConfig, config as default_config
from ml.imaging.model import HealthFusionResNet50
from ml.imaging.preprocessing import preprocess_single_image
from ml.imaging.utils import get_device, load_checkpoint


def predict_chest_xray(
    image: Union[str, Path, Image.Image],
    checkpoint_path: Optional[Union[str, Path]] = None,
    cfg: ImagingConfig = default_config,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Programmatic inference API for an individual chest X-ray image.
    
    Returns dictionary with:
      - 'prediction': class name ('NORMAL' or 'PNEUMONIA')
      - 'class_index': 0 or 1
      - 'probability': confidence score for predicted class
      - 'prob_normal': probability of Normal
      - 'prob_pneumonia': probability of Pneumonia
      - 'features': torch.Tensor of shape [1, 2048] (F_img)
    """
    dev = device or (torch.device(cfg.device) if hasattr(cfg, "device") else get_device())
    chk_path = Path(checkpoint_path) if checkpoint_path else cfg.best_model_path

    # Instantiate model
    model = HealthFusionResNet50(
        num_classes=cfg.num_classes,
        pretrained=True,
        dropout_rate=cfg.dropout_rate,
    )

    if chk_path.exists():
        load_checkpoint(model, chk_path, dev)
    else:
        print(f"[Inference Warning] Checkpoint '{chk_path}' not found. Using pretrained baseline.")

    model.to(dev)
    model.eval()

    # Preprocess
    img_tensor = preprocess_single_image(
        image,
        image_size=cfg.image_size,
        mean=cfg.norm_mean,
        std=cfg.norm_std,
    ).to(dev)

    with torch.no_grad():
        logits, features = model(img_tensor, return_features=True)
        probs = F.softmax(logits, dim=1).cpu().squeeze(0)  # [2]
        features = features.cpu()  # [1, 2048]

    pred_idx = int(torch.argmax(probs).item())
    pred_class = cfg.idx_to_class[pred_idx]
    pred_prob = float(probs[pred_idx].item())

    return {
        "prediction": pred_class,
        "class_index": pred_idx,
        "probability": pred_prob,
        "prob_normal": float(probs[0].item()),
        "prob_pneumonia": float(probs[1].item()),
        "features": features,  # Shape: [1, 2048]
    }


def main():
    parser = argparse.ArgumentParser(description="HealthFusion-Transformer: Single Chest X-Ray Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input chest X-ray image file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint")
    parser.add_argument("--save-feature", type=str, default=None, help="Optional path to save extracted F_img tensor (.pt)")
    parser.add_argument("--print-vector", action="store_true", help="Print the 2048-D feature vector values")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[Error] Image file not found: {image_path}")
        return

    result = predict_chest_xray(image_path, checkpoint_path=args.checkpoint)

    print("\n" + "=" * 55)
    print("HEALTHFUSION-TRANSFORMER: INFERENCE RESULT")
    print("=" * 55)
    print(f" Input Image   : {image_path.name}")
    print(f" Prediction    : {result['prediction']}")
    print(f" Probability   : {result['probability']:.4f} ({result['probability'] * 100:.2f}%)")
    print(f" Normal Prob   : {result['prob_normal']:.4f}")
    print(f" Pneumonia Prob: {result['prob_pneumonia']:.4f}")
    print(f" F_img Shape   : {list(result['features'].shape)} (Expected: [1, 2048])")
    print(f" F_img Dtype   : {result['features'].dtype}")
    print("=" * 55)

    if args.save_feature:
        save_pt = Path(args.save_feature)
        save_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result["features"], save_pt)
        print(f"[Feature Extracted] Saved F_img to: {save_pt}")

    if args.print_vector:
        print("\nFirst 10 values of F_img:")
        print(result["features"][0, :10].tolist())


if __name__ == "__main__":
    main()
