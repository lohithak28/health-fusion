from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class HealthFusionResNet50(nn.Module):
    """
    Medical Imaging Branch of the HealthFusion-Transformer Architecture.

    Architecture Flow:
        Input [B, 3, 224, 224]
               ↓
        ResNet-50 Convolutional Backbone
        (conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4)
               ↓
        Adaptive Average Pooling (1x1)
               ↓
        Flatten
               ↓
        F_img [B, 2048] (Reusable Multimodal Feature Vector)
               ↓
        Dropout (p=0.2, supports optional MC-Dropout for uncertainty)
               ↓
        Linear(2048 -> num_classes)
               ↓
        Logits [B, num_classes]

    Contract:
        extract_features(x) strictly outputs F_img of shape [B, 2048] with dtype float32,
        guaranteeing seamless compatibility with the downstream Hierarchical Cross-Attentive Transformer.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 2048

        # 1. Load ImageNet-pretrained ResNet-50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = resnet50(weights=weights)

        # 2. Extract feature representation backbone (all layers prior to original fc)
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        self.avgpool = base_resnet.avgpool

        # 3. Clinical task classification head
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the raw 2048-D multimodal visual feature representation F_img.
        
        Args:
            x: Input batch tensor of shape [B, 3, 224, 224]
        Returns:
            F_img: Feature tensor of shape [B, 2048]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # Shape: [B, 2048]
        return features

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass returning classification logits and feature vector.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            return_features: When True, returns (logits, F_img) tuple
        Returns:
            logits: [B, num_classes]
            features: [B, 2048] (if return_features=True)
        """
        features = self.extract_features(x)
        dropped = self.dropout(features)
        logits = self.classifier(dropped)

        if return_features:
            return logits, features
        return logits

    def freeze_backbone(self) -> None:
        """
        Stage 1 Transfer Learning:
        Freeze all backbone weights (conv1 through layer4), training only the classifier.
        """
        for name, param in self.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("[Model] Backbone layers frozen. Stage 1: Only classifier head is trainable.")

    def unfreeze_stage2(self) -> None:
        """
        Stage 2 Fine-Tuning:
        Unfreeze high-level convolutional block (layer4) and classifier head.
        Lower layers (layer1-3, conv1) remain frozen to prevent catastrophic forgetting.
        """
        for name, param in self.named_parameters():
            if "layer4" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("[Model] Stage 2: Unfroze layer4 and classifier for deep feature adaptation.")

    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] All parameters unfrozen.")

    def get_stage2_param_groups(
        self,
        backbone_lr: float = 5e-5,
        classifier_lr: float = 2e-4,
        weight_decay: float = 1e-4,
    ):
        """
        Returns parameter groups with differentiated learning rates for staged fine-tuning.
        """
        backbone_params = []
        classifier_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": classifier_params, "lr": classifier_lr, "weight_decay": weight_decay},
        ]
