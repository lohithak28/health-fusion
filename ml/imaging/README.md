# HealthFusion-Transformer: Medical Imaging Branch (Chest X-Ray)

> **Paper / Project Title**: *HealthFusion-Transformer: A Novel Multimodal Data Fusion Framework for Enhanced Clinical Decision-Making Using Cross-Attentive Transformer Architectures*

This module implements the **Chest X-Ray (Medical Imaging) Branch** of HealthFusion-Transformer. Its primary engineering and clinical purpose is to extract a dense, clinically discriminative **2048-dimensional visual feature vector $\mathbf{F}_{\text{img}}$** from raw radiography scans using an ImageNet-pretrained ResNet-50 backbone with staged fine-tuning.

---

## 1. Architectural Role in HealthFusion-Transformer

The overall framework fuses three clinical modalities:
```text
Chest X-Ray Imaging (This Module)      EHR Clinical Records             Wearables / Time-Series
        ↓ (ResNet-50)                        ↓ (Transformer)                  ↓ (Bi-GRU)
   F_img [B, 2048]                      F_ehr [B, d_ehr]                 F_sens [B, d_sens]
        └────────────────────────────────────┬────────────────────────────────┘
                                             ↓
                        Hierarchical Cross-Attentive Transformer
                                             ↓
                                      Modality Dropout
                                             ↓
                                Attention-Guided Late Fusion
                                             ↓
                          Clinical Prediction & Uncertainty
```

In this module:
- The ResNet-50 backbone is trained on chest radiography classification (`NORMAL` vs `PNEUMONIA`).
- The final classification layer is strictly decoupled from the feature representation.
- The output handed off to the fusion module is **$\mathbf{F}_{\text{img}} \in \mathbb{R}^{B \times 2048}$**, *not* logits or probabilities.

---

## 2. Integration Contract ($\mathbf{F}_{\text{img}}$ Interface)

| Attribute | Specification | Notes |
| :--- | :--- | :--- |
| **Variable Name** | `F_img` | Standardized multimodal feature representation |
| **Data Type** | `torch.float32` | Standard PyTorch floating-point tensor |
| **Tensor Shape** | `[B, 2048]` | Batch size $B$, feature dimension 2048 |
| **Origin Layer** | Adaptive Average Pooling (post-`layer4`, pre-classifier) | Global spatial pooling of ResNet-50 |
| **Downstream Destination** | Hierarchical Cross-Attentive Transformer encoder | Multimodal fusion module |

### Code Example: How Future Fusion Modules Consume $\mathbf{F}_{\text{img}}$

```python
import torch
import torch.nn as nn

class CrossAttentiveFusion(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        # Project 2048-D imaging feature into common transformer embedding space
        self.img_projection = nn.Linear(2048, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, F_img: torch.Tensor, F_ehr: torch.Tensor, F_sens: torch.Tensor):
        """
        F_img:  [B, 2048] from ml.imaging
        F_ehr:  [B, d_ehr]
        F_sens: [B, d_sens]
        """
        assert F_img.shape[-1] == 2048, f"Expected 2048-D F_img, got {F_img.shape}"
        assert F_img.dtype == torch.float32

        # 1. Project into common d_model space
        H_img = self.norm(self.img_projection(F_img)).unsqueeze(1)  # [B, 1, d_model]
        
        # 2. Multi-head cross attention with EHR and Sensor modalities
        # ... downstream transformer cross-attention layers ...
        return H_img
```

---

## 3. Dataset Specification

The module uses the Chest X-Ray Images (Pneumonia) dataset:
- `NORMAL` (Class 0): 1,341 train | 8 val | 234 test
- `PNEUMONIA` (Class 1): 3,875 train | 8 val | 390 test
- **Class Imbalance Strategy**: Inverse-frequency weighted Cross-Entropy Loss ($w_0 \approx 1.9448, w_1 \approx 0.6730$) to prevent majority-class bias.

> **Research Integrity**: This dataset provides an experimental chest X-ray representation. It does not implement the paper's full 10-disease multi-label clinical taxonomy.

---

## 4. Installation & Environment

From the project root:

```bash
pip install -r requirements.txt
```

Required packages: `torch>=2.14.0`, `torchvision>=0.29.0`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pillow`, `tqdm`.

---

## 5. Configuration

Paths and parameters are managed in `ml/imaging/config.py`.

By default, the dataset path is resolved dynamically relative to the project root:
```python
PROJECT_ROOT / "archive" / "chest_xray"
```

To override the dataset location with a custom directory, set the `CHEST_XRAY_DATA_DIR` environment variable:
```bash
# Windows PowerShell
$env:CHEST_XRAY_DATA_DIR = "D:\custom_path\archive\chest_xray"

# Command Prompt
set CHEST_XRAY_DATA_DIR=D:\custom_path\archive\chest_xray
```
Alternatively, pass `--data-dir <path>` to any CLI script.

---

## 6. Execution Commands

### 6.1 Training
Execute staged transfer learning (Stage 1: freeze backbone; Stage 2: fine-tune layer 4):
```bash
python -m ml.imaging.train --batch-size 32
```
Options:
- `--stage1-epochs <int>`: Epochs for classifier head training (default: 3)
- `--stage2-epochs <int>`: Epochs for deep fine-tuning (default: 5)
- `--batch-size <int>`: Batch size (default: 32)
- `--max-train-batches <int>`: Optional limit for fast testing / dry-runs

### 6.2 Evaluation
Evaluate the best checkpoint on the official test set (624 images):
```bash
python -m ml.imaging.evaluate
```

### 6.3 Single-Image Inference & $\mathbf{F}_{\text{img}}$ Extraction
Run diagnostic prediction on a single radiograph:
```bash
python -m ml.imaging.inference --image "path/to/chest_xray.jpeg"
```
Optional flags:
- `--save-feature "features/sample_f_img.pt"`: Save extracted [1, 2048] tensor to disk
- `--print-vector`: Print raw feature vector values in terminal

### 6.4 Batch Feature Extraction for Downstream Fusion
Extract 2048-D features for an entire split:
```bash
# Extract test split features
python -m ml.imaging.feature_extractor --split test

# Extract all splits (train, val, test)
python -m ml.imaging.feature_extractor --split all
```

---

## 7. Programmatic API

```python
from ml.imaging.model import HealthFusionResNet50
from ml.imaging.feature_extractor import ImageFeatureExtractor
from ml.imaging.inference import predict_chest_xray

# 1. Direct prediction and feature extraction
result = predict_chest_xray("path/to/xray.jpeg")
print(result["prediction"])      # 'NORMAL' or 'PNEUMONIA'
print(result["probability"])     # e.g. 0.88
F_img = result["features"]       # torch.Tensor with shape [1, 2048]

# 2. Reusable Feature Extractor
extractor = ImageFeatureExtractor()
f_img = extractor.extract_from_path("path/to/xray.jpeg") # [1, 2048]
```

---

## 8. Output Artifacts

All outputs are saved deterministically:

- `models/imaging/best_resnet50.pth`: Best fine-tuned checkpoint
- `models/imaging/image_feature_extractor.pth`: Feature extractor weights
- `outputs/imaging/metrics.json`: Accuracy, Precision, Recall, F1, ROC AUC
- `outputs/imaging/classification_report.txt`: Tabular metrics breakdown
- `outputs/imaging/confusion_matrix.png`: Normalized test confusion matrix
- `outputs/imaging/training_history.png`: Staged training and validation loss/F1 curves
- `outputs/imaging/sample_predictions.csv`: Sample-level predictions and confidence
- `features/image_features.pt`: Dense feature tensor `[N, 2048]`
- `features/image_feature_metadata.csv`: Aligned sample index, filenames, and labels

---

## 9. Troubleshooting

1. **CUDA Out of Memory / Slow Execution**: Set `--batch-size 16`. If CUDA is not available, the pipeline automatically detects CPU.
2. **Corrupted or Hidden Files**: Handled automatically; hidden files starting with `.` or non-image extensions are ignored.
3. **Headless Server Error (`_tkinter`)**: The pipeline configures the `Agg` non-interactive matplotlib backend by default.
