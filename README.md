# HealthFusion-Transformer

## A Novel Multimodal Data Fusion Framework for Enhanced Clinical Decision-Making Using Cross-Attentive Transformer Architectures

---

### Project Overview

**HealthFusion-Transformer** is an advanced multimodal deep learning framework designed to integrate heterogeneous clinical data sources for enhanced diagnostic precision:
1. **Medical Imaging Branch** (Chest Radiography / CXR): ResNet-50 backbone extracting dense visual feature representations ($\mathbf{F}_{\text{img}} \in \mathbb{R}^{B \times 2048}$).
2. **Electronic Health Records (EHR) Branch**: Transformer encoder processing tabular clinical variables and laboratory records ($\mathbf{F}_{\text{ehr}}$).
3. **Wearable / Sensor Branch**: Bidirectional GRU modeling physiological time-series signals ($\mathbf{F}_{\text{sens}}$).
4. **Multimodal Fusion Core**: Hierarchical Cross-Attentive Transformer with modality dropout, attention-guided late fusion, and clinical uncertainty estimation.

---

### Current Implementation Status: Medical Imaging Branch (Complete)

This repository contains the complete, tested, and validated implementation of the **Medical Imaging (Chest X-Ray) Branch**:

- **Backbone**: ResNet-50 with staged transfer learning (Stage 1: frozen backbone; Stage 2: layer 4 fine-tuning).
- **Clinical Task**: Pneumonia detection (NORMAL = 0, PNEUMONIA = 1) on experimental chest X-ray radiography.
- **Multimodal Output**: Decoupled $\mathbf{F}_{\text{img}}$ visual feature representation ($\text{shape} = [B, 2048]$, dtype `float32`).
- **Class Imbalance Strategy**: Inverse-frequency weighted Cross-Entropy Loss.

---

### Directory Structure

```text
c:/health-transformer/
├── archive/
│   └── chest_xray/
│       ├── train/                   <- NORMAL & PNEUMONIA
│       ├── val/                     <- Validation images
│       └── test/                    <- Official test evaluation split
├── data/
│   └── imaging/
│       └── README.md                <- Dataset layout & class distribution
├── features/
│   ├── image_features.pt            <- Extracted [N, 2048] tensor (F_img)
│   └── image_feature_metadata.csv   <- Sample metadata and feature alignment
├── ml/
│   └── imaging/
│       ├── __init__.py              <- Package exports
│       ├── config.py                <- Dynamic configuration & paths
│       ├── dataset.py               <- PyTorch Dataset, loader, class weights
│       ├── preprocessing.py         <- Medical conservative transforms
│       ├── model.py                 <- HealthFusionResNet50 architecture
│       ├── train.py                 <- Staged fine-tuning training pipeline
│       ├── evaluate.py              <- Evaluation & clinical metric generation
│       ├── inference.py             <- Single-image diagnostic CLI & API
│       ├── feature_extractor.py     <- 2048-D F_img extraction pipeline
│       ├── utils.py                 <- Checkpoints, seed, plotting
│       └── README.md                <- Imaging branch documentation & contract
├── models/
│   └── imaging/
│       ├── best_resnet50.pth        <- Fine-tuned model checkpoint
│       └── image_feature_extractor.pth <- Feature extractor weights
├── outputs/
│   └── imaging/
│       ├── classification_report.txt<- Test precision, recall, F1
│       ├── confusion_matrix.png     <- Test confusion matrix plot
│       ├── metrics.json             <- Quantitative test evaluation metrics
│       ├── sample_predictions.csv   <- Individual test predictions & confidence
│       ├── train_metadata.json      <- Staged training hyperparameters & logs
│       └── training_history.png     <- Loss and F1 training curves
├── tests/
│   └── test_imaging.py              <- Comprehensive unit & integration tests
├── requirements.txt                 <- Dependency specification
└── README.md                        <- Project documentation
```

---

### Quick Start

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run Test Suite
```bash
python -m unittest tests/test_imaging.py
```

#### 3. Single-Image Inference
```bash
python -m ml.imaging.inference --image "path/to/xray.jpeg"
```

#### 4. Train the Imaging Branch
```bash
python -m ml.imaging.train --batch-size 32
```

#### 5. Evaluate on Test Split
```bash
python -m ml.imaging.evaluate
```

#### 6. Extract Multimodal Features ($\mathbf{F}_{\text{img}}$)
```bash
python -m ml.imaging.feature_extractor --split test
```

---

### Downstream Fusion Contract ($\mathbf{F}_{\text{img}}$)

The Medical Imaging branch strictly adheres to the integration contract:
- **Output**: $\mathbf{F}_{\text{img}} \in \mathbb{R}^{B \times 2048}$, `torch.float32`.
- Logits or probabilities are **not** passed to the fusion module.
- The downstream Hierarchical Cross-Attentive Transformer receives $\mathbf{F}_{\text{img}}$ directly from `HealthFusionResNet50.extract_features()` or `ImageFeatureExtractor`.

Refer to [`ml/imaging/README.md`](file:///c:/health-transformer/ml/imaging/README.md) for architectural details and fusion code examples.
