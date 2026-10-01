# HealthFusion-Transformer: Medical Imaging Dataset Specification

## 1. Dataset Overview

This directory documents the medical imaging dataset utilized for implementing and evaluating the chest X-ray branch of the **HealthFusion-Transformer** architecture.

- **Dataset Name**: Chest X-Ray Images (Pneumonia)
- **Modality**: Chest Radiography (CXR) - Posterior-Anterior (PA) / Anterior-Posterior (AP) views
- **Development Task**: Binary classification (NORMAL = 0, PNEUMONIA = 1)
- **Local Source Path**: `PROJECT_ROOT/archive/chest_xray` (configurable via `CHEST_XRAY_DATA_DIR` environment variable)

## 2. Directory Structure

The dataset must follow the standard split layout:

```text
chest_xray/
├── train/
│   ├── NORMAL/      (1,341 images)
│   └── PNEUMONIA/   (3,875 images)
├── val/
│   ├── NORMAL/      (8 images)
│   └── PNEUMONIA/   (8 images)
└── test/
    ├── NORMAL/      (234 images)
    └── PNEUMONIA/   (390 images)
```

> **Note**: An internal redundant `chest_xray/chest_xray/` folder and any macOS metadata folders (`__MACOSX`) are ignored by the loader pipeline.

## 3. Class Imbalance Profile

- **Training Distribution**:
  - `NORMAL`: 1,341 (25.71%)
  - `PNEUMONIA`: 3,875 (74.29%)
  - **Imbalance Ratio**: ~1:2.89
- **Handling Strategy**: Inverse frequency class weighting in `CrossEntropyLoss`:
  $$w_c = \frac{N_{\text{total}}}{C \cdot N_c}$$
  - $w_{\text{NORMAL}} \approx 1.9448$
  - $w_{\text{PNEUMONIA}} \approx 0.6730$
  This penalizes minority class misclassification without distorting natural chest X-ray feature distributions with artificial oversampling.

## 4. Academic and Clinical Integrity Notice

This dataset is an experimental prototype dataset for engineering and evaluating the deep visual representation branch of the project.
- It **does NOT** represent the complete 10-disease multi-label classification defined in the full HealthFusion-Transformer clinical scope.
- It is **NOT** a clinically validated medical device.
