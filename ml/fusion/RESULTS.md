# HealthFusion-Transformer — Fusion Results

## Overview

Member 3 implemented a three-modality fusion prototype combining:

1. Chest X-ray image features
2. EHR features
3. ECG physiological time-series features

The fusion pipeline includes modality-specific encoders followed by
Transformer-based cross-modal attention.

---

## Input Modalities

### Imaging

Source:

`features/image_features.pt`

Shape:

`[624, 2048]`

The 2048-dimensional representation is produced by the existing
ResNet-50 imaging branch.

### EHR

Source:

`features/labels/admission_labels.csv`

The current prototype uses:

- Age
- Gender

as the EHR representation.

### ECG

Source:

`features/ecg_waveforms/ecg_waveforms.npy`

Shape:

`[659, 12, 1000]`

This represents 12-lead ECG recordings sampled over 1000 time points.

---

# 1. Medical Imaging Branch — CXR

The standalone imaging branch uses a ResNet-50 backbone for
chest X-ray pneumonia classification.

## Test Dataset

- Total test samples: 624
- NORMAL: 234
- PNEUMONIA: 390

## Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 0.7660 |
| Macro Precision | 0.7652 |
| Macro Recall | 0.7829 |
| Macro F1 | 0.7621 |
| ROC-AUC | 0.8729 |

## Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.6419 | 0.8504 | 0.7316 | 234 |
| PNEUMONIA | 0.8885 | 0.7154 | 0.7926 | 390 |

## Confusion Matrix

| Actual / Predicted | NORMAL | PNEUMONIA |
|---|---:|---:|
| NORMAL | 199 | 35 |
| PNEUMONIA | 111 | 279 |

The extracted downstream imaging representation is:

`F_img ∈ R[B, 2048]`

and is stored in:

`features/image_features.pt`

---

# 2. EHR + ECG Baseline

The EHR + ECG pipeline was constructed using the MIMIC-IV
clinical and ECG data.

The resulting admission-level dataset contains:

- 149 admission-level examples
- 136 negative examples
- 13 positive examples
- Positive rate: 8.72%

The EHR representation currently consists of:

- Age
- Gender

The ECG representation is generated from 12-lead ECG waveforms.

## EHR + ECG Training

The baseline fusion model was trained using:

- BCEWithLogitsLoss
- Positive-class weighting
- AdamW optimization
- 20 training epochs

The positive-class weight was approximately:

`10.4615`

## Initial Pipeline Metrics

These metrics were calculated on the same cohort used for training and
therefore represent training-cohort performance rather than held-out
generalization performance.

| Metric | Result |
|---|---:|
| Accuracy | 0.9664 |
| Balanced Accuracy | 0.8425 |
| Precision | 0.9000 |
| Recall | 0.6923 |
| F1 | 0.7826 |
| AUROC | 0.9231 |

Model:

`features/models/member3_ehr_ecg_baseline.pt`

---

# 3. Three-Way Fusion Dataset

The three-way prototype combines:

```text
CXR + EHR + ECG