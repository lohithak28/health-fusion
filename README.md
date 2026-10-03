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

---

### Backend API, Database & Persistence Layer

HealthFusion includes a persistent FastAPI backend with PostgreSQL and structured file storage.

#### 1. PostgreSQL Database & Configuration
- **Database Engine**: PostgreSQL via SQLAlchemy ORM.
- **Connection Configuration**: Configured via the `DATABASE_URL` environment variable.
- A template `.env.example` is provided:
  ```bash
  DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/healthfusion
  UPLOAD_STORAGE_DIR=backend/uploads
  ```

#### 2. Local Database Setup
To set up and run with PostgreSQL:
1. Ensure PostgreSQL is installed and running (default port `5432`).
2. Create the application database:
   ```powershell
   # In psql or PostgreSQL command line:
   CREATE DATABASE healthfusion;
   ```
3. Set your connection URL in your terminal or `.env`:
   ```powershell
   $env:DATABASE_URL="postgresql+psycopg2://postgres:<your_password>@localhost:5432/healthfusion"
   ```
4. Tables are initialized automatically on startup by `Base.metadata.create_all()` via `backend/database/database.py`.

#### 3. Database Schema
- **`patients`**: Stores primary demographics (`id`, `patient_id` [indexed, unique], `age`, `gender`, `created_at`).
- **`uploaded_files`**: Stores metadata for clinical uploads (`id`, `patient_id` [FK], `prediction_id` [FK], `file_type`, `original_filename`, `stored_filename`, `stored_path`, `mime_type`, `file_size`, `uploaded_at`).
- **`predictions`**: Stores multimodal prediction records (`id`, `patient_id` [FK], `cxr_prediction`, `cxr_probability`, `multimodal_prediction`, `multimodal_probability`, `uncertainty`, `created_at`).

#### 4. File Storage Structure
Uploaded files are stored safely in persistent filesystem storage under `backend/uploads/` using collision-resistant filenames and strict path traversal protection:
```text
backend/uploads/
  └── <patient_id>/
        └── <prediction_id>/
              ├── cxr/
              │     └── cxr_<original_stem>_<uuid>.jpeg
              └── ecg/
                    └── ecg_<original_stem>_<uuid>.npy
```

#### 5. Core API Endpoints
- `GET /health` — Service health check.
- `POST /upload/image` — Upload CXR image, run ResNet-50 inference, extract and persist 2048-D $F_{img}$ feature tensor.
- `POST /ehr/` — Ingest and validate patient age (0–120) and gender (M/F), returning normalized inputs ($[1, 2]$).
- `POST /sensor/` — Validate 12-lead $\times$ 1000-sample ECG waveform array.
- `POST /predict/` — Run three-way multimodal prediction from features and persist record if DB is active.
- `POST /patients/analyze` — High-level multipart endpoint accepting CXR file/sample + EHR + ECG file/sample, running the entire ResNet-50 + Cross-Attention pipeline and persisting to PostgreSQL.
- `GET /patients/demo/samples` — Retrieve available sample CXR images and real ECG waveform indices for 1-click UI testing.
- `GET /patients/demo/image` — Stream sample CXR image for UI thumbnail preview.
- `GET /patients/{patient_id}` — Retrieve patient demographics and aggregated prediction/file counts.
- `GET /patients/{patient_id}/predictions` — Retrieve historical predictions for a patient (newest first).
- `GET /patients/{patient_id}/files` — Retrieve uploaded file history and metadata for a patient.

#### 6. Automated Testing
Run the complete integration and regression test suites:
```powershell
# Run backend & ML end-to-end integration test:
.venv\Scripts\python tests/test_full_integration.py

# Run comprehensive database, storage, and persistence tests:
.venv\Scripts\python tests/test_database.py

# Run presentation real-data demo:
.venv\Scripts\python scripts/demo_real_multimodal.py
```

---

### Modern Clinical Web Frontend (React + Vite + TypeScript)

The project includes a complete, presentation-ready web frontend in `frontend/`:

- **Framework**: React 18 with TypeScript and Vite.
- **Iconography**: Lucide React.
- **Zero Raw 12k Number Entry**: Supports uploading `.npy`/`.json` ECG files or 1-click loading of pre-validated real repository samples.
- **Real-Time Integration**: Direct communication with FastAPI (`http://127.0.0.1:8000`) and PostgreSQL persistence.

#### 1. Quick Start the Frontend
```powershell
# In a new terminal window:
cd frontend
npm install
npm run dev
```
The application will launch at `http://localhost:5173`.

#### 2. Environment Configuration
Frontend configuration is managed via `frontend/.env`:
```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```
An example template is available at `frontend/.env.example`.

#### 3. Frontend Features & Navigation
- **Multimodal Dashboard**:
  - Structured EHR input with real-time validation (Patient ID, Age 0–120, Gender M/F) and quick demo patient buttons.
  - Chest X-Ray upload with drag-and-drop, thumbnail preview, and quick selection of real `NORMAL` or `PNEUMONIA` test images from `archive/chest_xray/test/`.
  - 12-lead ECG sensor input supporting `.npy`/`.json` file upload or 1-click loading of real MIMIC-IV waveform samples from `features/ecg_waveforms/`.
  - Complete Results display showing:
    - Standalone CXR classification (NORMAL / PNEUMONIA) and probabilities from Member 1 ResNet-50.
    - Multimodal Cross-Attention assessment (Negative / Positive), confidence score, and uncertainty metric from Member 3 fusion.
    - PostgreSQL persistence record ID badge (e.g. `PostgreSQL Record #...`).
    - Input modality provenance summary.
- **Patient History & Audit Tab**:
  - Search any patient ID to inspect their registered record, total predictions count, and stored files count.
  - Interactive table of all past multimodal predictions with timestamps and confidence scores.
  - Interactive table of all uploaded files with safe server paths and file sizes.
- **Architecture & Limitations Tab**:
  - Detailed overview of the three branches (ResNet-50 visual features $F_{img} \in \mathbb{R}^{2048}$, tabular EHR $[1, 2]$, 12-lead $\times$ 1000 ECG temporal matrix).
  - Cross-attention fusion mechanism description.
  - Prominent mandatory clinical research disclaimers.

#### 4. Production Build
```powershell
cd frontend
npm run build
```
Generates optimized static assets in `frontend/dist/`.

---

### Critical Dataset Limitation & Clinical Safety

> **Important Clinical & Research Notice:**  
> 1. **Separate, Non-Matched Cohorts:** The Chest X-Ray dataset (Kaggle Pneumonia) and the MIMIC-IV EHR / ECG datasets are separate data sources and are **NOT** patient-matched. The current demonstration illustrates multimodal data alignment, feature projection, and cross-attentive fusion architecture, **not** patient-level clinical validation.  
> 2. **Research Prototype Only:** This system is an academic research prototype. Outputs are statistical model assessments, not definitive diagnoses.  
> 3. **Non-Diagnostic:** Under no circumstances should this system be used for clinical triage, medical diagnosis, or treatment decisions.

