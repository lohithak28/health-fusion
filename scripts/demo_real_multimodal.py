"""
HealthFusion-Transformer: Real Multimodal Pipeline Demonstration.

Demonstrates the multimodal pipeline on real project files:
- Real Chest X-Ray: archive/chest_xray/test/NORMAL/IM-0001-0001.jpeg
- Real EHR Record: features/labels/admission_labels.csv (MIMIC-IV clinical demo)
- Real 12-lead ECG: features/ecg_waveforms/ecg_waveforms.npy (MIMIC-IV-ECG demo)
- Model: features/models/member3_three_way_cross_attention.pt
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.imaging.inference import predict_chest_xray
from backend.services.patient_prediction_service import predict_patient


def run_demo():
    # 1. Real CXR image
    cxr_path = PROJECT_ROOT / "archive" / "chest_xray" / "test" / "NORMAL" / "IM-0001-0001.jpeg"
    if not cxr_path.exists():
        raise FileNotFoundError(f"CXR image not found: {cxr_path}")

    # 2. Member 1 CXR feature extraction & inference
    cxr_direct = predict_chest_xray(cxr_path)

    # 3. Real EHR clinical record from processed MIMIC data
    labels_path = PROJECT_ROOT / "features" / "labels" / "admission_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"EHR data file not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    first_record = labels_df.iloc[0]

    waveform_idx = int(first_record["waveform_index"])
    subject_id = str(first_record["subject_id"])
    age = float(first_record["anchor_age"])
    gender = str(first_record["gender"])

    # 4. Real ECG waveform from processed MIMIC-IV-ECG array
    ecg_path = PROJECT_ROOT / "features" / "ecg_waveforms" / "ecg_waveforms.npy"
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG array not found: {ecg_path}")

    ecg_all = np.load(ecg_path)
    ecg_waveform = ecg_all[waveform_idx]  # Shape: (12, 1000)

    # 5. Full Multimodal End-to-End Prediction
    result = predict_patient(
        image_path=cxr_path,
        age=age,
        gender=gender,
        ecg_waveform=ecg_waveform,
        patient_id=f"MIMIC_{subject_id}",
        save_features=False,
    )

    # Print presentation report
    print("=" * 50)
    print("HEALTHFUSION-TRANSFORMER REAL DATA DEMO")
    print("=" * 50)
    print("")
    print("CXR")
    print("----")
    print(f"Image: {cxr_path.name}")
    print(f"CXR prediction: {result['cxr_prediction']}")
    print(f"Normal probability: {result['cxr_prob_normal']:.4f}")
    print(f"Pneumonia probability: {result['cxr_prob_pneumonia']:.4f}")
    print(f"Feature shape: {list(cxr_direct['features'].shape)}")
    print("")
    print("EHR")
    print("---")
    print(f"Age: {int(age)}")
    print(f"Gender: {gender}")
    print("")
    print("ECG")
    print("---")
    print(f"Leads: {ecg_waveform.shape[0]}")
    print(f"Samples per lead: {ecg_waveform.shape[1]}")
    print(f"Waveform shape: {list(ecg_waveform.shape)}")
    print("")
    print("MULTIMODAL FUSION")
    print("------------------")
    print(f"Final prediction: {result['multimodal_prediction']}")
    print(f"Probability: {result['multimodal_probability']:.4f}")
    print("")
    print("IMPORTANT LIMITATION")
    print("--------------------")
    print("The CXR and MIMIC EHR/ECG records come from separate")
    print("datasets and are NOT patient-matched. This demonstration")
    print("shows the multimodal architecture and integration pipeline,")
    print("not patient-level clinical validation.")


if __name__ == "__main__":
    run_demo()
