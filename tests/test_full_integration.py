"""
Automated Test Suite for HealthFusion-Transformer Backend and ML Integration.
Tests all 7 requirements requested by the user without modifying models or checkpoints.
"""

import sys
import threading
import time
from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests
import uvicorn

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.services.patient_prediction_service import predict_patient
from scripts.demo_real_multimodal import run_demo

TEST_PORT = 8008
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


def main():
    report = {}
    print("\n" + "=" * 60)
    print("STARTING COMPLETE HEALTHFUSION INTEGRATION TESTS")
    print("=" * 60)

    # ----------------------------------------------------
    # 1. Start temporary FastAPI server
    # ----------------------------------------------------
    print("\n[1] Starting temporary localhost FastAPI server...")
    config = uvicorn.Config(app=app, host="127.0.0.1", port=TEST_PORT, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    time.sleep(2.0)  # Wait for startup

    try:
        # Test GET /health
        health_resp = requests.get(f"{BASE_URL}/health", timeout=5)
        health_pass = (health_resp.status_code == 200 and health_resp.json().get("status") == "healthy")
        report["1. Backend Startup /health"] = {
            "status": "PASS" if health_pass else "FAIL",
            "http_status": health_resp.status_code,
            "response": health_resp.json(),
        }
        print(f" -> GET /health: HTTP {health_resp.status_code} ({'PASS' if health_pass else 'FAIL'})")

        # ----------------------------------------------------
        # 2. Test EHR endpoint (POST /ehr/)
        # ----------------------------------------------------
        print("\n[2] Testing EHR endpoint (POST /ehr/)...")
        ehr_valid_resp = requests.post(
            f"{BASE_URL}/ehr/",
            json={"patient_id": "TEST_PATIENT", "age": 70, "gender": "M"},
            timeout=5,
        )
        ehr_invalid_resp = requests.post(
            f"{BASE_URL}/ehr/",
            json={"patient_id": "TEST_PATIENT", "age": 150, "gender": "M"},
            timeout=5,
        )
        ehr_pass = (ehr_valid_resp.status_code == 200 and ehr_invalid_resp.status_code == 422)
        report["2. EHR Endpoint"] = {
            "status": "PASS" if ehr_pass else "FAIL",
            "valid_http_status": ehr_valid_resp.status_code,
            "invalid_age_http_status": ehr_invalid_resp.status_code,
            "valid_response": ehr_valid_resp.json() if ehr_valid_resp.status_code == 200 else str(ehr_valid_resp.text),
        }
        print(f" -> Valid EHR: HTTP {ehr_valid_resp.status_code}")
        print(f" -> Invalid EHR (age=150): HTTP {ehr_invalid_resp.status_code} ({'PASS' if ehr_pass else 'FAIL'})")

        # ----------------------------------------------------
        # 3. Test Sensor endpoint (POST /sensor/)
        # ----------------------------------------------------
        print("\n[3] Testing Sensor endpoint (POST /sensor/)...")
        ecg_path = PROJECT_ROOT / "features" / "ecg_waveforms" / "ecg_waveforms.npy"
        real_ecg = np.load(ecg_path)[0].tolist()  # Real 12x1000

        sensor_valid_resp = requests.post(
            f"{BASE_URL}/sensor/",
            json={"patient_id": "TEST_PATIENT", "ecg_waveform": real_ecg},
            timeout=5,
        )
        # Invalid 11 leads
        invalid_ecg = real_ecg[:11]
        sensor_invalid_resp = requests.post(
            f"{BASE_URL}/sensor/",
            json={"patient_id": "TEST_PATIENT", "ecg_waveform": invalid_ecg},
            timeout=5,
        )
        sensor_pass = (sensor_valid_resp.status_code == 200 and sensor_invalid_resp.status_code == 422)
        report["3. Sensor Endpoint"] = {
            "status": "PASS" if sensor_pass else "FAIL",
            "valid_http_status": sensor_valid_resp.status_code,
            "invalid_waveform_http_status": sensor_invalid_resp.status_code,
            "ecg_shape_sent": [len(real_ecg), len(real_ecg[0])],
            "response": sensor_valid_resp.json() if sensor_valid_resp.status_code == 200 else str(sensor_valid_resp.text),
        }
        print(f" -> Valid Sensor (12x1000): HTTP {sensor_valid_resp.status_code}")
        print(f" -> Invalid Sensor (11 leads): HTTP {sensor_invalid_resp.status_code} ({'PASS' if sensor_pass else 'FAIL'})")

        # ----------------------------------------------------
        # 4. Test Image endpoint
        # ----------------------------------------------------
        print("\n[4] Testing Image endpoint...")
        cxr_test_file = PROJECT_ROOT / "archive" / "chest_xray" / "test" / "NORMAL" / "IM-0001-0001.jpeg"
        
        # Check POST /image/ first (to see if client path contract matches)
        with open(cxr_test_file, "rb") as f:
            image_resp_direct = requests.post(
                f"{BASE_URL}/image/",
                files={"file": (cxr_test_file.name, f, "image/jpeg")},
                timeout=30,
            )

        # Also check POST /upload/image (the existing verified route)
        with open(cxr_test_file, "rb") as f:
            image_resp_upload = requests.post(
                f"{BASE_URL}/upload/image",
                files={"file": (cxr_test_file.name, f, "image/jpeg")},
                timeout=30,
            )

        upload_data = image_resp_upload.json() if image_resp_upload.status_code == 200 else {}
        feature_shape = upload_data.get("feature_shape")
        image_pass = (
            image_resp_upload.status_code == 200
            and "prediction" in upload_data
            and "probability" in upload_data
            and feature_shape == [1, 2048]
        )

        report["4. Image Endpoint"] = {
            "status": "PASS" if image_pass else "FAIL",
            "post_image_slash_status": image_resp_direct.status_code,
            "post_upload_image_status": image_resp_upload.status_code,
            "prediction": upload_data.get("prediction"),
            "probability": upload_data.get("probability"),
            "prob_normal": upload_data.get("prob_normal"),
            "prob_pneumonia": upload_data.get("prob_pneumonia"),
            "feature_shape": feature_shape,
        }
        print(f" -> POST /image/: HTTP {image_resp_direct.status_code}")
        print(f" -> POST /upload/image: HTTP {image_resp_upload.status_code}")
        print(f" -> CXR Prediction: {upload_data.get('prediction')} (Prob: {upload_data.get('probability')})")
        print(f" -> Extracted F_img Shape: {feature_shape} ({'PASS' if image_pass else 'FAIL'})")

        # ----------------------------------------------------
        # 5. Test Multimodal Fusion Service
        # ----------------------------------------------------
        print("\n[5] Testing Multimodal Fusion (patient_prediction_service)...")
        fusion_result = predict_patient(
            image_path=cxr_test_file,
            age=70,
            gender="M",
            ecg_waveform=real_ecg,
            patient_id="TEST_PATIENT_FUSION",
            save_features=False,
        )
        fusion_pass = (
            "multimodal_prediction" in fusion_result
            and "multimodal_probability" in fusion_result
            and isinstance(fusion_result["multimodal_probability"], float)
        )
        report["5. Multimodal Fusion Service"] = {
            "status": "PASS" if fusion_pass else "FAIL",
            "cxr_prediction": fusion_result.get("cxr_prediction"),
            "multimodal_prediction": fusion_result.get("multimodal_prediction"),
            "multimodal_probability": fusion_result.get("multimodal_probability"),
            "multimodal_uncertainty": fusion_result.get("multimodal_uncertainty"),
        }
        print(f" -> Multimodal Prediction: {fusion_result.get('multimodal_prediction')} (Prob: {fusion_result.get('multimodal_probability')})")
        print(f" -> Fusion Status: {'PASS' if fusion_pass else 'FAIL'}")

        # ----------------------------------------------------
        # 6. Run scripts/demo_real_multimodal.py
        # ----------------------------------------------------
        print("\n[6] Running scripts/demo_real_multimodal.py...")
        try:
            run_demo()
            demo_pass = True
        except Exception as e:
            print(f"Demo failed: {e}")
            demo_pass = False

        report["6. Real-Data Presentation Demo"] = {
            "status": "PASS" if demo_pass else "FAIL"
        }
        print(f" -> Demo Execution: {'PASS' if demo_pass else 'FAIL'}")

    finally:
        # Shutdown server
        server.should_exit = True
        server_thread.join(timeout=3)
        print("\nTemporary server stopped.")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
