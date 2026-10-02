# HealthFusion-Transformer: EHR + ECG Branch Report

## 1. Objective

The objective of this branch is to process electronic health record (EHR) data and electrocardiogram (ECG) signals, integrate the two modalities, and investigate their use for predicting in-hospital mortality using deep learning.

## 2. Data preprocessing

### EHR preprocessing
Patient demographics, admissions, diagnoses, and ICU stay information were extracted from the MIMIC-IV Clinical Database Demo. Patient-level features were created and saved to `features/ehr_patient_features.csv`.

The processed EHR dataset contains 100 patients and includes demographic information and admission counts.

### ECG preprocessing
ECG recordings were loaded using WFDB, resampled to 100 Hz, and standardized to 10 seconds with 12 leads, producing arrays of shape `(12, 1000)` per recording.

A total of 659 recordings were processed successfully. Non-finite signal values were handled during preprocessing. The resulting waveforms were saved to `features/ecg_waveforms/ecg_waveforms.npy`, with an associated index CSV.

### EHR–ECG integration
ECG metadata was linked to EHR features using patient identifiers. All 659 ECG recordings were associated with EHR features, representing 92 unique patients. This integration is patient-level metadata linkage and does not establish precise clinical time alignment.

## 3. Outcome labels and data splitting

ECG timestamps were matched to hospital admission intervals. The preliminary labeling procedure selected the earliest eligible ECG for each matched admission and used the admission's `hospital_expire_flag` as the outcome.

This produced 149 admission-level examples from 81 patients: 136 non-death outcomes and 13 in-hospital deaths.

Data were divided into training, validation, and test sets by patient to prevent the same patient from appearing in multiple splits:

- Training: 94 admissions from 56 patients
- Validation: 24 admissions from 12 patients
- Test: 31 admissions from 13 patients

The test set contained only four deaths, limiting the certainty of the evaluation.

## 4. Models

Three learned models were evaluated:

- **EHR-only baseline:** A multilayer perceptron using age and gender.
- **ECG-only baseline:** A convolutional neural network that extracts features from the 12-lead ECG waveform.
- **EHR–ECG fusion model:** A convolutional/Transformer ECG encoder, an EHR encoder, and cross-attention to combine ECG and EHR representations.

A majority-class baseline was also included for comparison.

Models were trained using weighted binary cross-entropy to address class imbalance. AdamW was used for optimization, and validation loss was used for checkpoint selection and early stopping where implemented.

## 5. Test results

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1-score | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.8710 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| EHR-only | 0.1290 | 0.5000 | 0.1290 | 1.0000 | 0.2286 | 0.7454 |
| ECG-only | 0.6129 | 0.5648 | 0.1667 | 0.5000 | 0.2500 | 0.5370 |
| EHR–ECG fusion | 0.8064 | 0.4630 | 0.0000 | 0.0000 | 0.0000 | 0.4630 |

Threshold-based metrics use a classification threshold of 0.5.

The ECG-only model detected two of the four deaths in the test set. The EHR-only model predicted every test example as a death, while the fusion model did not identify any of the four deaths at the selected threshold. The fusion model therefore did not demonstrate improved held-out test performance over the baselines in this experiment.

The EHR-only AUROC and its threshold-based results measure different aspects of performance: the model's score ranking was not equivalent to useful classifications at the selected threshold.

## 6. Limitations

1. The cohort is small, with only 13 positive examples overall and four in the test set.
2. Performance estimates are consequently unstable and should not be generalized to other populations.
3. The mortality labels are admission-level outcomes, not ECG diagnostic labels.
4. Patient-level linkage does not guarantee that all EHR features were available at the ECG timestamp.
5. Selecting the earliest eligible ECG does not by itself establish a clinically useful early-prediction task.
6. The current experiment does not establish clinical utility or readiness for clinical use.

## 7. Conclusion and future work

The EHR and ECG preprocessing, integration, dataset construction, baseline training, and cross-attention fusion experiments were completed. However, the current held-out results do not demonstrate that the fusion model improves mortality prediction.

Future work should examine prediction-time feature availability, refine the clinical prediction task, evaluate threshold selection and calibration, and validate the models on a larger cohort with more outcome-positive cases. The EHR–ECG branch can then be integrated with the imaging branch for multimodal evaluation, using the same patient-level split strategy where feasible.
