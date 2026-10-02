# Member 2 — EHR–ECG Multimodal Fusion

## Objective

Develop and evaluate a multimodal model that combines Electronic Health Record (EHR) features and 12-lead ECG signals for experimental in-hospital mortality prediction.

## Pipeline

- `ehr_preprocessing.py` — generates patient-level EHR features.
- `ecg_preprocessing.py` — preprocesses ECG waveforms.
- `dataset.py` — integrates EHR features with ECG metadata.
- `validate_target.py` — checks ECG timing against admission intervals.
- `prepare_labels.py` — creates admission-level mortality labels.
- `prepare_label_splits.py` — creates patient-level train, validation and test splits.
- `train_model.py` — trains the cross-attention EHR–ECG fusion model.
- `ehr_baseline.py` — trains the EHR-only baseline.
- `ecg_baseline.py` — trains the ECG-only baseline.
- `compare_models.py` — compares model performance.

## Inputs

- MIMIC-IV Clinical Database Demo.
- MIMIC-IV-ECG Demo matched subset.

## Main Outputs

- `features/ehr_patient_features.csv` — processed patient-level EHR features.
- `features/ecg_waveforms/` — preprocessed ECG waveforms and their index.
- `features/ehr_ecg_integrated.csv` — integrated EHR–ECG metadata.
- `features/labels/admission_labels.csv` — admission-level mortality labels.
- Patient-level train, validation and test split CSV files.
- Model checkpoints under `features/models/`.
- Model comparison tables and plots under `features/results/`.

## Model Architecture

The fusion model combines two modalities:

- **ECG encoder:** Uses convolutional layers and a Transformer encoder to learn representations from 12-lead ECG signals.
- **EHR encoder:** Encodes demographic features, including age and gender.
- **Cross-attention fusion:** Allows ECG representations to attend to the EHR representation.
- **Mortality classifier:** Uses the fused representation to produce a prediction logit for in-hospital mortality.

## Evaluation

The fusion model is evaluated against an EHR-only baseline, an ECG-only baseline and a majority-class baseline.

The evaluation metrics include:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- Area Under the Receiver Operating Characteristic Curve (AUROC)

The dataset contains a small number of mortality cases. The held-out test set contains only four deaths, so the reported metrics are uncertain and should be interpreted cautiously.

## Results Summary

The recorded test results did not demonstrate improved performance from the EHR–ECG fusion model compared with the baselines.

The fusion model achieved a test accuracy of approximately 80.65% and an AUROC of approximately 0.463. At the selected classification threshold of 0.5, it failed to identify any of the four deaths in the test set.

These findings indicate that further model development and validation are required.

## Limitations

- EHR–ECG linkage is based on patient identifiers and admission timing; comprehensive clinical time alignment has not been established.
- The dataset is small and contains relatively few positive mortality cases.
- Only four deaths are present in the held-out test set, making performance estimates unstable.
- The fusion model did not demonstrate improved held-out test performance.
- The current results do not establish clinical usefulness or generalizability.
- This project is an experimental research prototype, not a clinically validated decision-support system.

## Future Work

- Evaluate the models on a larger dataset with more mortality cases.
- Improve temporal alignment between ECG recordings and clinical events.
- Investigate class imbalance and classification thresholds using validation data.
- Perform additional experiments with fixed random seeds and reproducible configurations.
- Validate the models on an independent dataset before considering clinical applications.
- Investigate whether the EHR–ECG branch can be integrated with the imaging branch using a compatible prediction target and verified patient-level linkage.

## Conclusion

The Member 2 branch implements an EHR–ECG preprocessing and modelling pipeline, including data integration, mortality-label preparation, patient-level splitting, multimodal fusion, baseline comparisons and evaluation.

Although the pipeline and experiments have been completed, the fusion model did not demonstrate improved performance on the held-out test set. Further development, larger datasets and independent validation are needed before drawing conclusions about clinical effectiveness.
