export interface PatientData {
  patient_id: string;
  age: number | '';
  gender: 'M' | 'F' | '';
}

export interface PredictionResult {
  patient_id: string;
  prediction_id?: number | null;
  cxr_prediction?: string | null;
  cxr_probability?: number | null;
  cxr_prob_normal?: number | null;
  cxr_prob_pneumonia?: number | null;
  multimodal_prediction: string;
  multimodal_probability: number;
  multimodal_uncertainty: number;
  feature_saved_to?: string | null;
  created_at?: string;
}

export interface UploadedFileItem {
  id: number;
  file_type: string;
  original_filename: string;
  stored_path: string;
  mime_type?: string | null;
  file_size?: number | null;
  uploaded_at: string;
  prediction_id?: number | null;
}

export interface PredictionHistoryItem {
  id: number;
  patient_id: string;
  cxr_prediction?: string | null;
  cxr_probability?: number | null;
  multimodal_prediction: string;
  multimodal_probability: number;
  uncertainty: number;
  created_at: string;
  files: UploadedFileItem[];
}

export interface PatientDetail {
  id: number;
  patient_id: string;
  age: number;
  gender: string;
  created_at: string;
  total_predictions: number;
  total_files: number;
}

export interface DemoSampleItem {
  name: string;
  category: string;
  relative_path: string;
}

export interface DemoSamplesResponse {
  sample_cxr_images: DemoSampleItem[];
  sample_ecg_available: boolean;
  sample_ecg_indices: number[];
}
