import {
  DemoSamplesResponse,
  PatientDetail,
  PredictionHistoryItem,
  PredictionResult,
  UploadedFileItem,
} from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorMessage = `Request failed with status ${res.status}`;
    try {
      const errorData = await res.json();
      if (typeof errorData.detail === 'string') {
        errorMessage = errorData.detail;
      } else if (Array.isArray(errorData.detail)) {
        errorMessage = errorData.detail.map((e: any) => `${e.loc?.join('.')}: ${e.msg}`).join(', ');
      } else if (errorData.message) {
        errorMessage = errorData.message;
      }
    } catch {
      // Non-JSON error body
    }
    throw new ApiError(errorMessage, res.status);
  }
  return res.json();
}

export const api = {
  async getHealth(): Promise<{ status: string; service: string }> {
    const res = await fetch(`${BASE_URL}/health`);
    return handleResponse(res);
  },

  async getDemoSamples(): Promise<DemoSamplesResponse> {
    const res = await fetch(`${BASE_URL}/patients/demo/samples`);
    return handleResponse(res);
  },

  async analyzePatient(formData: FormData): Promise<PredictionResult> {
    const res = await fetch(`${BASE_URL}/patients/analyze`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<PredictionResult>(res);
  },

  async getPatient(patientId: string): Promise<PatientDetail> {
    const res = await fetch(`${BASE_URL}/patients/${encodeURIComponent(patientId)}`);
    return handleResponse<PatientDetail>(res);
  },

  async getPatientPredictions(patientId: string): Promise<PredictionHistoryItem[]> {
    const res = await fetch(`${BASE_URL}/patients/${encodeURIComponent(patientId)}/predictions`);
    return handleResponse<PredictionHistoryItem[]>(res);
  },

  async getPatientFiles(patientId: string): Promise<UploadedFileItem[]> {
    const res = await fetch(`${BASE_URL}/patients/${encodeURIComponent(patientId)}/files`);
    return handleResponse<UploadedFileItem[]>(res);
  },
};
