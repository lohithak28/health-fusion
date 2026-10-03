import React, { useState, useEffect } from 'react';
import { Play, RotateCcw, AlertCircle, Loader2 } from 'lucide-react';
import { Header } from './components/Header';
import { PatientForm } from './components/PatientForm';
import { CxrUpload } from './components/CxrUpload';
import { EcgInput } from './components/EcgInput';
import { ResultDisplay } from './components/ResultDisplay';
import { PatientHistory } from './components/PatientHistory';
import { AboutLimitations } from './components/AboutLimitations';
import { api } from './services/api';
import {
  DemoSamplesResponse,
  PatientData,
  PredictionResult,
} from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'history' | 'about'>('dashboard');

  // Form State
  const [patientData, setPatientData] = useState<PatientData>({
    patient_id: 'PATIENT_001',
    age: 54,
    gender: 'M',
  });

  const [cxrFile, setCxrFile] = useState<File | null>(null);
  const [selectedSamplePath, setSelectedSamplePath] = useState<string | null>(null);

  const [ecgFile, setEcgFile] = useState<File | null>(null);
  const [sampleEcgIndex, setSampleEcgIndex] = useState<number | null>(0);

  // Demo Samples from Repository
  const [demoSamples, setDemoSamples] = useState<DemoSamplesResponse>({
    sample_cxr_images: [],
    sample_ecg_available: false,
    sample_ecg_indices: [],
  });

  // Inference State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  // Fetch demo samples from backend on mount
  useEffect(() => {
    api.getDemoSamples()
      .then((data) => {
        setDemoSamples(data);
        // Pre-select first sample CXR and ECG if available
        if (data.sample_cxr_images.length > 0 && !cxrFile && !selectedSamplePath) {
          setSelectedSamplePath(data.sample_cxr_images[0].relative_path);
        }
        if (data.sample_ecg_indices.length > 0 && !ecgFile && sampleEcgIndex === null) {
          setSampleEcgIndex(data.sample_ecg_indices[0]);
        }
      })
      .catch((err) => {
        console.warn('Could not load repository demo samples:', err);
      });
  }, []);

  const handleReset = () => {
    setPatientData({ patient_id: '', age: '', gender: '' });
    setCxrFile(null);
    setSelectedSamplePath(null);
    setEcgFile(null);
    setSampleEcgIndex(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Validation
    if (!patientData.patient_id.trim()) {
      setError('Patient ID is required.');
      return;
    }
    if (patientData.age === '' || patientData.age < 0 || patientData.age > 120) {
      setError('A valid Age between 0 and 120 is required.');
      return;
    }
    if (!patientData.gender) {
      setError('Patient Gender (M or F) is required.');
      return;
    }
    if (!cxrFile && !selectedSamplePath) {
      setError('Chest X-Ray image is required (upload an image or select a sample).');
      return;
    }
    if (!ecgFile && sampleEcgIndex === null) {
      setError('ECG data is required (upload .npy/.json or select a sample).');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('patient_id', patientData.patient_id.trim());
      formData.append('age', String(patientData.age));
      formData.append('gender', patientData.gender);

      if (cxrFile) {
        formData.append('image', cxrFile);
      } else if (selectedSamplePath) {
        formData.append('sample_image_path', selectedSamplePath);
      }

      if (ecgFile) {
        formData.append('ecg_file', ecgFile);
      } else if (sampleEcgIndex !== null) {
        formData.append('sample_ecg_index', String(sampleEcgIndex));
      }

      const prediction = await api.analyzePatient(formData);
      setResult(prediction);
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="container" style={{ flex: 1 }}>
        {activeTab === 'dashboard' && (
          <div>
            <form onSubmit={handleAnalyze}>
              {/* Three Modality Input Cards */}
              <div className="grid-3">
                <PatientForm
                  data={patientData}
                  onChange={setPatientData}
                  disabled={loading}
                />

                <CxrUpload
                  file={cxrFile}
                  selectedSamplePath={selectedSamplePath}
                  onFileSelect={(f) => {
                    setCxrFile(f);
                    if (f) setSelectedSamplePath(null);
                  }}
                  onSampleSelect={(s) => {
                    setSelectedSamplePath(s ? s.relative_path : null);
                    if (s) setCxrFile(null);
                  }}
                  demoSamples={demoSamples.sample_cxr_images}
                  disabled={loading}
                />

                <EcgInput
                  file={ecgFile}
                  sampleIndex={sampleEcgIndex}
                  onFileSelect={(f) => {
                    setEcgFile(f);
                    if (f) setSampleEcgIndex(null);
                  }}
                  onSampleSelect={(idx) => {
                    setSampleEcgIndex(idx);
                    if (idx !== null) setEcgFile(null);
                  }}
                  availableIndices={demoSamples.sample_ecg_indices}
                  disabled={loading}
                />
              </div>

              {/* Action Buttons */}
              <div style={{
                marginTop: '1.5rem',
                display: 'flex',
                justifyContent: 'flex-end',
                alignItems: 'center',
                gap: '0.75rem',
              }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleReset}
                  disabled={loading}
                >
                  <RotateCcw size={15} />
                  <span>Reset Inputs</span>
                </button>

                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading}
                  style={{ minWidth: '180px' }}
                >
                  {loading ? (
                    <>
                      <Loader2 size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Play size={16} />
                      <span>Analyze</span>
                    </>
                  )}
                </button>
              </div>
            </form>

            {/* Error Message */}
            {error && (
              <div style={{
                marginTop: '1.25rem',
                background: '#fef2f2',
                border: '1px solid #fecaca',
                color: '#b91c1c',
                padding: '0.8rem 1rem',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.85rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
              }}>
                <AlertCircle size={17} />
                <span>{error}</span>
              </div>
            )}

            {/* Prediction Result Display */}
            {result && (
              <ResultDisplay
                result={result}
                patientData={patientData}
              />
            )}
          </div>
        )}

        {activeTab === 'history' && (
          <PatientHistory initialPatientId={patientData.patient_id || 'PATIENT_001'} />
        )}

        {activeTab === 'about' && (
          <AboutLimitations />
        )}
      </main>

      <footer style={{
        marginTop: 'auto',
        borderTop: '1px solid var(--border-color)',
        padding: '1.25rem 1.5rem',
        textAlign: 'center',
        fontSize: '0.78rem',
        color: 'var(--text-muted)',
        background: '#ffffff',
      }}>
        HealthFusion • Clinical Decision Support Demonstration Prototype
      </footer>
    </div>
  );
};
