import React from 'react';
import {
  Activity,
  CheckCircle2,
  Layers,
  Database,
  Info,
  ShieldAlert,
  Cpu,
} from 'lucide-react';
import { PatientData, PredictionResult } from '../types';

interface ResultDisplayProps {
  result: PredictionResult;
  patientData: PatientData;
  cxrFileName?: string;
  ecgSummary?: string;
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({
  result,
  patientData,
  cxrFileName,
  ecgSummary,
}) => {
  const isCxrPneumonia = result.cxr_prediction === 'PNEUMONIA';
  const cxrProb = result.cxr_probability !== null && result.cxr_probability !== undefined
    ? (result.cxr_probability * 100).toFixed(1)
    : null;

  const multiProb = (result.multimodal_probability * 100).toFixed(1);
  const multiUncertainty = (result.multimodal_uncertainty * 100).toFixed(1);
  const isMultiPositive =
    result.multimodal_prediction.toLowerCase().includes('positive') ||
    result.multimodal_prediction.toLowerCase().includes('high');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '1.5rem' }}>
      
      {/* Top Banner: Success & Persistence Badge */}
      <div style={{
        background: 'linear-gradient(90deg, #f0fdf4 0%, #ffffff 100%)',
        border: '1px solid var(--success-border)',
        borderRadius: 'var(--radius-md)',
        padding: '1rem 1.25rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '0.75rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <CheckCircle2 size={22} style={{ color: 'var(--success)' }} />
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Analysis Completed Successfully
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Patient {patientData.patient_id} • Age {patientData.age} • Gender {patientData.gender}
            </p>
          </div>
        </div>

        {result.prediction_id && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span className="badge badge-success" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <Database size={13} />
              <span>PostgreSQL Record #{result.prediction_id}</span>
            </span>
          </div>
        )}
      </div>

      {/* Main Results Grid */}
      <div className="grid-2">
        
        {/* Card 1: Standalone CXR ResNet-50 */}
        <div className="card" style={{ borderLeft: `4px solid ${isCxrPneumonia ? 'var(--danger)' : 'var(--success)'}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <div className="card-title" style={{ margin: 0 }}>
              <Cpu size={18} style={{ color: 'var(--primary)' }} />
              <span>Chest X-Ray — ResNet-50</span>
            </div>
            <span className={`badge ${isCxrPneumonia ? 'badge-danger' : 'badge-success'}`}>
              {result.cxr_prediction || 'PROCESSED'}
            </span>
          </div>

          <div style={{ marginBottom: '0.85rem' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)', display: 'block' }}>
              CXR Classification: Normal / Pneumonia
            </span>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.2rem', lineHeight: 1.4 }}>
              Standalone image-only binary classification (Normal vs Pneumonia). This task does not screen for other pulmonary conditions and is evaluated independently from multimodal fusion.
            </p>
          </div>

          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.3rem' }}>
              <span style={{ fontWeight: 500, color: 'var(--text-secondary)' }}>Confidence</span>
              <span style={{ fontWeight: 700, color: isCxrPneumonia ? 'var(--danger)' : 'var(--success)' }}>
                {cxrProb}%
              </span>
            </div>
            <div style={{ width: '100%', height: '8px', background: '#f1f5f9', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{
                width: `${cxrProb || 0}%`,
                height: '100%',
                background: isCxrPneumonia ? 'var(--danger)' : 'var(--success)',
                transition: 'width 0.4s ease',
              }} />
            </div>
          </div>

          {result.cxr_prob_normal !== undefined && result.cxr_prob_pneumonia !== undefined && (
            <div style={{
              background: '#f8fafc',
              borderRadius: 'var(--radius-sm)',
              padding: '0.75rem',
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '0.5rem',
              fontSize: '0.8rem',
              marginBottom: '1rem',
            }}>
              <div>
                <span style={{ color: 'var(--text-muted)', display: 'block' }}>Normal Probability</span>
                <span style={{ fontWeight: 600 }}>{((result.cxr_prob_normal ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)', display: 'block' }}>Pneumonia Probability</span>
                <span style={{ fontWeight: 600 }}>{((result.cxr_prob_pneumonia ?? 0) * 100).toFixed(1)}%</span>
              </div>
            </div>
          )}

          <div style={{
            fontSize: '0.75rem',
            color: 'var(--text-muted)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: '#ffffff',
            padding: '0.5rem 0.65rem',
            border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-sm)'
          }}>
            <Layers size={14} style={{ color: 'var(--primary)', flexShrink: 0 }} />
            <span>Extracted visual feature tensor <strong>F_img [1, 2048]</strong> fed into downstream fusion.</span>
          </div>
        </div>

        {/* Card 2: Multimodal Cross-Attention Fusion */}
        <div className="card" style={{ borderLeft: `4px solid ${isMultiPositive ? 'var(--warning)' : 'var(--primary)'}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <div className="card-title" style={{ margin: 0 }}>
              <Layers size={18} style={{ color: 'var(--primary)' }} />
              <span>Multimodal Cross-Attention Fusion</span>
            </div>
            <span className={`badge ${isMultiPositive ? 'badge-warning' : 'badge-primary'}`}>
              {result.multimodal_prediction}
            </span>
          </div>

          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: 1.4 }}>
            Separate multimodal fusion task integrating CXR visual features (2048-d) + EHR demographics (2-d) + 12-lead ECG sensor stream (12×1000).
          </p>

          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.3rem' }}>
              <span style={{ fontWeight: 500, color: 'var(--text-secondary)' }}>Multimodal Probability</span>
              <span style={{ fontWeight: 700, color: 'var(--primary)' }}>
                {multiProb}%
              </span>
            </div>
            <div style={{ width: '100%', height: '8px', background: '#f1f5f9', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{
                width: `${multiProb}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #0284c7 0%, #0369a1 100%)',
                transition: 'width 0.4s ease',
              }} />
            </div>
          </div>

          <div style={{
            background: '#f8fafc',
            borderRadius: 'var(--radius-sm)',
            padding: '0.75rem',
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '0.5rem',
            fontSize: '0.8rem',
            marginBottom: '1rem',
          }}>
            <div>
              <span style={{ color: 'var(--text-muted)', display: 'block' }}>Multimodal Model Output</span>
              <span style={{ fontWeight: 600, color: isMultiPositive ? 'var(--warning)' : 'var(--text-primary)' }}>
                {result.multimodal_prediction}
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)', display: 'block' }}>Uncertainty Proxy</span>
              <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }} title="Calculated as 1 - probability; not calibrated clinical uncertainty">
                {multiUncertainty}%
              </span>
            </div>
          </div>

          <div style={{
            fontSize: '0.75rem',
            color: 'var(--text-muted)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: '#ffffff',
            padding: '0.5rem 0.65rem',
            border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-sm)'
          }}>
            <Activity size={14} style={{ color: 'var(--primary)', flexShrink: 0 }} />
            <span>Three-way cross-attention weights computed across visual, tabular, and temporal modalities.</span>
          </div>
        </div>

      </div>

      {/* Input Artifact Summary & Storage Info */}
      <div className="card">
        <h4 style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Info size={16} style={{ color: 'var(--primary)' }} />
          Input Modality Provenance & Storage
        </h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem', fontSize: '0.82rem' }}>
          <div style={{ background: '#f8fafc', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>CXR Input Source</span>
            <span style={{ fontWeight: 500 }}>{cxrFileName || 'Uploaded Chest X-Ray'}</span>
          </div>
          <div style={{ background: '#f8fafc', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>ECG Sensor Input</span>
            <span style={{ fontWeight: 500 }}>{ecgSummary || '12 Leads × 1000 Samples'}</span>
          </div>
          <div style={{ background: '#f8fafc', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>EHR Demographics</span>
            <span style={{ fontWeight: 500 }}>Age: {patientData.age}, Gender: {patientData.gender}</span>
          </div>
          <div style={{ background: '#f8fafc', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>Storage Layer</span>
            <span style={{ fontWeight: 500 }}>uploads/ directory & PostgreSQL</span>
          </div>
        </div>
      </div>

      {/* Prominent Clinical Disclaimer */}
      <div style={{
        background: '#fffbeb',
        border: '1px solid #fde68a',
        borderRadius: 'var(--radius-md)',
        padding: '1rem',
        display: 'flex',
        gap: '0.85rem',
      }}>
        <ShieldAlert size={24} style={{ color: '#d97706', flexShrink: 0, marginTop: '0.1rem' }} />
        <div style={{ fontSize: '0.82rem', color: '#92400e', lineHeight: 1.5 }}>
          <strong>Clinical Research Prototype Notice:</strong>
          <p style={{ marginTop: '0.2rem' }}>
            This system is an academic proof-of-concept demonstrating cross-attention fusion across chest radiographs, structured EHR parameters, and 12-lead electrocardiograms.
            The CXR and MIMIC-IV EHR/ECG datasets are separate cohorts and are not patient-matched, so the result does not represent patient-level clinical validation. This output reflects a real-data multimodal architecture/integration demonstration.
            <strong> Do not use these predictions for medical diagnosis or clinical treatment planning.</strong>
          </p>
        </div>
      </div>

    </div>
  );
};
