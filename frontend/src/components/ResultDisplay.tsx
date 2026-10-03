import React from 'react';
import { PatientData, PredictionResult } from '../types';

interface ResultDisplayProps {
  result: PredictionResult;
  patientData: PatientData;
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({
  result,
  patientData,
}) => {
  const isCxrPneumonia = result.cxr_prediction === 'PNEUMONIA';
  const cxrProb = result.cxr_probability !== null && result.cxr_probability !== undefined
    ? (result.cxr_probability * 100).toFixed(1)
    : null;

  const isMultiPositive =
    result.multimodal_prediction.toLowerCase().includes('positive') ||
    result.multimodal_prediction.toLowerCase().includes('high');

  // Confidence is the probability of the predicted class
  const multiConfidenceRaw = isMultiPositive
    ? result.multimodal_probability
    : 1.0 - result.multimodal_probability;

  const multiProb = (multiConfidenceRaw * 100).toFixed(1);
  const multiUncertainty = ((1.0 - multiConfidenceRaw) * 100).toFixed(1);

  return (
    <div style={{ marginTop: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      
      {/* Main Analysis Result Card */}
      <div className="card" style={{ padding: '1.5rem' }}>
        
        {/* Card Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          borderBottom: '1px solid var(--border-color)',
          paddingBottom: '0.85rem',
          marginBottom: '1.25rem',
        }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Analysis Result
            </h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
              Patient {patientData.patient_id} • Age {patientData.age} • Gender {patientData.gender === 'M' ? 'Male' : 'Female'}
            </p>
          </div>

          {result.prediction_id && (
            <span style={{
              fontSize: '0.76rem',
              color: 'var(--text-secondary)',
              background: '#f1f5f9',
              padding: '0.25rem 0.6rem',
              borderRadius: '6px',
              fontWeight: 500,
            }}>
              Record #{result.prediction_id}
            </span>
          )}
        </div>

        {/* Two Clean Result Columns */}
        <div className="grid-2">
          
          {/* Section 1: Chest X-Ray Prediction */}
          <div style={{
            background: '#f8fafc',
            border: '1px solid var(--border-color)',
            borderLeft: `4px solid ${isCxrPneumonia ? 'var(--danger)' : 'var(--success)'}`,
            borderRadius: 'var(--radius-sm)',
            padding: '1.15rem',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                Chest X-Ray
              </span>
              <span className={`badge ${isCxrPneumonia ? 'badge-danger' : 'badge-success'}`}>
                {result.cxr_prediction === 'PNEUMONIA' ? 'Pneumonia' : 'Normal'}
              </span>
            </div>

            <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
              Chest X-Ray Prediction: {result.cxr_prediction === 'PNEUMONIA' ? 'Pneumonia' : 'Normal'}
            </div>

            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.85rem' }}>
              CXR Classification: Normal / Pneumonia
            </p>

            {cxrProb && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.3rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Confidence</span>
                  <span style={{ fontWeight: 600, color: isCxrPneumonia ? 'var(--danger)' : 'var(--success)' }}>
                    {cxrProb}%
                  </span>
                </div>
                <div style={{ width: '100%', height: '6px', background: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${cxrProb}%`,
                    height: '100%',
                    background: isCxrPneumonia ? 'var(--danger)' : 'var(--success)',
                  }} />
                </div>
              </div>
            )}
          </div>

          {/* Section 2: Multimodal Model Output */}
          <div style={{
            background: '#f8fafc',
            border: '1px solid var(--border-color)',
            borderLeft: `4px solid ${isMultiPositive ? 'var(--warning)' : 'var(--primary)'}`,
            borderRadius: 'var(--radius-sm)',
            padding: '1.15rem',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                Multimodal Analysis
              </span>
              <span className={`badge ${isMultiPositive ? 'badge-warning' : 'badge-primary'}`}>
                {result.multimodal_prediction}
              </span>
            </div>

            <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
              Multimodal Model Output: {result.multimodal_prediction}
            </div>

            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.85rem' }}>
              Combined decision-support assessment
            </p>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.3rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Confidence</span>
                <span style={{ fontWeight: 600, color: 'var(--primary)' }}>
                  {multiProb}%
                </span>
              </div>
              <div style={{ width: '100%', height: '6px', background: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{
                  width: `${multiProb}%`,
                  height: '100%',
                  background: 'var(--primary)',
                }} />
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginTop: '0.75rem', color: 'var(--text-muted)' }}>
              <span>Uncertainty Proxy</span>
              <span style={{ fontWeight: 500 }}>{multiUncertainty}%</span>
            </div>
          </div>

        </div>

        {/* Small subtle disclaimer */}
        <div style={{
          marginTop: '1.25rem',
          paddingTop: '0.85rem',
          borderTop: '1px solid var(--border-color)',
          textAlign: 'center',
          fontSize: '0.76rem',
          color: 'var(--text-muted)',
        }}>
          For demonstration and decision-support purposes only. This prototype is not a medical diagnosis.
        </div>

      </div>

    </div>
  );
};
