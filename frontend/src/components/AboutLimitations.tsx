import React from 'react';
import { ShieldAlert, Image, HeartPulse, User, Layers } from 'lucide-react';

export const AboutLimitations: React.FC = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* Overview Card */}
      <div className="card">
        <div className="card-title">
          <Layers size={18} style={{ color: 'var(--primary)' }} />
          <span>About HealthFusion</span>
        </div>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          HealthFusion is a clinical decision-support demonstration system designed to analyze and combine three types of patient data: chest radiographs, demographic records, and electrocardiogram (ECG) heart recordings.
        </p>
      </div>

      {/* The 3 Modalities */}
      <div className="grid-3">
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <Image size={18} style={{ color: 'var(--primary)' }} />
            <h4 style={{ fontSize: '0.92rem', fontWeight: 600 }}>1. Chest X-Ray</h4>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Ingests chest radiograph images and performs binary image classification (Normal vs Pneumonia).
          </p>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <User size={18} style={{ color: 'var(--primary)' }} />
            <h4 style={{ fontSize: '0.92rem', fontWeight: 600 }}>2. Patient Information</h4>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Considers patient demographic variables including age and gender.
          </p>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <HeartPulse size={18} style={{ color: 'var(--primary)' }} />
            <h4 style={{ fontSize: '0.92rem', fontWeight: 600 }}>3. ECG Data</h4>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Evaluates continuous multi-lead electrocardiogram waveforms to assess cardiac electrical activity.
          </p>
        </div>
      </div>

      {/* Important Limitations and Disclaimers */}
      <div style={{
        background: '#fffbeb',
        border: '1px solid #fde68a',
        borderRadius: 'var(--radius-md)',
        padding: '1.25rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.6rem' }}>
          <ShieldAlert size={22} style={{ color: '#d97706' }} />
          <h3 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#92400e' }}>
            Important Clinical Safety Notice & Limitations
          </h3>
        </div>

        <div style={{ fontSize: '0.83rem', color: '#78350f', lineHeight: 1.6, display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
          <p>
            <strong>Medical Safety Disclaimer:</strong> For demonstration and decision-support purposes only. This prototype is not a medical diagnosis and should not be used to guide patient triage, prescription, or clinical treatment decisions.
          </p>
          <p>
            <strong>Dataset Limitation:</strong> The chest X-ray and EHR/ECG data come from separate datasets and are not matched to the same patients. This prototype demonstrates system integration rather than patient-level clinical validation.
          </p>
        </div>
      </div>

    </div>
  );
};
