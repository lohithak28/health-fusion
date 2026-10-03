import React from 'react';
import { User } from 'lucide-react';
import { PatientData } from '../types';

interface PatientFormProps {
  data: PatientData;
  onChange: (data: PatientData) => void;
  disabled?: boolean;
}

export const PatientForm: React.FC<PatientFormProps> = ({ data, onChange, disabled }) => {
  const handleIdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...data, patient_id: e.target.value.trim().toUpperCase() });
  };

  const handleAgeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (val === '') {
      onChange({ ...data, age: '' });
    } else {
      const num = parseInt(val, 10);
      if (!isNaN(num)) {
        onChange({ ...data, age: num });
      }
    }
  };

  const handleGenderChange = (gender: 'M' | 'F') => {
    onChange({ ...data, gender });
  };

  const setQuickPatient = (id: string, age: number, gender: 'M' | 'F') => {
    onChange({ patient_id: id, age, gender });
  };

  return (
    <div className="card">
      <div className="card-title">
        <User size={18} style={{ color: 'var(--primary)' }} />
        <span>Electronic Health Record (EHR)</span>
      </div>

      <div className="form-group">
        <label className="form-label" htmlFor="patient-id">
          Patient Identifier <span style={{ color: 'var(--danger)' }}>*</span>
        </label>
        <div style={{ position: 'relative' }}>
          <input
            id="patient-id"
            className="form-input"
            type="text"
            placeholder="e.g. PATIENT_001"
            value={data.patient_id}
            onChange={handleIdChange}
            disabled={disabled}
            maxLength={64}
          />
        </div>
        <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.4rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Quick demo IDs:</span>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ padding: '0.15rem 0.45rem', fontSize: '0.72rem' }}
            onClick={() => setQuickPatient('PATIENT_001', 54, 'M')}
            disabled={disabled}
          >
            PATIENT_001 (54M)
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ padding: '0.15rem 0.45rem', fontSize: '0.72rem' }}
            onClick={() => setQuickPatient('PATIENT_002', 68, 'F')}
            disabled={disabled}
          >
            PATIENT_002 (68F)
          </button>
        </div>
      </div>

      <div className="grid-2">
        <div className="form-group">
          <label className="form-label" htmlFor="patient-age">
            Age (Years) <span style={{ color: 'var(--danger)' }}>*</span>
          </label>
          <input
            id="patient-age"
            className="form-input"
            type="number"
            min={0}
            max={120}
            placeholder="0 – 120"
            value={data.age}
            onChange={handleAgeChange}
            disabled={disabled}
          />
          {typeof data.age === 'number' && (data.age < 0 || data.age > 120) && (
            <span style={{ color: 'var(--danger)', fontSize: '0.75rem', marginTop: '0.2rem', display: 'block' }}>
              Age must be between 0 and 120
            </span>
          )}
        </div>

        <div className="form-group">
          <label className="form-label">
            Gender <span style={{ color: 'var(--danger)' }}>*</span>
          </label>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              onClick={() => handleGenderChange('M')}
              disabled={disabled}
              style={{
                flex: 1,
                padding: '0.6rem',
                borderRadius: 'var(--radius-sm)',
                border: data.gender === 'M' ? '2px solid var(--primary)' : '1px solid var(--border-color)',
                background: data.gender === 'M' ? 'var(--primary-light)' : '#ffffff',
                color: data.gender === 'M' ? 'var(--primary)' : 'var(--text-secondary)',
                fontWeight: data.gender === 'M' ? 600 : 400,
                fontSize: '0.9rem',
              }}
            >
              Male (M)
            </button>
            <button
              type="button"
              onClick={() => handleGenderChange('F')}
              disabled={disabled}
              style={{
                flex: 1,
                padding: '0.6rem',
                borderRadius: 'var(--radius-sm)',
                border: data.gender === 'F' ? '2px solid var(--primary)' : '1px solid var(--border-color)',
                background: data.gender === 'F' ? 'var(--primary-light)' : '#ffffff',
                color: data.gender === 'F' ? 'var(--primary)' : 'var(--text-secondary)',
                fontWeight: data.gender === 'F' ? 600 : 400,
                fontSize: '0.9rem',
              }}
            >
              Female (F)
            </button>
          </div>
        </div>
      </div>

      <div style={{
        marginTop: '0.5rem',
        padding: '0.6rem 0.8rem',
        background: '#f8fafc',
        borderRadius: 'var(--radius-sm)',
        fontSize: '0.78rem',
        color: 'var(--text-muted)',
        border: '1px dashed var(--border-color)'
      }}>
        EHR inputs are normalized into the EHR feature vector [1, 2] and integrated into the cross-attention fusion layer.
      </div>
    </div>
  );
};
