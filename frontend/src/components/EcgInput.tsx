import React, { useRef } from 'react';
import { HeartPulse, Upload, X, CheckCircle, FileCode } from 'lucide-react';

interface EcgInputProps {
  file: File | null;
  sampleIndex: number | null;
  onFileSelect: (file: File | null) => void;
  onSampleSelect: (idx: number | null) => void;
  availableIndices: number[];
  disabled?: boolean;
}

export const EcgInput: React.FC<EcgInputProps> = ({
  file,
  sampleIndex,
  onFileSelect,
  onSampleSelect,
  availableIndices,
  disabled,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onSampleSelect(null);
      onFileSelect(e.target.files[0]);
    }
  };

  const handleClear = () => {
    onFileSelect(null);
    onSampleSelect(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSelectSample = (idx: number) => {
    onFileSelect(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onSampleSelect(idx);
  };

  return (
    <div className="card">
      <div className="card-title">
        <HeartPulse size={18} style={{ color: 'var(--primary)' }} />
        <span>12-Lead ECG Sensor Waveform</span>
      </div>

      {/* Input Selection Area */}
      {!file && sampleIndex === null ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div
            onClick={() => !disabled && fileInputRef.current?.click()}
            style={{
              border: '2px dashed var(--border-color)',
              borderRadius: 'var(--radius-md)',
              padding: '1.25rem 1rem',
              textAlign: 'center',
              background: '#f8fafc',
              cursor: disabled ? 'not-allowed' : 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <Upload size={28} style={{ color: 'var(--primary)', marginBottom: '0.35rem' }} />
            <p style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
              Upload 12-Lead ECG File
            </p>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Accepts .npy or .json (must be exactly 12 leads × 1000 samples)
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".npy,.json"
              style={{ display: 'none' }}
              onChange={handleInputChange}
              disabled={disabled}
            />
          </div>

          {availableIndices.length > 0 && (
            <div>
              <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                Or select a verified real MIMIC-IV ECG waveform from the repository:
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {availableIndices.map((idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSelectSample(idx)}
                    disabled={disabled}
                    className="btn btn-secondary"
                    style={{
                      padding: '0.4rem 0.75rem',
                      fontSize: '0.78rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.35rem',
                    }}
                  >
                    <HeartPulse size={14} style={{ color: 'var(--primary)' }} />
                    <span>Real ECG #{idx} (12×1000)</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={{
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-md)',
          padding: '1rem',
          background: '#ffffff',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{
                background: 'var(--primary-light)',
                padding: '0.6rem',
                borderRadius: 'var(--radius-sm)',
                color: 'var(--primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                <FileCode size={22} />
              </div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <CheckCircle size={16} style={{ color: 'var(--success)' }} />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                    {file ? file.name : `Dataset ECG Waveform Sample #${sampleIndex}`}
                  </span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                  {file
                    ? `Uploaded custom ECG • ${(file.size / 1024).toFixed(1)} KB`
                    : `Loaded from features/ecg_waveforms/ecg_waveforms.npy`}
                </p>
                <div style={{ marginTop: '0.35rem', display: 'flex', gap: '0.4rem' }}>
                  <span className="badge badge-success">12 Leads Validated</span>
                  <span className="badge badge-primary">1000 Samples / Lead</span>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={handleClear}
              disabled={disabled}
              title="Remove ECG"
              style={{
                border: 'none',
                background: '#f1f5f9',
                borderRadius: '50%',
                width: '30px',
                height: '30px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      <div style={{
        marginTop: '0.8rem',
        padding: '0.6rem 0.8rem',
        background: '#f8fafc',
        borderRadius: 'var(--radius-sm)',
        fontSize: '0.78rem',
        color: 'var(--text-muted)',
        border: '1px dashed var(--border-color)'
      }}>
        ECG signals are validated strictly at 12 leads × 1000 temporal samples (12,000 float points) and passed into the cross-attention fusion network.
      </div>
    </div>
  );
};
