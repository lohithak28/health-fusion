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
        <span>ECG Data</span>
      </div>

      {/* Input Selection Area */}
      {!file && sampleIndex === null ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div
            onClick={() => !disabled && fileInputRef.current?.click()}
            style={{
              border: '2px dashed var(--border-color)',
              borderRadius: 'var(--radius-md)',
              padding: '1.5rem 1rem',
              textAlign: 'center',
              background: '#f8fafc',
              cursor: disabled ? 'not-allowed' : 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <Upload size={30} style={{ color: 'var(--primary)', marginBottom: '0.4rem' }} />
            <p style={{ fontWeight: 600, fontSize: '0.92rem', color: 'var(--text-primary)' }}>
              Upload ECG Recording
            </p>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
              Select ECG recording file (.npy, .json)
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
              <p style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                Or select a sample recording:
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                {availableIndices.map((idx, index) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSelectSample(idx)}
                    disabled={disabled}
                    className="btn btn-secondary"
                    style={{
                      padding: '0.35rem 0.65rem',
                      fontSize: '0.75rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                    }}
                  >
                    <HeartPulse size={13} style={{ color: 'var(--primary)' }} />
                    <span>Sample ECG {index + 1}</span>
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
          padding: '0.85rem',
          background: '#ffffff',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{
                background: 'var(--primary-light)',
                padding: '0.55rem',
                borderRadius: 'var(--radius-sm)',
                color: 'var(--primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                <FileCode size={20} />
              </div>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <CheckCircle size={15} style={{ color: 'var(--success)' }} />
                  <span style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                    {file
                      ? file.name
                      : `Sample ECG ${sampleIndex !== null && availableIndices.indexOf(sampleIndex) >= 0 ? availableIndices.indexOf(sampleIndex) + 1 : 1}`}
                  </span>
                </div>
                <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '0.1rem' }}>
                  Ready for analysis
                </p>
                <div style={{ marginTop: '0.25rem' }}>
                  <span className="badge badge-success" style={{ fontSize: '0.7rem', padding: '0.15rem 0.45rem' }}>
                    ECG Loaded
                  </span>
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
                width: '28px',
                height: '28px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              <X size={15} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
