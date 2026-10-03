import React, { useRef, useState, useEffect } from 'react';
import { Image as ImageIcon, Upload, X, CheckCircle } from 'lucide-react';
import { DemoSampleItem } from '../types';

interface CxrUploadProps {
  file: File | null;
  selectedSamplePath: string | null;
  onFileSelect: (file: File | null) => void;
  onSampleSelect: (sample: DemoSampleItem | null) => void;
  demoSamples: DemoSampleItem[];
  disabled?: boolean;
}

export const CxrUpload: React.FC<CxrUploadProps> = ({
  file,
  selectedSamplePath,
  onFileSelect,
  onSampleSelect,
  demoSamples,
  disabled,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else if (selectedSamplePath) {
      setPreviewUrl(`${BASE_URL}/patients/demo/image?relative_path=${encodeURIComponent(selectedSamplePath)}`);
    } else {
      setPreviewUrl(null);
    }
  }, [file, selectedSamplePath, BASE_URL]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0];
      if (dropped.type.startsWith('image/')) {
        onSampleSelect(null);
        onFileSelect(dropped);
      }
    }
  };

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

  const handleSampleClick = (sample: DemoSampleItem) => {
    onFileSelect(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onSampleSelect(sample);
  };

  return (
    <div className="card">
      <div className="card-title">
        <ImageIcon size={18} style={{ color: 'var(--primary)' }} />
        <span>Chest X-Ray (CXR) Modality</span>
      </div>

      {/* Upload Zone */}
      {!file && !selectedSamplePath ? (
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
          style={{
            border: `2px dashed ${dragActive ? 'var(--primary)' : 'var(--border-color)'}`,
            borderRadius: 'var(--radius-md)',
            padding: '1.75rem 1rem',
            textAlign: 'center',
            background: dragActive ? 'var(--primary-light)' : '#f8fafc',
            cursor: disabled ? 'not-allowed' : 'pointer',
            transition: 'all 0.15s ease',
          }}
        >
          <Upload size={32} style={{ color: 'var(--primary)', marginBottom: '0.5rem' }} />
          <p style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text-primary)' }}>
            Upload Chest X-Ray Image
          </p>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Drag & drop or click to browse (.jpeg, .png)
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            style={{ display: 'none' }}
            onChange={handleInputChange}
            disabled={disabled}
          />
        </div>
      ) : (
        <div style={{
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-md)',
          padding: '1rem',
          background: '#ffffff',
        }}>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            {previewUrl && (
              <div style={{
                width: '90px',
                height: '90px',
                borderRadius: 'var(--radius-sm)',
                overflow: 'hidden',
                border: '1px solid var(--border-color)',
                flexShrink: 0,
                background: '#000',
              }}>
                <img
                  src={previewUrl}
                  alt="CXR Preview"
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
              </div>
            )}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.2rem' }}>
                <CheckCircle size={16} style={{ color: 'var(--success)' }} />
                <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)', wordBreak: 'break-all' }}>
                  {file ? file.name : selectedSamplePath?.split(/[/\\]/).pop()}
                </span>
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                {file
                  ? `Custom upload • ${(file.size / 1024).toFixed(1)} KB`
                  : `Repository sample CXR • Ready for inference`}
              </p>
              <div style={{ marginTop: '0.4rem' }}>
                <span className="badge badge-primary">ResNet-50 Target</span>
              </div>
            </div>
            <button
              type="button"
              onClick={handleClear}
              disabled={disabled}
              title="Remove image"
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

      {/* Demo Samples Selector */}
      {demoSamples.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
            Or select a pre-validated real test CXR:
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            {demoSamples.map((sample) => {
              const isSelected = selectedSamplePath === sample.relative_path;
              return (
                <button
                  key={sample.relative_path}
                  type="button"
                  onClick={() => handleSampleClick(sample)}
                  disabled={disabled}
                  style={{
                    padding: '0.35rem 0.65rem',
                    fontSize: '0.75rem',
                    borderRadius: 'var(--radius-sm)',
                    border: isSelected ? '1px solid var(--primary)' : '1px solid var(--border-color)',
                    background: isSelected ? 'var(--primary-light)' : '#ffffff',
                    color: isSelected ? 'var(--primary)' : 'var(--text-secondary)',
                    fontWeight: isSelected ? 600 : 400,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.3rem',
                  }}
                >
                  <span
                    style={{
                      width: '6px',
                      height: '6px',
                      borderRadius: '50%',
                      background: sample.category === 'NORMAL' ? 'var(--success)' : 'var(--danger)',
                    }}
                  />
                  <span>{sample.category}: {sample.name.slice(0, 15)}...</span>
                </button>
              );
            })}
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
        ResNet-50 extracts visual feature tensor F_img [1, 2048] and generates standalone CXR binary classification (Normal / Pneumonia) independently from multimodal cross-attention.
      </div>
    </div>
  );
};
