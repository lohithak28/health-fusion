import React, { useState, useEffect } from 'react';
import { Search, AlertCircle, Clock, FolderOpen } from 'lucide-react';
import { api } from '../services/api';
import { PatientDetail, PredictionHistoryItem, UploadedFileItem } from '../types';

interface PatientHistoryProps {
  initialPatientId?: string;
}

export const PatientHistory: React.FC<PatientHistoryProps> = ({ initialPatientId = 'PATIENT_001' }) => {
  const [searchId, setSearchId] = useState(initialPatientId);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [patient, setPatient] = useState<PatientDetail | null>(null);
  const [predictions, setPredictions] = useState<PredictionHistoryItem[]>([]);
  const [files, setFiles] = useState<UploadedFileItem[]>([]);

  const fetchHistory = async (idToFetch: string) => {
    if (!idToFetch.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const [patientData, predsData, filesData] = await Promise.all([
        api.getPatient(idToFetch).catch(() => null),
        api.getPatientPredictions(idToFetch).catch(() => []),
        api.getPatientFiles(idToFetch).catch(() => []),
      ]);

      if (!patientData && predsData.length === 0 && filesData.length === 0) {
        setError(`No records found for Patient ID "${idToFetch}".`);
        setPatient(null);
        setPredictions([]);
        setFiles([]);
      } else {
        setPatient(patientData);
        setPredictions(predsData);
        setFiles(filesData);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch patient records');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (initialPatientId) {
      fetchHistory(initialPatientId);
    }
  }, [initialPatientId]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchHistory(searchId);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* Search Header */}
      <div className="card">
        <div className="card-title">
          <Search size={18} style={{ color: 'var(--primary)' }} />
          <span>Patient History</span>
        </div>

        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: '240px' }}>
            <input
              type="text"
              className="form-input"
              placeholder="Enter Patient ID (e.g. PATIENT_001)"
              value={searchId}
              onChange={(e) => setSearchId(e.target.value.trim().toUpperCase())}
            />
          </div>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            <span>{loading ? 'Searching...' : 'Search'}</span>
          </button>
        </form>

        <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.75rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Quick lookup:</span>
          {['PATIENT_001', 'PATIENT_002', 'PATIENT_REAL_DEMO'].map((id) => (
            <button
              key={id}
              type="button"
              className="btn btn-secondary"
              style={{ padding: '0.15rem 0.5rem', fontSize: '0.72rem' }}
              onClick={() => {
                setSearchId(id);
                fetchHistory(id);
              }}
            >
              {id}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div style={{
          background: '#fef2f2',
          border: '1px solid #fecaca',
          color: '#b91c1c',
          padding: '0.85rem 1rem',
          borderRadius: 'var(--radius-sm)',
          fontSize: '0.85rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
        }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Patient Overview */}
      {patient && (
        <div className="card" style={{ background: '#f8fafc' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <span className="badge badge-primary" style={{ marginBottom: '0.35rem' }}>Active Record</span>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Patient: {patient.patient_id}
              </h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                {patient.age} years old • {patient.gender === 'M' ? 'Male' : 'Female'}
              </p>
            </div>

            <div style={{ display: 'flex', gap: '1.5rem', textAlign: 'right' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Total Analyses</span>
                <span style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--primary)' }}>
                  {patient.total_predictions}
                </span>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Uploaded Files</span>
                <span style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                  {patient.total_files}
                </span>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block' }}>Registered</span>
                <span style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-secondary)' }}>
                  {new Date(patient.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Past Analyses Table */}
      <div className="card">
        <div className="card-title">
          <Clock size={18} style={{ color: 'var(--primary)' }} />
          <span>Past Analysis Records ({predictions.length})</span>
        </div>

        {predictions.length === 0 ? (
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', padding: '0.75rem 0' }}>
            No analyses recorded yet for this patient.
          </p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>Record #</th>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>Date & Time</th>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>Chest X-Ray Prediction</th>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>Multimodal Model Output</th>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>Confidence</th>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>Uncertainty Proxy</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p) => (
                  <tr key={p.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>#{p.id}</td>
                    <td style={{ padding: '0.65rem 0.5rem', color: 'var(--text-secondary)' }}>
                      {new Date(p.created_at).toLocaleString()}
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem' }}>
                      <span className={`badge ${p.cxr_prediction === 'PNEUMONIA' ? 'badge-danger' : 'badge-success'}`}>
                        {p.cxr_prediction || 'N/A'} {typeof p.cxr_probability === 'number' ? `(${(p.cxr_probability * 100).toFixed(0)}%)` : ''}
                      </span>
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>
                      {p.multimodal_prediction}
                    </td>
                    {(() => {
                      const isPos = p.multimodal_prediction.toLowerCase().includes('positive') || p.multimodal_prediction.toLowerCase().includes('high');
                      const conf = isPos ? p.multimodal_probability : 1.0 - p.multimodal_probability;
                      const unc = 1.0 - conf;
                      return (
                        <>
                          <td style={{ padding: '0.65rem 0.5rem' }}>
                            {(conf * 100).toFixed(1)}%
                          </td>
                          <td style={{ padding: '0.65rem 0.5rem', color: 'var(--text-muted)' }}>
                            {(unc * 100).toFixed(1)}%
                          </td>
                        </>
                      );
                    })()}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Uploaded Files Table */}
      <div className="card">
        <div className="card-title">
          <FolderOpen size={18} style={{ color: 'var(--primary)' }} />
          <span>Uploaded Clinical Files ({files.length})</span>
        </div>

        {files.length === 0 ? (
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', padding: '0.75rem 0' }}>
            No files recorded for this patient.
          </p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>ID</th>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>File Type</th>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>File Name</th>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>Size</th>
                  <th style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>Uploaded</th>
                </tr>
              </thead>
              <tbody>
                {files.map((f) => (
                  <tr key={f.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '0.65rem 0.5rem', fontWeight: 600 }}>#{f.id}</td>
                    <td style={{ padding: '0.65rem 0.5rem' }}>
                      <span className="badge badge-primary">{f.file_type.toUpperCase()}</span>
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem', fontWeight: 500 }}>{f.original_filename}</td>
                    <td style={{ padding: '0.65rem 0.5rem', color: 'var(--text-muted)' }}>
                      {f.file_size ? `${(f.file_size / 1024).toFixed(1)} KB` : 'N/A'}
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem', color: 'var(--text-secondary)' }}>
                      {new Date(f.uploaded_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
};
