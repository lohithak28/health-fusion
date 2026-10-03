import { Layers, Database, ShieldAlert, Cpu, HeartPulse } from 'lucide-react';

export const AboutLimitations: React.FC = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* Overview Card */}
      <div className="card">
        <div className="card-title">
          <Layers size={20} style={{ color: 'var(--primary)' }} />
          <span>HealthFusion-Transformer Architecture</span>
        </div>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          HealthFusion-Transformer is an end-to-end multimodal clinical decision support research system. It integrates three heterogeneous data modalities—high-resolution chest radiographs, structured electronic health records, and continuous multi-lead electrocardiogram waveforms—via cross-attentive neural representations.
        </p>
      </div>

      {/* The 3 Modalities + Fusion Architecture */}
      <div className="grid-3">
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <Cpu size={20} style={{ color: 'var(--primary)' }} />
            <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>1. Imaging Modality</h4>
          </div>
          <span className="badge badge-primary" style={{ marginBottom: '0.75rem' }}>Chest X-Ray — ResNet-50</span>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Ingests Chest X-Rays (CXR), executes ImageNet-initialized ResNet-50 deep feature extraction, and produces:
          </p>
          <ul style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem', paddingLeft: '1.2rem', lineHeight: 1.5 }}>
            <li>Standalone CXR prediction (NORMAL / PNEUMONIA)</li>
            <li>Dense visual feature tensor <strong>F_img ∈ ℝ²⁰⁴⁸</strong></li>
          </ul>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <HeartPulse size={20} style={{ color: 'var(--primary)' }} />
            <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>2. EHR & Sensor Signals</h4>
          </div>
          <span className="badge badge-primary" style={{ marginBottom: '0.75rem' }}>EHR + 12-Lead ECG</span>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Normalizes patient tabular demographic information and continuous sensor streams:
          </p>
          <ul style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem', paddingLeft: '1.2rem', lineHeight: 1.5 }}>
            <li>EHR vector [Age normalized ∈ [0, 1], Gender binary ∈ {`{0, 1}`}]</li>
            <li>ECG matrix exactly 12 leads × 1000 temporal samples</li>
          </ul>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <Layers size={20} style={{ color: 'var(--primary)' }} />
            <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>3. Cross-Attention Fusion</h4>
          </div>
          <span className="badge badge-primary" style={{ marginBottom: '0.75rem' }}>Multimodal Cross-Attention Fusion</span>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Joint transformer-based cross-attention mechanism fusing visual, tabular, and temporal embeddings:
          </p>
          <ul style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem', paddingLeft: '1.2rem', lineHeight: 1.5 }}>
            <li>Projects heterogeneous representations into shared latent spaces</li>
            <li>Computes inter-modality attention weights</li>
            <li>Outputs probability score and uncertainty proxy (1 - probability)</li>
          </ul>
        </div>
      </div>

      {/* Persistence & Database Architecture */}
      <div className="card">
        <div className="card-title">
          <Database size={18} style={{ color: 'var(--primary)' }} />
          <span>PostgreSQL Persistence & File Provenance Layer</span>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '1rem' }}>
          All patient evaluations are fully audited and persisted in a relational PostgreSQL schema:
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem', fontSize: '0.82rem' }}>
          <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.25rem' }}>patients Table</strong>
            <span style={{ color: 'var(--text-muted)' }}>Unique patient identifier, age, gender, and creation timestamp. Automatically updated or fetched upon analysis.</span>
          </div>
          <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.25rem' }}>predictions Table</strong>
            <span style={{ color: 'var(--text-muted)' }}>Foreign-keyed to patient, storing CXR prediction & probability, multimodal fusion prediction, probability, uncertainty, and execution timestamp.</span>
          </div>
          <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.25rem' }}>uploaded_files Table</strong>
            <span style={{ color: 'var(--text-muted)' }}>Stores safe collision-free filenames, original filenames, MIME types, file sizes, and filesystem paths in the isolated uploads directory.</span>
          </div>
        </div>
      </div>

      {/* CRITICAL LIMITATIONS AND CLINICAL SAFETY DISCLAIMER */}
      <div style={{
        background: '#fffbeb',
        border: '2px solid #fde68a',
        borderRadius: 'var(--radius-md)',
        padding: '1.5rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.75rem' }}>
          <ShieldAlert size={24} style={{ color: '#d97706' }} />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#92400e' }}>
            Mandatory Clinical Safety Notice & Dataset Limitations
          </h3>
        </div>

        <div style={{ fontSize: '0.85rem', color: '#78350f', lineHeight: 1.6, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <p>
            <strong>1. Separate, Non-Matched Cohorts:</strong> The CXR and MIMIC-IV EHR/ECG datasets are separate cohorts and are not patient-matched, so the result does not represent patient-level clinical validation. This output reflects a real-data multimodal architecture/integration demonstration.
          </p>
          <p>
            <strong>2. Research Prototype Scope:</strong> This software is an engineering and machine learning prototype built solely for research, benchmarking, and demonstration purposes. Model assessments and confidence scores should never be interpreted as clinical diagnoses or indications of actual patient health.
          </p>
          <p>
            <strong>3. Regulatory & Diagnostic Exclusion:</strong> This system is not cleared or approved by the FDA, EMA, or any national health regulatory authority. It must NOT be used to guide patient triage, diagnosis, medication, surgery, or clinical decision-making.
          </p>
        </div>
      </div>

    </div>
  );
};
