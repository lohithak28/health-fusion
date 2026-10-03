import React, { useEffect, useState } from 'react';
import { Activity } from 'lucide-react';
import { api } from '../services/api';

interface HeaderProps {
  activeTab: 'dashboard' | 'history' | 'about';
  setActiveTab: (tab: 'dashboard' | 'history' | 'about') => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab }) => {
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    api.getHealth()
      .then(() => {
        if (mounted) setBackendHealthy(true);
      })
      .catch(() => {
        if (mounted) setBackendHealthy(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <header style={{ background: '#ffffff', borderBottom: '1px solid var(--border-color)', marginBottom: '1.75rem' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '1rem 1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          
          {/* Logo & Subtitle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div style={{
              background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
              color: '#ffffff',
              padding: '0.65rem',
              borderRadius: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 4px rgba(2, 132, 199, 0.2)'
            }}>
              <Activity size={24} strokeWidth={2.4} />
            </div>
            <div>
              <h1 style={{ fontSize: '1.35rem', fontWeight: '700', letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
                HealthFusion
              </h1>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Clinical Decision Support Demo
              </p>
            </div>
          </div>

          {/* Backend Status & Navigation */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.82rem' }}>
              <span style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: backendHealthy === true ? '#10b981' : backendHealthy === false ? '#ef4444' : '#f59e0b',
                display: 'inline-block'
              }} />
              <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
                {backendHealthy === true ? 'System Ready' : backendHealthy === false ? 'Service Offline' : 'Connecting...'}
              </span>
            </div>

            <nav style={{ display: 'flex', gap: '0.35rem', background: '#f1f5f9', padding: '0.25rem', borderRadius: '8px' }}>
              <button
                onClick={() => setActiveTab('dashboard')}
                style={{
                  padding: '0.4rem 0.85rem',
                  fontSize: '0.85rem',
                  fontWeight: activeTab === 'dashboard' ? 600 : 500,
                  border: 'none',
                  borderRadius: '6px',
                  background: activeTab === 'dashboard' ? '#ffffff' : 'transparent',
                  color: activeTab === 'dashboard' ? 'var(--primary)' : 'var(--text-secondary)',
                  boxShadow: activeTab === 'dashboard' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                }}
              >
                Analysis
              </button>
              <button
                onClick={() => setActiveTab('history')}
                style={{
                  padding: '0.4rem 0.85rem',
                  fontSize: '0.85rem',
                  fontWeight: activeTab === 'history' ? 600 : 500,
                  border: 'none',
                  borderRadius: '6px',
                  background: activeTab === 'history' ? '#ffffff' : 'transparent',
                  color: activeTab === 'history' ? 'var(--primary)' : 'var(--text-secondary)',
                  boxShadow: activeTab === 'history' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                }}
              >
                Patient History
              </button>
              <button
                onClick={() => setActiveTab('about')}
                style={{
                  padding: '0.4rem 0.85rem',
                  fontSize: '0.85rem',
                  fontWeight: activeTab === 'about' ? 600 : 500,
                  border: 'none',
                  borderRadius: '6px',
                  background: activeTab === 'about' ? '#ffffff' : 'transparent',
                  color: activeTab === 'about' ? 'var(--primary)' : 'var(--text-secondary)',
                  boxShadow: activeTab === 'about' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                }}
              >
                About & Limitations
              </button>
            </nav>
          </div>

        </div>
      </div>
    </header>
  );
};
