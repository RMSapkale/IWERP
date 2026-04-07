/**
 * Dashboard.jsx — IWERP Sovereign API Management Dashboard
 */
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Key, RotateCw, Shield, FlaskConical, Activity, Copy, Check, BarChart3, Globe } from 'lucide-react';
import Navbar from '../components/Navbar';
import { api, getUser, getTenant } from '../api';
import styles from './Dashboard.module.css';

export default function Dashboard() {
  const [keys, setKeys] = useState([]);
  const [rotating, setRotating] = useState(false);
  const [copied, setCopied] = useState(false);
  const [newRotateKey, setNewRotateKey] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const apiBase = import.meta.env.VITE_API_BASE_URL || 'https://iwerp.com/api';
  
  const user = getUser();
  const tenant = getTenant();

  useEffect(() => { fetchKeys(); }, []);

  const fetchKeys = async () => {
    try {
      const res = await api.listKeys();
      const data = await res.json();
      setKeys(Array.isArray(data) ? data : []);
    } catch {}
  };

  const handleCreateKey = async () => {
    setRotating(true);
    setNewRotateKey(null);
    try {
      const res = await api.createKey(newKeyName);
      const data = await res.json();
      if (data.api_key) {
        setNewRotateKey(data.api_key);
        setShowCreateModal(false);
        setNewKeyName("");
        fetchKeys();
      }
    } catch (err) {
      console.error("Failed to create key", err);
    }
    setRotating(false);
  };

  const revokeKey = async (id) => {
    if (!confirm("Are you sure you want to revoke this key? This action cannot be undone.")) return;
    try {
      const res = await api.revokeKey(id);
      if (!res.ok) {
        const error = await res.json();
        console.error("Revoke failed:", error);
        alert(`Failed to revoke key: ${error.detail || 'Unknown error'}`);
      } else {
        fetchKeys();
      }
    } catch (err) {
      console.error("Error revoking key:", err);
    }
  };

  const copyKey = (key) => {
    if (!key) return;
    navigator.clipboard.writeText(key);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const STATS = [
    { icon: Key, label: 'Active Keys', value: keys.filter(k => k.is_active).length, color: '#2563eb' },
    { icon: Globe, label: 'Tenant', value: tenant || '—', color: '#7c3aed', small: true },
    { icon: Activity, label: 'Model Status', value: 'Healthy', color: '#22c55e' },
    { icon: BarChart3, label: 'Model Version', value: 'ORPO v3.1', color: '#f59e0b' },
  ];

  return (
    <div className={styles.page}>
      <Navbar authenticated />
      <div className={styles.content}>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <div className={styles.pageHeader}>
            <div>
              <h1 className={styles.heading}>Welcome back, <span style={{ background: 'var(--iw-gradient-text)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{user}</span></h1>
              <p className={styles.headingSub}>Manage your API keys and monitor your IWERP platform</p>
            </div>
            <div style={{ display: 'flex', gap: '10px' }}>
              <a href="/settings" className="btn-outline" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Shield size={16} /> Security Settings
              </a>
              <a href="/lab" className="btn-primary">
                <FlaskConical size={16} /> Open Laboratory
              </a>
            </div>
          </div>

          <div className={styles.statsGrid}>
            {STATS.map(({ icon: Icon, label, value, color, small }) => (
              <motion.div key={label} className={styles.statCard}
                initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <div className={styles.statIcon} style={{ background: `${color}18`, border: `1px solid ${color}30` }}>
                  <Icon size={18} color={color} />
                </div>
                <div>
                  <div className={styles.statLabel}>{label}</div>
                  <div className={styles.statValue} style={{ fontSize: small ? 14 : undefined }}>{value}</div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className={styles.section}>
            <div className={styles.sectionHeader}>
              <div>
                <h2 className={styles.sectionTitle}>Sovereign API Management</h2>
                <p className={styles.sectionSub}>Generate and manage high-security keys for external integrations</p>
              </div>
              <button className="btn-primary" onClick={() => setShowCreateModal(true)}>
                <Key size={14} /> Create New Key
              </button>
            </div>

            {newRotateKey && (
              <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                className={styles.newKeyAlert}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <h4 style={{ color: '#16a34a', fontSize: 14, fontWeight: 700, marginBottom: 4 }}>New Key Generated</h4>
                    <p style={{ color: '#475569', fontSize: 13 }}>Copy this key now. For your security, it will NOT be shown again.</p>
                  </div>
                  <button className="btn-primary" onClick={() => copyKey(newRotateKey)} style={{ background: '#16a34a' }}>
                    {copied ? <Check size={14} /> : <Copy size={14} />} Copy Secret Key
                  </button>
                </div>
                <code className={styles.fullKeyCode}>{newRotateKey}</code>
              </motion.div>
            )}

            <div className={styles.tableWrapper}>
              <table className={styles.keyTable}>
                <thead>
                  <tr>
                    <th>Key Name</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Last Used</th>
                    <th style={{ textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {keys.length === 0 ? (
                    <tr>
                      <td colSpan="5" className={styles.emptyState}>
                        <Key size={24} color="#64748b" />
                        <p>No active API keys found.</p>
                      </td>
                    </tr>
                  ) : keys.map((item) => (
                    <tr key={item.id}>
                      <td>
                        <span className={styles.keyName}>{item.name}</span>
                        <code className={styles.keyMeta}>{item.api_key_masked}</code>
                      </td>
                      <td>
                        <span className={styles.keyBadge} style={{ background: item.is_active ? undefined : '#f1f5f9', color: item.is_active ? undefined : '#64748b' }}>
                          {item.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                      <td className={styles.keyMeta}>{new Date(item.created_at).toLocaleDateString()}</td>
                      <td className={styles.keyMeta}>{item.last_used_at ? new Date(item.last_used_at).toLocaleTimeString() : 'Never'}</td>
                      <td style={{ textAlign: 'right' }}>
                        <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                          <button className={styles.copyBtn} onClick={() => copyKey(item.api_key_masked)}>
                            {copied ? <Check size={14} /> : <Copy size={14} />}
                          </button>
                          <button className={styles.revokeBtn} onClick={() => revokeKey(item.id)}>Revoke</button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className={styles.section}>
            <h2 className={styles.sectionTitle}>Developer Specifications</h2>
            <p className={styles.sectionSub} style={{ marginBottom: 20 }}>Connect external reasoning agents, LangChain wrappers, or custom applications directly to IWERP_1.0.</p>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
              <div>
                <h3 className={styles.codeTitle}>cURL (Terminal)</h3>
                <div className={styles.codeBlock}>
                  <pre>{`curl -X POST ${apiBase}/v1/sovereign/chat/completions \\
  -H "x-api-key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "Generate GL reconciliation SQL"}],
    "debug": false
  }'`}</pre>
                </div>
              </div>

              <div>
                <h3 className={styles.codeTitle}>Python (Requests)</h3>
                <div className={styles.codeBlock}>
                  <pre>{`import requests

res = requests.post(
    "${apiBase}/v1/sovereign/responses",
    headers={"x-api-key": "YOUR_API_KEY"},
    json={
        "input": "Write a Fast Formula",
        "debug": False
    }
)
print(res.json()["output_text"])`}</pre>
                </div>
              </div>

              <div>
                <h3 className={styles.codeTitle}>Agent Integration (JS/TS)</h3>
                <div className={styles.codeBlock}>
                  <pre>{`const fetchIwerp = async (prompt) => {
  const res = await fetch('${apiBase}/v1/sovereign/chat/completions', {
    method: 'POST',
    headers: {
      'x-api-key': 'YOUR_API_KEY',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      messages: [{ role: 'user', content: prompt }]
    })
  });
  return (await res.json()).output_text;
};`}</pre>
                </div>
              </div>
            </div>
            
            <div style={{ marginTop: 24, padding: '16px', background: 'rgba(37,99,235,0.05)', borderRadius: 10, border: '1px solid rgba(37,99,235,0.2)' }}>
              <h4 style={{ fontSize: 13, color: 'var(--iw-blue)', marginBottom: 8 }}>Agent Context Memory (Session Tracking)</h4>
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                The IWERP backend uses Redis persistent memory for conversational multi-turn context. To maintain a context window over multiple calls (e.g. for an autonomous ReAct loop), pass a consistent <code>session_id</code> string in your request body. If omitted, the platform will auto-generate an ephemeral ID and treat the query as zero-shot.
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {showCreateModal && (
        <div className={styles.modalOverlay}>
          <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className={styles.modal}>
            <h3 className={styles.modalTitle}>Create New API Key</h3>
            <p className={styles.modalSub}>Give your key a name to identify it later in your platform logs.</p>
            <div className={styles.inputGroup}>
              <label>Key Name (Optional)</label>
              <input 
                className={styles.input} 
                placeholder="e.g. Production Bot" 
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                autoFocus
              />
            </div>
            <div className={styles.modalActions}>
              <button className="btn-outline" onClick={() => setShowCreateModal(false)}>Cancel</button>
              <button className="btn-primary" onClick={handleCreateKey} disabled={rotating}>
                {rotating ? <RotateCw className={styles.spin} size={14} /> : 'Generate Key'}
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
