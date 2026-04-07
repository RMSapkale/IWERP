/**
 * Register.jsx — IWERP Registration
 */
import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Shield, Building2, Eye, EyeOff } from 'lucide-react';
import { api } from '../api';
import styles from './Auth.module.css';

export default function Register() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ username: '', password: '', tenant_name: '' });
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (form.password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    setLoading(true);
    try {
      const res = await api.register(form.username, form.password, form.tenant_name || 'Default_Tenant');
      const data = await res.json();
      if (res.ok) {
        navigate('/login?registered=1');
      } else {
        setError(data.detail || 'Registration failed');
      }
    } catch {
      setError('Cannot reach IWERP server. Is it running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.bg} />
      <motion.div className={styles.card}
        initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}>

        <Link to="/" className={styles.logoBlock}>
            <span>IWERP</span>
        </Link>
        <h2 className={styles.title}>Create your IWERP ID</h2>
        <p className={styles.sub}>Join the platform to access Oracle intelligence</p>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.field}>
            <label>Username</label>
            <input className="form-input" type="text" placeholder="choose_username"
              value={form.username} onChange={e => setForm({ ...form, username: e.target.value })} required />
          </div>
          <div className={styles.field}>
            <label>Password</label>
            <div className={styles.pwWrapper}>
              <input className="form-input" type={showPw ? 'text' : 'password'} placeholder="min. 8 characters"
                value={form.password} onChange={e => setForm({ ...form, password: e.target.value })} required />
              <button type="button" className={styles.eyeBtn} onClick={() => setShowPw(!showPw)}>
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>
          <div className={styles.field}>
            <label><Building2 size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} />Company / Tenant Name</label>
            <input className="form-input" type="text" placeholder="e.g. INTEGRATIONWINGS_CORP"
              value={form.tenant_name} onChange={e => setForm({ ...form, tenant_name: e.target.value })} />
            <p className={styles.hint}>This becomes your isolated data namespace across the platform</p>
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <button type="submit" className="btn-primary" style={{ width: '100%', justifyContent: 'center' }} disabled={loading}>
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>

        <p className={styles.switchLink}>
          Already have an account? <Link to="/login">Sign In</Link>
        </p>
      </motion.div>
    </div>
  );
}
