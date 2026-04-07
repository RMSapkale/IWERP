/**
 * Login.jsx — IWERP Login Screen
 */
import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Shield, Eye, EyeOff } from 'lucide-react';
import { api, setSession } from '../api';
import styles from './Auth.module.css';

export default function Login() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ tenant_name: '', password: '' });
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await api.login(form.tenant_name, form.password);
      const data = await res.json();
      if (res.ok) {
        setSession(data);
        navigate('/lab');
      } else {
        setError(data.detail || 'Invalid credentials');
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
        <h2 className={styles.title}>Welcome back</h2>
        <p className={styles.sub}>Access the Intelligence platform</p>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.field}>
            <label>Company / Tenant Name</label>
            <input
              className="form-input"
              type="text"
              placeholder="e.g. IWings"
              value={form.tenant_name}
              onChange={e => setForm({ ...form, tenant_name: e.target.value })}
              required
            />
          </div>
          <div className={styles.field}>
            <label>Password</label>
            <div className={styles.pwWrapper}>
              <input
                className="form-input"
                type={showPw ? 'text' : 'password'}
                placeholder="••••••••"
                value={form.password}
                onChange={e => setForm({ ...form, password: e.target.value })}
                required
              />
              <button type="button" className={styles.eyeBtn} onClick={() => setShowPw(!showPw)}>
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <button type="submit" className="btn-primary" style={{ width: '100%', justifyContent: 'center' }} disabled={loading}>
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <p className={styles.switchLink}>
          No account yet?{' '}
          <Link to="/register">Create IWERP ID</Link>
        </p>
      </motion.div>
    </div>
  );
}
