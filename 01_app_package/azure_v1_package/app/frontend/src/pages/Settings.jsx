import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Key, RefreshCw, Trash2, Copy, Check, Eye, EyeOff, ShieldAlert, ShieldCheck } from 'lucide-react';
import { api, getKey, getUser, getTenant } from '../api';
import styles from './Settings.module.css';

const Settings = () => {
    const [keys, setKeys] = useState([]);
    const [loading, setLoading] = useState(true);
    const [newKey, setNewKey] = useState(null);
    const [copied, setCopied] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchKeys();
    }, []);

    const fetchKeys = async () => {
        try {
            setLoading(true);
            const res = await api.listKeys();
            if (!res.ok) throw new Error("Failed to load keys");
            const data = await res.json();
            setKeys(Array.isArray(data) ? data : []);
            setError(null);
        } catch (err) {
            setError(err.message || "Failed to fetch API keys");
        } finally {
            setLoading(false);
        }
    };

    const handleRotate = async () => {
        if (!window.confirm("Rotating your key will immediately invalidate the old one. You MUST save the new key to maintain session.")) return;
        try {
            const res = await api.rotateKey();
            if (!res.ok) throw new Error("Rotation failed");
            const data = await res.json();
            setNewKey(data.key);
        } catch (err) {
            setError("Failed to rotate API key");
        }
    };

    const handleRevoke = async () => {
        if (!window.confirm("Are you sure you want to REVOKE your API key? You will be logged out.")) return;
        try {
            await api.revokeKey();
            localStorage.clear();
            window.location.href = '/login';
        } catch (err) {
            setError("Failed to revoke API key");
        }
    };

    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className={styles.container}>
            <header className={styles.header}>
                <motion.div 
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={styles.titleSection}
                >
                    <h1>Security & Sovereignty</h1>
                    <p>Orchestrate your IWERP SCM access credentials across the mesh</p>
                </motion.div>
            </header>

            <main className={styles.content}>
                <section className={styles.card}>
                    <div className={styles.cardHeader}>
                        <Key className={styles.icon} />
                        <h2>API Key Management</h2>
                    </div>
                    
                    <div className={styles.cardBody}>
                        {loading ? (
                            <div className={styles.loadingState}>
                                <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }}>
                                    <RefreshCw size={24} color="#00ffcc" />
                                </motion.div>
                                <p>Scanning mesh for active credentials...</p>
                            </div>
                        ) : (
                            <div className={styles.keyList}>
                                <AnimatePresence>
                                    {keys.map((k, idx) => (
                                        <motion.div 
                                            key={idx} 
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            className={styles.keyItem}
                                        >
                                            <div className={styles.keyDetails}>
                                                <code className={styles.keyCode}>{k.key}</code>
                                                <div className={styles.keyMeta}>
                                                    <span className={styles.statusBadge}>
                                                        <ShieldCheck size={12} /> {k.status}
                                                    </span>
                                                    <span className={styles.tenantBadge}>Tenant: {k.tenant}</span>
                                                </div>
                                            </div>
                                            <div className={styles.actions}>
                                                <button onClick={handleRotate} className={styles.toolButton} title="Rotate Key">
                                                    <RefreshCw size={18} />
                                                </button>
                                                <button onClick={handleRevoke} className={styles.toolButton} style={{ color: '#ff4444' }} title="Revoke Key">
                                                    <Trash2 size={18} />
                                                </button>
                                            </div>
                                        </motion.div>
                                    ))}
                                </AnimatePresence>
                                {keys.length === 0 && !error && <p className={styles.emptyMsg}>No active keys found in current tenant.</p>}
                            </div>
                        )}

                        <AnimatePresence>
                            {newKey && (
                                <motion.div 
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    className={styles.newKeyBox}
                                >
                                    <div className={styles.alertHeader}>
                                        <ShieldAlert size={24} />
                                        <strong>CRITICAL: New API Key Materialized</strong>
                                    </div>
                                    <p className={styles.alertText}>
                                        Copy this key immediately. For your security, this represents a unique session token that **cannot be retrieved again**.
                                    </p>
                                    <div className={styles.codeBlock}>
                                        <code>{newKey}</code>
                                        <button onClick={() => copyToClipboard(newKey)} className={styles.copyBtn}>
                                            {copied ? <Check size={16} /> : <Copy size={16} />}
                                            {copied ? 'Copied' : 'Copy'}
                                        </button>
                                    </div>
                                    <button 
                                        onClick={() => {
                                            localStorage.setItem('iwerp_key', newKey);
                                            setNewKey(null);
                                            fetchKeys();
                                        }}
                                        className={styles.applyButton}
                                    >
                                        Apply New Key to Mesh Session
                                    </button>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {error && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className={styles.errorBanner}>
                                <ShieldAlert size={18} /> {error}
                            </motion.div>
                        )}
                    </div>
                </section>
                
                <section className={styles.card}>
                    <div className={styles.cardHeader}>
                        <ShieldAlert className={styles.icon} />
                        <h2>Sovereignty Guidelines</h2>
                    </div>
                    <div className={styles.cardBody}>
                        <ul className={styles.guideList}>
                            <li>Mesh credentials are cryptographically bound to your tenant signature.</li>
                            <li>Rotate keys every 30 days to maintain SCM parity.</li>
                            <li>In case of a breach, utilize the REVOKE trigger immediately to isolate the tenant.</li>
                            <li>Avoid plain-text storage of SCM tokens in client-side artifacts.</li>
                        </ul>
                    </div>
                </section>
            </main>
        </div>
    );
};

export default Settings;
