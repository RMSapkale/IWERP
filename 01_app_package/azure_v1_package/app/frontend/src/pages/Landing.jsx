/**
 * Landing.jsx — IWERP Platform Landing Page
 * Light mode enterprise aesthetic.
 */
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Shield, Database, Zap, Globe, Lock, BarChart3,
  Brain, Code2, ArrowRight, CheckCircle2, ChevronRight
} from 'lucide-react';
import Navbar from '../components/Navbar';
import styles from './Landing.module.css';

const fadeUp = { hidden: { opacity: 0, y: 30 }, show: { opacity: 1, y: 0 } };
const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.1 } } };

const FEATURES = [
  { icon: Brain, title: 'Enterprise Intelligence', desc: 'Proprietary LLM trained exclusively on Oracle Fusion ERP with zero dependency on external AI providers.' },
  { icon: Database, title: 'Enterprise SQL Validation', desc: 'Real-time SQL validation loop checking queries against a robust representation of 5,000+ Oracle Fusion tables.' },
  { icon: Shield, title: 'Multi-Tenant Isolation', desc: 'Per-organization data boundaries, namespaced vector stores, and API key tenancy from day one.' },
  { icon: Globe, title: 'RAG Knowledge Base', desc: 'Retrieval-augmented generation grounded in thousands of Oracle documents and golden SQL examples.' },
  { icon: Zap, title: 'Streaming Responses', desc: 'Real-time token streaming with contextual conversation memory across complex multi-turn sessions.' },
  { icon: Lock, title: 'No-Trace Privacy', desc: 'All model weights hosted internally. Your prompts never leave your secure boundary.' },
];

const PRODUCTS = [
  { name: 'IWERP_1.0', tag: 'Core LLM', desc: 'The foundational Language Model for Oracle Fusion, fine-tuned with ORPO alignment.', color: '#2563eb' },
  { name: 'SQL Validator', tag: 'Validation Engine', desc: 'Self-correcting SQL validation service that executes and repairs Oracle queries before delivery.', color: '#7c3aed' },
  { name: 'Vector RAG', tag: 'Knowledge Layer', desc: 'Distributed vector retrieval over Oracle documentation, schemas, and golden SQL benchmarks.', color: '#06b6d4' },
  { name: 'Agent Runtime', tag: 'Agentic Platform', desc: 'Multi-step ReAct planning with tool calling, memory, and HITL approval gates for complex ERP tasks.', color: '#10b981' },
  { name: 'Adapter Hub', tag: 'Model Registry', desc: 'Specialized LoRA adapters per Oracle module — Finance, SCM, HCM, Procurement.', color: '#f59e0b' },
  { name: 'Compliance Layer', tag: 'Governance', desc: 'Audit trails, PII scrubbing, policy enforcement and row-level tenant isolation out of the box.', color: '#ef4444' },
];

const CAPABILITIES = [
  'Oracle Fusion GL, AP, AR, FA expert SQL generation',
  'Complex multi-org MOAC queries with PK/FK annotations',
  'Fast Formula authoring and validation',
  'FBDI/HDL template generation for data migration',
  'OTBI/BIP report SQL with semantic joins',
  'Oracle Integration Cloud (OIC) flow guidance',
  'HCM Payroll and Absence formula logic',
  'Security role and RBAC navigation guidance',
];

export default function Landing() {
  return (
    <div className={styles.page}>
      <Navbar />

      {/* Hero */}
      <section className={styles.hero}>
        <div className={styles.heroBg} />
        <div className={styles.heroGrid} />
        <motion.div
          className={styles.heroContent}
          initial="hidden" animate="show" variants={stagger}
        >
          <motion.div variants={fadeUp} className={styles.heroPill}>
            <span className="tag"><Zap size={12} /> Enterprise AI Platform</span>
          </motion.div>

          <motion.h1 variants={fadeUp} className={styles.heroTitle}>
            The Global Ecosystem of<br />
            <span className="gradient-text">Enterprise Intelligence</span>
          </motion.h1>

          <motion.p variants={fadeUp} className={styles.heroSub}>
            IWERP is a production-grade, self-contained AI for Oracle Fusion ERP.
            Expert-tier SQL. Multi-tenant isolation. Zero external AI dependency.
            Built by <strong>IntegrationWings</strong>.
          </motion.p>

          <motion.div variants={fadeUp} className={styles.heroCtas}>
            <Link to="/register" className="btn-primary">
              Start Free Trial <ArrowRight size={16} />
            </Link>
            <Link to="/login" className="btn-outline">
              Sign In to Platform
            </Link>
          </motion.div>

          <motion.div variants={fadeUp} className={styles.heroStats}>
            {[['5,000+', 'Oracle Tables'], ['287K', 'Training Samples'], ['<200ms', 'P95 Latency'], ['99.9%', 'Uptime SLA']].map(([n, l]) => (
              <div key={l} className={styles.stat}>
                <div className={styles.statNum}>{n}</div>
                <div className={styles.statLabel}>{l}</div>
              </div>
            ))}
          </motion.div>
        </motion.div>

        {/* Animated terminal preview */}
        <motion.div
          className={styles.terminal}
          initial={{ opacity: 0, x: 60 }} animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4, duration: 0.8 }}
        >
          <div className={styles.terminalHeader}>
            <span className={styles.dot} style={{ background: '#ef4444' }} />
            <span className={styles.dot} style={{ background: '#f59e0b' }} />
            <span className={styles.dot} style={{ background: '#22c55e' }} />
            <span style={{ color: '#64748b', fontSize: 12, marginLeft: 8 }}>IWERP Laboratory</span>
          </div>
          <div className={styles.terminalBody}>
            <div className={styles.termLine}><span className={styles.prompt}>user</span> Generate GL to AP reconciliation SQL with PK/FK</div>
            <div className={styles.termLine} style={{ marginTop: 12, color: '#3b82f6' }}>IWERP_1.0 ⟶</div>
            <div className={styles.termCode}>
              {`SELECT 
  -- PK: GL_CODE_COMBINATIONS.CODE_COMBINATION_ID
  gcc.CODE_COMBINATION_ID,
  gcc.SEGMENT1 AS "Company",
  gcc.SEGMENT2 AS "Department",
  SUM(gb.PERIOD_NET_DR) AS Total_Debits,
  SUM(gb.PERIOD_NET_CR) AS Total_Credits,
  -- FK: AP_INVOICE_DISTRIBUTIONS_ALL 
  --     .DIST_CODE_COMBINATION_ID → GCC
  COUNT(DISTINCT aid.INVOICE_ID) AS AP_Invoices
FROM GL_BALANCES gb
  JOIN GL_CODE_COMBINATIONS gcc
    ON gcc.CODE_COMBINATION_ID = gb.CODE_COMBINATION_ID
  JOIN AP_INVOICE_DISTRIBUTIONS_ALL aid
    ON aid.DIST_CODE_COMBINATION_ID = gcc.CODE_COMBINATION_ID
WHERE gb.LEDGER_ID = :p_ledger_id
  AND gb.PERIOD_NAME = :p_period
GROUP BY gcc.CODE_COMBINATION_ID, 
         gcc.SEGMENT1, gcc.SEGMENT2`}
            </div>
            <div className={styles.termLine} style={{ marginTop: 10, color: '#16a34a', fontSize: 12 }}>
              ✅ Verified for syntactical and relational accuracy · 3 PK/FK annotations
            </div>
          </div>
        </motion.div>
      </section>

      {/* Features */}
      <section className={styles.section} id="features">
        <motion.div className={styles.sectionHeader}
          initial="hidden" whileInView="show" viewport={{ once: true }} variants={stagger}>
          <motion.p variants={fadeUp} className={styles.eyebrow}>Platform Capabilities</motion.p>
          <motion.h2 variants={fadeUp}>Expert Services for <span className="gradient-text">Enterprise Intelligence</span></motion.h2>
          <motion.p variants={fadeUp} className={styles.sectionSub}>
            Everything you need to run a sovereign AI platform — from model serving to compliance, built for Oracle Fusion experts.
          </motion.p>
        </motion.div>
        <motion.div className={styles.grid3}
          initial="hidden" whileInView="show" viewport={{ once: true }} variants={stagger}>
          {FEATURES.map(({ icon: Icon, title, desc }) => (
            <motion.div key={title} className="card" variants={fadeUp}>
              <div className={styles.featureIcon}><Icon size={22} /></div>
              <h3 className={styles.cardTitle}>{title}</h3>
              <p className={styles.cardDesc}>{desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Benchmarks Section */}
      <section className={styles.sectionAlt}>
        <motion.div className={styles.sectionHeader}
          initial="hidden" whileInView="show" viewport={{ once: true }} variants={stagger}>
          <motion.p variants={fadeUp} className={styles.eyebrow}>Performance Certification</motion.p>
          <motion.h2 variants={fadeUp}>Certified <span className="gradient-text">Absolute Domain Mastery</span></motion.h2>
          <motion.p variants={fadeUp} className={styles.sectionSub}>
            IWERP 1.0 is audited against 1,000 baseline Oracle Fusion scenarios, outperforming generalist models by a 2.4x margin.
          </motion.p>
        </motion.div>

        <motion.div className={styles.benchmarkGrid}
          initial="hidden" whileInView="show" viewport={{ once: true }} variants={stagger}>
          <motion.div className={styles.benchmarkCard} variants={fadeUp}>
            <div className={styles.benchHeader}>
              <BarChart3 size={20} color="#2563eb" />
              <h4>Sovereign Functional Mastery</h4>
            </div>
            <p className={styles.benchDetail}>
              Functionally verified against 1,000 Oracle Fusion business objects.
            </p>
            <div className={styles.benchMetrics}>
              <div className={styles.benchMetric}>
                <span>Expert Audit Score</span>
                <strong>98.2%</strong>
              </div>
              <div className={styles.benchMetric}>
                <span>Business Rule Accuracy</span>
                <strong>97.4%</strong>
              </div>
              <div className={styles.benchMetric}>
                <span>SQL Schema Precision</span>
                <strong>98.4%</strong>
              </div>
            </div>
          </motion.div>

          <motion.div className={styles.benchmarkCard} variants={fadeUp}>
            <div className={styles.benchHeader}>
              <Code2 size={20} color="#7c3aed" />
              <h4>Advanced NLP Audit (BERT / Semantic)</h4>
            </div>
            <p className={styles.benchDetail}>
              Measuring semantic truth and functional intent, overcoming string-match limitations.
            </p>
            <div className={styles.benchMetrics}>
              <div className={styles.benchMetric}>
                <span>BERTScore Similarity</span>
                <strong>0.845</strong>
              </div>
              <div className={styles.benchMetric}>
                <span>Exact Match (Token)</span>
                <strong>15.2%</strong>
              </div>
              <div className={styles.benchMetric}>
                <span>Domain Saturation</span>
                <strong>100%</strong>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </section>

      {/* Products */}
      <section className={styles.section} id="products">
        <motion.div className={styles.sectionHeader}
          initial="hidden" whileInView="show" viewport={{ once: true }} variants={stagger}>
          <motion.p variants={fadeUp} className={styles.eyebrow}>Our Products</motion.p>
          <motion.h2 variants={fadeUp}>The <span className="gradient-text">Enterprise Stack</span></motion.h2>
        </motion.div>
        <motion.div className={styles.grid3}
          initial="hidden" whileInView="show" viewport={{ once: true }} variants={stagger}>
          {PRODUCTS.map(({ name, tag, desc, color }) => (
            <motion.div key={name} className={styles.productCard} variants={fadeUp}
              style={{ '--accent': color }}>
              <div className={styles.productTag}>{tag}</div>
              <h3 className={styles.productName}>{name}</h3>
              <p className={styles.cardDesc}>{desc}</p>
              <div className={styles.productLink}>Explore <ChevronRight size={14} /></div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Capabilities */}
      <section className={styles.sectionAlt} id="capabilities">
        <motion.div className={styles.capabilitiesInner}
          initial="hidden" whileInView="show" viewport={{ once: true }} variants={stagger}>
          <motion.div variants={fadeUp} className={styles.capLeft}>
            <p className={styles.eyebrow}>Revolutionize Your ERP</p>
            <h2>Deep Oracle Fusion<br /><span className="gradient-text">Domain Mastery</span></h2>
            <p style={{ color: 'var(--text-secondary)', marginTop: 16, lineHeight: 1.8 }}>
              IWERP is trained on 287,000+ Oracle Fusion ERP examples. It understands your tables,
              your business processes, and your compliance requirements natively.
            </p>
            <Link to="/register" className="btn-primary" style={{ marginTop: 28, display: 'inline-flex' }}>
              Try the Laboratory <ArrowRight size={16} />
            </Link>
          </motion.div>
          <motion.div variants={fadeUp} className={styles.capRight}>
            {CAPABILITIES.map((cap) => (
              <div key={cap} className={styles.capItem}>
                <CheckCircle2 size={16} color="#16a34a" style={{ flexShrink: 0 }} />
                <span>{cap}</span>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </section>

      {/* CTA Banner */}
      <section className={styles.ctaBanner}>
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <h2>Ready to Deploy <span className="gradient-text">Enterprise Intelligence</span>?</h2>
          <p>Join Oracle Fusion experts already using IWERP to generate production-grade SQL and accelerate ERP delivery.</p>
          <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginTop: 32, flexWrap: 'wrap' }}>
            <Link to="/register" className="btn-primary">Create Account <ArrowRight size={16} /></Link>
            <Link to="/login" className="btn-outline">Sign In</Link>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className={styles.footer}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 12 }}>
          <Shield size={16} color="#2563eb" />
          <span style={{ fontFamily: 'Sora', fontWeight: 700 }}>IWERP</span>
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
          © 2026 IntegrationWings. All rights reserved.
        </p>
      </footer>
    </div>
  );
}
