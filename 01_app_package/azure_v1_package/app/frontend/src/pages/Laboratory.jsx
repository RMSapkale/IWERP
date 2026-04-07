/**
 * Laboratory.jsx — IWERP Chat Laboratory
 * V1 runtime parity surface for certified grounded capabilities.
 */
import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Trash2, Shield, User, Loader2, BarChart3, MessageSquare } from 'lucide-react';
import Navbar from '../components/Navbar';
import Benchmarks from './Benchmarks';
import { api } from '../api';
import styles from './Laboratory.module.css';

const FAIL_CLOSED_PATTERN = /insufficient grounded data|cannot generate verified answer/i;
const ADMIN_DEBUG = import.meta.env.VITE_ENABLE_ADMIN_DEBUG === '1';
const PLSQL_ENABLED = import.meta.env.VITE_ENABLE_PLSQL === '1';

const CAPABILITIES = [
  { label: 'SQL', status: 'Supported' },
  { label: 'Fast Formula', status: 'Supported' },
  { label: 'Oracle Fusion Q&A', status: 'Supported' },
  { label: 'Procedures', status: 'Supported' },
  { label: 'Troubleshooting', status: 'Supported' },
  { label: 'PL/SQL', status: PLSQL_ENABLED ? 'Beta' : 'Hidden' },
];

const PROMPTS = [
  'what is EPM?',
  'how to create custom ESS job?',
  'what is payroll gratuity?',
  'Create an Oracle Fusion SQL query to extract AP invoice distribution details with validated and accounted filters.',
  'Create a Fast Formula for sick leave accrual with default and return logic.',
];

function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatReply(text) {
  const safe = escapeHtml(text || '');
  return safe
    .replace(/```sql\s*([\s\S]*?)```/gi, '<pre class="sql-block"><code>$1</code></pre>')
    .replace(/```(?:fastformula|formula)?\s*([\s\S]*?)```/gi, '<pre class="formula-block"><code>$1</code></pre>')
    .replace(/```([\s\S]*?)```/g, '<pre class="sql-block"><code>$1</code></pre>')
    .replace(/^\[(.+?)\]\s*$/gm, '<span class="section-label">[$1]</span>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>');
}

function toTaskLabel(taskType) {
  const normalized = String(taskType || '').trim();
  if (!normalized) return 'general';
  return normalized.replace(/_/g, ' ');
}

function inferResultType(meta, text) {
  const payload = String(text || '').toLowerCase();
  const task = String(meta?.taskType || '').toLowerCase();

  if (meta?.refusal) return 'refusal';
  if (task === 'sql_generation' || task === 'sql_troubleshooting') return 'sql';
  if (task === 'fast_formula_generation' || task === 'fast_formula_troubleshooting') return 'fast_formula';
  if (task === 'procedure' || task === 'navigation') return 'procedure';
  if (task === 'troubleshooting') return 'troubleshooting';
  if (task === 'summary' || task === 'qa' || task === 'general') return 'summary';

  if (task.includes('sql') || /\[sql\]|```sql|select\s+/i.test(payload)) return 'sql';
  if (task.includes('fast_formula') || /\[formula\]|formula_type|default for|inputs are|return/i.test(payload)) return 'fast_formula';
  if (/symptom|likely causes|resolution/i.test(payload)) return 'troubleshooting';
  if (/steps:|ordered steps|prerequisites/i.test(payload)) return 'procedure';
  return 'summary';
}

function normalizeRefusalText(rawText) {
  if (!FAIL_CLOSED_PATTERN.test(rawText || '')) {
    return rawText;
  }
  return [
    'I cannot verify this request with grounded Oracle Fusion evidence in the current corpus.',
    '',
    'Try adding a specific module, business object, or task context so I can route to supported grounding.',
  ].join('\n');
}

export default function Laboratory() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [view, setView] = useState('lab'); // 'lab' or 'audit'
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'IWERP V1 is ready for grounded Oracle Fusion SQL, Fast Formula, Q&A, procedure, and troubleshooting requests. PL/SQL is currently beta/guarded.',
      meta: {
        taskType: 'summary',
        selectedModule: 'Oracle Fusion',
        citations: [],
        grounded: true,
        refusal: false,
        verifierStatus: 'PASSED',
        resultType: 'summary',
      },
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, view]);

  const send = async (msg) => {
    const text = msg || input.trim();
    if (!text || loading) return;
    setInput('');

    const newMessages = [...messages, { role: 'user', content: text }];
    setMessages(newMessages);
    setLoading(true);

    try {
      const res = await api.sovereignChat(
        [{ role: 'user', content: text }],
        { metadata: { session_id: sessionId, surface: 'v1_ui' }, debug: ADMIN_DEBUG }
      );
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail || 'Request failed.');
      }

      let reply = '';
      if (data.output_text) {
        reply = data.output_text;
      } else if (data.choices && data.choices[0] && data.choices[0].message) {
        reply = data.choices[0].message.content;
      } else {
        reply = data.reply || data.detail || 'No response received.';
      }

      if (typeof reply === 'object') {
        reply = `Error: ${JSON.stringify(reply)}`;
      }

      const refusal = Boolean(data.refusal || FAIL_CLOSED_PATTERN.test(reply));
      const normalizedReply = refusal ? normalizeRefusalText(reply) : reply;
      const meta = {
        taskType: data.task_type || 'general',
        selectedModule: data.selected_module || 'UNKNOWN',
        citations: Array.isArray(data.citations) ? data.citations : [],
        grounded: Boolean(data.grounded || (Array.isArray(data.citations) && data.citations.length > 0)),
        refusal,
        verifierStatus: data.verifier_status || 'UNKNOWN',
        resultType: inferResultType({ taskType: data.task_type, refusal }, normalizedReply),
        traceId: data.id || null,
        decisionTrace: ADMIN_DEBUG ? (data.decision_trace || null) : null,
        groundingTrace: ADMIN_DEBUG ? (data.grounding_trace || null) : null,
      };
      setMessages(prev => [...prev, { role: 'assistant', content: normalizedReply, meta }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${e?.message || 'Could not reach IWERP server.'}`,
        meta: {
          taskType: 'error',
          selectedModule: 'N/A',
          citations: [],
          grounded: false,
          refusal: true,
          verifierStatus: 'FAILED',
          resultType: 'refusal',
        },
      }]);
    }
    setLoading(false);
  };


  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div className={styles.page}>
      <Navbar authenticated />

      <div className={styles.layout}>
        {/* Sidebar with quick prompts */}
        <aside className={styles.sidebar}>
          <div className={styles.sidebarHeader}>
            <Shield size={16} color="#2563eb" />
            <span>Navigation</span>
          </div>
          <button 
            className={`${styles.promptBtn} ${view === 'lab' ? styles.activeTab : ''}`} 
            onClick={() => setView('lab')}
          >
            <MessageSquare size={14} /> Laboratory Chat
          </button>
          <button 
            className={`${styles.promptBtn} ${view === 'audit' ? styles.activeTab : ''}`} 
            onClick={() => setView('audit')}
          >
            <BarChart3 size={14} /> Performance Audit
          </button>
          
          <div className={styles.sidebarDivider} />
          
          <div className={styles.sidebarHeader}>
            <Shield size={16} color="#2563eb" />
            <span>Quick Prompts</span>
          </div>
          {PROMPTS.map(p => (
            <button key={p} className={styles.promptBtn} onClick={() => { if(view!=='lab') setView('lab'); send(p); }}>{p}</button>
          ))}

          <div className={styles.sidebarDivider} />
          <div className={styles.sidebarHeader}>
            <Shield size={16} color="#2563eb" />
            <span>V1 Capabilities</span>
          </div>
          {CAPABILITIES
            .filter((cap) => cap.status !== 'Hidden')
            .map((cap) => (
              <div key={cap.label} className={styles.capabilityRow}>
                <span>{cap.label}</span>
                <span className={`${styles.capabilityBadge} ${cap.status === 'Beta' ? styles.capabilityBeta : styles.capabilitySupported}`}>
                  {cap.status}
                </span>
              </div>
            ))}
        </aside>

        {/* Main Content Area */}
        <div className={styles.chatArea}>
          {view === 'lab' ? (
            <>
              <div className={styles.chatHeader}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Shield size={18} color="#2563eb" />
                  <span className={styles.chatTitle}>IWERP Laboratory</span>
                  <span className={styles.modelTag}>ORPO v3.1 Sovereign</span>
                </div>
                <button className={styles.clearBtn} onClick={() => setMessages([messages[0]])}>
                  <Trash2 size={14} /> Clear
                </button>
              </div>

              <div className={styles.messages}>
                <AnimatePresence initial={false}>
                  {messages.map((msg, i) => (
                    <motion.div key={i} className={`${styles.row} ${msg.role === 'user' ? styles.userRow : ''}`}
                      initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
                      <div className={`${styles.avatar} ${msg.role === 'assistant' ? styles.botAvatar : ''}`}>
                        {msg.role === 'assistant' ? <Shield size={15} /> : <User size={15} />}
                      </div>
                      <div className={styles.messageStack}>
                        {msg.role === 'assistant' && msg.meta && (
                          <div className={styles.metaRow}>
                            <span className={styles.metaChip}>{toTaskLabel(msg.meta.taskType)}</span>
                            <span className={styles.metaChip}>{msg.meta.selectedModule || 'UNKNOWN'}</span>
                            <span className={styles.metaChip}>{msg.meta.resultType}</span>
                            {msg.meta.refusal ? <span className={styles.metaChipWarn}>refusal</span> : null}
                            {msg.meta.verifierStatus ? <span className={styles.metaChip}>{msg.meta.verifierStatus}</span> : null}
                          </div>
                        )}
                        <div
                          className={`${styles.bubble} ${msg.role === 'user' ? styles.userBubble : ''}`}
                          dangerouslySetInnerHTML={{ __html: formatReply(msg.content) }}
                        />
                        {msg.role === 'assistant' && Array.isArray(msg.meta?.citations) && msg.meta.citations.length > 0 && (
                          <div className={styles.citationBlock}>
                            <div className={styles.citationTitle}>Citations</div>
                            <ul className={styles.citationList}>
                              {msg.meta.citations.slice(0, 4).map((c) => (
                                <li key={`${c.citation_id}-${c.document_id || c.title}`}>
                                  {c.citation_id || '[D]'} {c.title || c.module || 'Grounded source'}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {ADMIN_DEBUG && msg.role === 'assistant' && msg.meta?.traceId && (
                          <div className={styles.debugLine}>trace: {msg.meta.traceId}</div>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>

                {loading && (
                  <motion.div className={styles.row} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div className={`${styles.avatar} ${styles.botAvatar}`}><Shield size={15} /></div>
                    <div className={styles.bubble}>
                      <div className={styles.thinking}>
                        <Loader2 size={14} className={styles.spin} />
                        <span>IWERP is reflecting...</span>
                      </div>
                    </div>
                  </motion.div>
                )}
                <div ref={bottomRef} />
              </div>

              <div className={styles.inputArea}>
                <div className={styles.inputBox}>
                  <textarea
                    className={styles.textarea}
                    placeholder="Ask about Oracle Fusion SQL, Fast Formula, Q&A, Procedures, or Troubleshooting..."
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={onKeyDown}
                    rows={1}
                  />
                  <button className={styles.sendBtn} onClick={() => send()} disabled={loading || !input.trim()}>
                    <Send size={16} />
                  </button>
                </div>
                <p className={styles.disclaimer}>V1 grounded runtime active. PL/SQL remains beta/guarded.</p>
              </div>
            </>
          ) : (
            <Benchmarks />
          )}
        </div>
      </div>
    </div>
  );
}
