import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Database,
  FileSearch,
  Filter,
  GitCompareArrows,
  Play,
  RefreshCw,
  Route,
  Search,
  ShieldCheck,
  Workflow,
} from 'lucide-react';

import { api } from '../api';
import styles from './Benchmarks.module.css';

const PRIMARY_METRICS = [
  ['trusted_outcome_pct', 'Trusted Outcome'],
  ['over_refusal_pct', 'Over-Refusal'],
  ['refusal_correctness_pct', 'Refusal Correctness'],
  ['hallucination_pct', 'Hallucination'],
  ['wrong_module_pct', 'Wrong-Module'],
  ['semantic_correctness_pct', 'Semantic Correctness'],
  ['citation_presence_pct', 'Citation Presence'],
  ['citation_correctness_pct', 'Citation Correctness'],
  ['verifier_approved_pct', 'Verifier Approved'],
  ['grounding_supported_answer_rate_pct', 'Grounding Supported'],
];

function formatPct(value) {
  const num = Number(value ?? 0);
  return `${num.toFixed(2)}%`;
}

function formatDate(value) {
  if (!value) return '—';
  return new Date(value * 1000).toLocaleString();
}

function deltaText(current, previous) {
  const cur = Number(current ?? 0);
  const prev = Number(previous ?? 0);
  const delta = cur - prev;
  const sign = delta > 0 ? '+' : '';
  return `${sign}${delta.toFixed(2)} pts`;
}

function exportJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function MetricCard({ label, value, delta, status = 'neutral', icon: Icon }) {
  return (
    <div className={`${styles.metricCard} ${styles[`metric${status}`] || ''}`}>
      <div className={styles.metricHeader}>
        <div className={styles.metricIcon}><Icon size={16} /></div>
        <span>{label}</span>
      </div>
      <div className={styles.metricValue}>{formatPct(value)}</div>
      <div className={styles.metricDelta}>{delta || '—'}</div>
    </div>
  );
}

export default function Benchmarks() {
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [runSummary, setRunSummary] = useState(null);
  const [cases, setCases] = useState([]);
  const [selectedCase, setSelectedCase] = useState(null);
  const [readiness, setReadiness] = useState(null);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingCases, setLoadingCases] = useState(false);
  const [startingRun, setStartingRun] = useState(false);
  const [evalLoading, setEvalLoading] = useState(false);
  const [adHocEval, setAdHocEval] = useState(null);
  const [form, setForm] = useState({
    dataset: '200',
    label: '',
    module_filters: '',
    task_filters: '',
    custom_input_path: '',
  });
  const [caseFilters, setCaseFilters] = useState({
    module: '',
    task_type: '',
    failure_category: '',
    primary_verdict: 'fail',
  });
  const [singleEvalForm, setSingleEvalForm] = useState({
    query: '',
    expected_module: '',
    expected_task_type: '',
    expected_answer: '',
  });

  const loadReadiness = async () => {
    const res = await api.getReadiness();
    const data = await res.json();
    setReadiness(data);
  };

  const loadRuns = async () => {
    setLoadingRuns(true);
    try {
      const res = await api.listBenchmarkRuns();
      const data = await res.json();
      setRuns(Array.isArray(data) ? data : []);
      if (!selectedRunId && data?.length) {
        setSelectedRunId(data[0].run_id);
      }
    } finally {
      setLoadingRuns(false);
    }
  };

  const loadSummary = async (runId) => {
    if (!runId) return;
    setLoadingSummary(true);
    try {
      const res = await api.getBenchmarkRun(runId);
      const data = await res.json();
      setRunSummary(data);
    } finally {
      setLoadingSummary(false);
    }
  };

  const loadCases = async (runId, filters = caseFilters) => {
    if (!runId) return;
    setLoadingCases(true);
    try {
      const res = await api.getBenchmarkCases(runId, { ...filters, limit: 200 });
      const data = await res.json();
      const rows = data?.cases || [];
      setCases(rows);
      setSelectedCase(rows[0] || null);
    } finally {
      setLoadingCases(false);
    }
  };

  useEffect(() => {
    loadReadiness();
    loadRuns();
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    loadSummary(selectedRunId);
    loadCases(selectedRunId);
  }, [selectedRunId]);

  const onStartRun = async () => {
    setStartingRun(true);
    try {
      const payload = {
        dataset: form.dataset,
        label: form.label || undefined,
        module_filters: form.module_filters.split(',').map((item) => item.trim()).filter(Boolean),
        task_filters: form.task_filters.split(',').map((item) => item.trim()).filter(Boolean),
        custom_input_path: form.custom_input_path || undefined,
      };
      const res = await api.startBenchmarkRun(payload);
      const data = await res.json();
      await loadRuns();
      if (data?.run_id) {
        setSelectedRunId(data.run_id);
      }
    } finally {
      setStartingRun(false);
    }
  };

  const onRunSingleEval = async () => {
    if (!singleEvalForm.query.trim()) return;
    setEvalLoading(true);
    try {
      const res = await api.evaluateCases([
        {
          case_id: `adhoc_${Date.now()}`,
          query: singleEvalForm.query,
          expected_module: singleEvalForm.expected_module || undefined,
          expected_task_type: singleEvalForm.expected_task_type || undefined,
          expected_answer: singleEvalForm.expected_answer || undefined,
        },
      ], { debug: true });
      const data = await res.json();
      setAdHocEval(data?.cases?.[0] || null);
    } finally {
      setEvalLoading(false);
    }
  };

  const evaluationSummary = runSummary?.summary?.evaluation_summary || {};
  const primary = evaluationSummary.primary_metrics || {};
  const previousRun = useMemo(() => runs.find((run) => run.run_id !== selectedRunId), [runs, selectedRunId]);
  const previousPrimary = previousRun?.primary_metrics || {};
  const perTask = evaluationSummary.per_task_type || {};
  const perModule = evaluationSummary.per_module || {};
  const perDifficulty = evaluationSummary.per_difficulty || {};
  const failureCategories = evaluationSummary.failure_categories || {};
  const scoringRubric = readiness?.scoring_rubric || {};

  return (
    <div className={styles.console}>
      <section className={styles.hero}>
        <div>
          <p className={styles.eyebrow}>Eval / Ops Console</p>
          <h1 className={styles.title}>Production Benchmark and Trace Review</h1>
          <p className={styles.subtitle}>
            Operator view for benchmark execution, task-aware scoring, case review, and trace inspection.
          </p>
        </div>
        <div className={styles.heroMeta}>
          <div className={styles.statusPill}>
            <ShieldCheck size={16} />
            <span>{readiness?.status || 'loading'}</span>
          </div>
          <div className={styles.heroStat}>
            <span>Model Backend</span>
            <strong>{readiness?.model_backend || '—'}</strong>
          </div>
          <div className={styles.heroStat}>
            <span>Rubric</span>
            <strong>{scoringRubric?.version || '—'}</strong>
          </div>
        </div>
      </section>

      <section className={styles.grid}>
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>Benchmark Runner</h2>
              <p>Launch 200 / 1000 / 5000 or custom runs against the current system.</p>
            </div>
            <button className={styles.iconButton} onClick={loadRuns} disabled={loadingRuns}>
              <RefreshCw size={16} className={loadingRuns ? styles.spin : ''} />
            </button>
          </div>

          <div className={styles.formGrid}>
            <label>
              <span>Dataset / Slice</span>
              <select value={form.dataset} onChange={(event) => setForm((prev) => ({ ...prev, dataset: event.target.value }))}>
                <option value="200">200</option>
                <option value="1000">1000</option>
                <option value="5000">5000</option>
                <option value="custom">Custom</option>
              </select>
            </label>
            <label>
              <span>Run Label</span>
              <input
                value={form.label}
                onChange={(event) => setForm((prev) => ({ ...prev, label: event.target.value }))}
                placeholder="ops_5000_apr02"
              />
            </label>
            <label>
              <span>Module Filters</span>
              <input
                value={form.module_filters}
                onChange={(event) => setForm((prev) => ({ ...prev, module_filters: event.target.value }))}
                placeholder="Financials, Procurement"
              />
            </label>
            <label>
              <span>Task Filters</span>
              <input
                value={form.task_filters}
                onChange={(event) => setForm((prev) => ({ ...prev, task_filters: event.target.value }))}
                placeholder="procedure, troubleshooting"
              />
            </label>
            <label className={styles.fullWidth}>
              <span>Custom Dataset Path</span>
              <input
                value={form.custom_input_path}
                onChange={(event) => setForm((prev) => ({ ...prev, custom_input_path: event.target.value }))}
                placeholder="/abs/path/to/custom_cases.json"
                disabled={form.dataset !== 'custom'}
              />
            </label>
          </div>

          <div className={styles.actions}>
            <button className={styles.primaryButton} onClick={onStartRun} disabled={startingRun}>
              <Play size={16} />
              <span>{startingRun ? 'Launching...' : 'Launch Benchmark'}</span>
            </button>
          </div>

          <div className={styles.runList}>
            {runs.map((run) => (
              <button
                key={run.run_id}
                className={`${styles.runItem} ${selectedRunId === run.run_id ? styles.runItemActive : ''}`}
                onClick={() => setSelectedRunId(run.run_id)}
              >
                <div>
                  <strong>{run.label}</strong>
                  <span>{run.status}</span>
                </div>
                <div>
                  <strong>{run.sample_size}</strong>
                  <span>cases</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>Single-Case Evaluation</h2>
              <p>Submit one prompt and inspect task-aware scoring without running a benchmark.</p>
            </div>
          </div>
          <div className={styles.formGrid}>
            <label className={styles.fullWidth}>
              <span>User Query</span>
              <textarea
                rows="4"
                value={singleEvalForm.query}
                onChange={(event) => setSingleEvalForm((prev) => ({ ...prev, query: event.target.value }))}
                placeholder="How do you perform receiving inspection in Oracle Fusion Purchasing?"
              />
            </label>
            <label>
              <span>Expected Module</span>
              <input
                value={singleEvalForm.expected_module}
                onChange={(event) => setSingleEvalForm((prev) => ({ ...prev, expected_module: event.target.value }))}
                placeholder="Purchasing"
              />
            </label>
            <label>
              <span>Expected Task</span>
              <input
                value={singleEvalForm.expected_task_type}
                onChange={(event) => setSingleEvalForm((prev) => ({ ...prev, expected_task_type: event.target.value }))}
                placeholder="procedure"
              />
            </label>
            <label className={styles.fullWidth}>
              <span>Expected Answer / Key Reference</span>
              <textarea
                rows="3"
                value={singleEvalForm.expected_answer}
                onChange={(event) => setSingleEvalForm((prev) => ({ ...prev, expected_answer: event.target.value }))}
                placeholder="Optional reference answer used for secondary lexical metrics."
              />
            </label>
          </div>
          <div className={styles.actions}>
            <button className={styles.primaryButton} onClick={onRunSingleEval} disabled={evalLoading}>
              <Search size={16} />
              <span>{evalLoading ? 'Evaluating...' : 'Run Evaluation'}</span>
            </button>
          </div>
          {adHocEval && (
            <div className={styles.inlineResult}>
              <div className={styles.inlineHeader}>
                <strong>{adHocEval.evaluation.primary_verdict.toUpperCase()}</strong>
                <span>{adHocEval.evaluation.task_type}</span>
              </div>
              <pre className={styles.responsePreview}>{adHocEval.response.output_text}</pre>
            </div>
          )}
        </div>
      </section>

      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>Result Dashboard</h2>
            <p>Task-aware primary metrics for the selected run.</p>
          </div>
          <div className={styles.inlineMeta}>
            <span>Selected Run</span>
            <strong>{runSummary?.label || selectedRunId || '—'}</strong>
          </div>
        </div>

        <div className={styles.metricGrid}>
          {PRIMARY_METRICS.map(([key, label], index) => {
            const icons = [ShieldCheck, AlertTriangle, CheckCircle2, AlertTriangle, Route, Activity, FileSearch, GitCompareArrows, Database, Workflow];
            const statuses = ['good', 'warn', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good'];
            return (
              <MetricCard
                key={key}
                label={label}
                value={primary[key]}
                delta={deltaText(primary[key], previousPrimary[key])}
                status={statuses[index]}
                icon={icons[index]}
              />
            );
          })}
        </div>

        <div className={styles.compareRow}>
          <div className={styles.compareCard}>
            <h3>Trend vs Prior Run</h3>
            <p>{previousRun ? previousRun.label : 'No prior run available'}</p>
            {previousRun && (
              <ul className={styles.compactList}>
                <li>Trusted outcome: {deltaText(primary.trusted_outcome_pct, previousPrimary.trusted_outcome_pct)}</li>
                <li>Over-refusal: {deltaText(primary.over_refusal_pct, previousPrimary.over_refusal_pct)}</li>
                <li>Wrong-module: {deltaText(primary.wrong_module_pct, previousPrimary.wrong_module_pct)}</li>
              </ul>
            )}
          </div>
          <div className={styles.compareCard}>
            <h3>Top Failure Clusters</h3>
            <ul className={styles.compactList}>
              {Object.entries(failureCategories).slice(0, 5).map(([name, count]) => (
                <li key={name}>{name}: {count}</li>
              ))}
              {!Object.keys(failureCategories).length && <li>No failures recorded.</li>}
            </ul>
          </div>
        </div>
      </section>

      <section className={styles.grid}>
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>Per-Task Breakdown</h2>
              <p>Primary and task-specific metric rollups by task type.</p>
            </div>
          </div>
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Task Type</th>
                  <th>Count</th>
                  <th>Trusted Outcome</th>
                  <th>Semantic</th>
                  <th>Verifier</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(perTask).map(([task, values]) => (
                  <tr key={task}>
                    <td>{task}</td>
                    <td>{values.count}</td>
                    <td>{formatPct(values.primary_metrics?.trusted_outcome_pct)}</td>
                    <td>{formatPct(values.primary_metrics?.semantic_correctness_pct)}</td>
                    <td>{formatPct(values.primary_metrics?.verifier_approved_pct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>Per-Module Breakdown</h2>
              <p>Weak modules are visible immediately from trusted outcome and refusal behavior.</p>
            </div>
          </div>
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Module</th>
                  <th>Count</th>
                  <th>Trusted Outcome</th>
                  <th>Over-Refusal</th>
                  <th>Wrong-Module</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(perModule).map(([module, values]) => (
                  <tr key={module}>
                    <td>{module}</td>
                    <td>{values.count}</td>
                    <td>{formatPct(values.primary_metrics?.trusted_outcome_pct)}</td>
                    <td>{formatPct(values.primary_metrics?.over_refusal_pct)}</td>
                    <td>{formatPct(values.primary_metrics?.wrong_module_pct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className={styles.grid}>
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>Failure Explorer</h2>
              <p>Filter failed cases by module, task type, failure reason, or verdict.</p>
            </div>
            <div className={styles.inlineActions}>
              <button className={styles.iconButton} onClick={() => loadCases(selectedRunId, caseFilters)} disabled={loadingCases}>
                <Filter size={16} />
              </button>
              <button className={styles.iconButton} onClick={() => exportJson(`${selectedRunId || 'run'}_failed_cases.json`, cases)}>
                <FileSearch size={16} />
              </button>
            </div>
          </div>

          <div className={styles.filterRow}>
            <input
              value={caseFilters.module}
              onChange={(event) => setCaseFilters((prev) => ({ ...prev, module: event.target.value }))}
              placeholder="Module"
            />
            <input
              value={caseFilters.task_type}
              onChange={(event) => setCaseFilters((prev) => ({ ...prev, task_type: event.target.value }))}
              placeholder="Task Type"
            />
            <input
              value={caseFilters.failure_category}
              onChange={(event) => setCaseFilters((prev) => ({ ...prev, failure_category: event.target.value }))}
              placeholder="Failure Category"
            />
            <select
              value={caseFilters.primary_verdict}
              onChange={(event) => setCaseFilters((prev) => ({ ...prev, primary_verdict: event.target.value }))}
            >
              <option value="">All Verdicts</option>
              <option value="fail">Fail</option>
              <option value="pass">Pass</option>
            </select>
            <button className={styles.primaryButton} onClick={() => loadCases(selectedRunId, caseFilters)} disabled={loadingCases}>
              <Search size={16} />
              <span>{loadingCases ? 'Loading...' : 'Apply'}</span>
            </button>
          </div>

          <div className={styles.caseList}>
            {cases.map((row) => (
              <button
                key={row.id}
                className={`${styles.caseItem} ${selectedCase?.id === row.id ? styles.caseItemActive : ''}`}
                onClick={() => setSelectedCase(row)}
              >
                <div>
                  <strong>{row.id}</strong>
                  <span>{row.benchmark_module} · {(row.evaluation || {}).task_type || row.benchmark_intent}</span>
                </div>
                <div>
                  <strong>{(row.evaluation || {}).primary_verdict || '—'}</strong>
                  <span>{row.failure_category || (row.evaluation || {}).failure_category || 'pass'}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>Case Review Workbench</h2>
              <p>Inspect query, output, citations, trace, and task-aware evaluation for one case.</p>
            </div>
          </div>

          {selectedCase ? (
            <div className={styles.caseReview}>
              <div className={styles.reviewBlock}>
                <span className={styles.reviewLabel}>User Query</span>
                <p>{selectedCase.question}</p>
              </div>
              <div className={styles.reviewGrid}>
                <div className={styles.reviewBlock}>
                  <span className={styles.reviewLabel}>Expected Module</span>
                  <p>{selectedCase.benchmark_module || '—'}</p>
                </div>
                <div className={styles.reviewBlock}>
                  <span className={styles.reviewLabel}>Actual Module</span>
                  <p>{selectedCase.module_detected || '—'}</p>
                </div>
                <div className={styles.reviewBlock}>
                  <span className={styles.reviewLabel}>Task Type</span>
                  <p>{(selectedCase.evaluation || {}).task_type || selectedCase.intent_detected || '—'}</p>
                </div>
                <div className={styles.reviewBlock}>
                  <span className={styles.reviewLabel}>Primary Verdict</span>
                  <p>{(selectedCase.evaluation || {}).primary_verdict || '—'}</p>
                </div>
              </div>

              <div className={styles.reviewBlock}>
                <span className={styles.reviewLabel}>Actual Output</span>
                <pre className={styles.responsePreview}>{selectedCase.output}</pre>
              </div>

              <div className={styles.reviewGrid}>
                <div className={styles.reviewBlock}>
                  <span className={styles.reviewLabel}>Decision Trace</span>
                  <pre className={styles.jsonBlock}>{JSON.stringify(selectedCase.decision_trace_summary || {}, null, 2)}</pre>
                </div>
                <div className={styles.reviewBlock}>
                  <span className={styles.reviewLabel}>Grounding Trace</span>
                  <pre className={styles.jsonBlock}>{JSON.stringify(selectedCase.grounding_trace_summary || {}, null, 2)}</pre>
                </div>
              </div>

              <div className={styles.reviewGrid}>
                <div className={styles.reviewBlock}>
                  <span className={styles.reviewLabel}>Evaluation Metrics</span>
                  <pre className={styles.jsonBlock}>{JSON.stringify(selectedCase.evaluation || {}, null, 2)}</pre>
                </div>
                <div className={styles.reviewBlock}>
                  <span className={styles.reviewLabel}>Citations</span>
                  <pre className={styles.jsonBlock}>{JSON.stringify(selectedCase.citations || [], null, 2)}</pre>
                </div>
              </div>
            </div>
          ) : (
            <div className={styles.emptyState}>No case selected.</div>
          )}
        </div>
      </section>

      <section className={styles.grid}>
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>Evaluation Dashboard</h2>
              <p>Secondary lexical metrics are displayed only where they are meaningful.</p>
            </div>
          </div>
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Difficulty</th>
                  <th>Trusted Outcome</th>
                  <th>Semantic</th>
                  <th>Over-Refusal</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(perDifficulty).map(([difficulty, values]) => (
                  <tr key={difficulty}>
                    <td>{difficulty}</td>
                    <td>{formatPct(values.primary_metrics?.trusted_outcome_pct)}</td>
                    <td>{formatPct(values.primary_metrics?.semantic_correctness_pct)}</td>
                    <td>{formatPct(values.primary_metrics?.over_refusal_pct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {selectedCase?.evaluation?.lexical_metrics && (
            <div className={styles.inlineMetrics}>
              {Object.entries(selectedCase.evaluation.lexical_metrics).map(([key, value]) => (
                <div key={key} className={styles.inlineMetric}>
                  <span>{key}</span>
                  <strong>{formatPct(value)}</strong>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <div>
              <h2>Admin / Config</h2>
              <p>Read-only production safety, scoring, and benchmark configuration visibility.</p>
            </div>
          </div>
          <div className={styles.configBlock}>
            <div className={styles.reviewBlock}>
              <span className={styles.reviewLabel}>Corpus Status</span>
              <pre className={styles.jsonBlock}>{JSON.stringify(readiness?.corpus_status || {}, null, 2)}</pre>
            </div>
            <div className={styles.reviewBlock}>
              <span className={styles.reviewLabel}>Safety Flags</span>
              <pre className={styles.jsonBlock}>{JSON.stringify(readiness?.safety_flags || {}, null, 2)}</pre>
            </div>
            <div className={styles.reviewBlock}>
              <span className={styles.reviewLabel}>Scoring Rubric</span>
              <pre className={styles.jsonBlock}>{JSON.stringify(scoringRubric || {}, null, 2)}</pre>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
