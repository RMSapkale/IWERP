/**
 * api.js — IWERP API client
 * Centralizes all calls to the backend with API key injection.
 */

function resolveApiBase() {
  const configuredBase = String(import.meta.env.VITE_API_BASE_URL || '').trim();
  if (configuredBase) {
    return configuredBase.replace(/\/+$/, '');
  }

  if (import.meta.env.DEV) {
    return 'http://localhost:8000';
  }

  throw new Error(
    'Missing VITE_API_BASE_URL for production build. Set VITE_API_BASE_URL to the deployed backend origin, for example https://iwerp.com/api.'
  );
}

const BASE = resolveApiBase();

export const getKey = () => localStorage.getItem('iwerp_key');
export const getToken = () => localStorage.getItem('iwerp_token');
export const getUser = () => localStorage.getItem('iwerp_user');
export const getTenant = () => localStorage.getItem('iwerp_tenant');

// Called after login/register — stores JWT; api_key stored separately after generation
export const setSession = (data) => {
  if (data.access_token) localStorage.setItem('iwerp_token', data.access_token);
  if (data.api_key)      localStorage.setItem('iwerp_key', data.api_key);
  if (data.username)     localStorage.setItem('iwerp_user', data.username);
  if (data.tenant_id)    localStorage.setItem('iwerp_tenant', data.tenant_id);
};

export const isLoggedIn = () => !!(getToken() || getKey());

export const clearSession = () => {
  localStorage.removeItem('iwerp_key');
  localStorage.removeItem('iwerp_token');
  localStorage.removeItem('iwerp_user');
  localStorage.removeItem('iwerp_tenant');
};

/**
 * Authenticated fetch wrapper — injects x-api-key or Bearer token.
 */
async function authFetch(path, options = {}) {
  const key   = getKey();
  const token = getToken();
  if (!options.headers) options.headers = {};
  if (key)   options.headers['x-api-key']     = key;
  if (token) options.headers['Authorization'] = `Bearer ${token}`;
  options.headers['Content-Type'] = 'application/json';

  const res = await fetch(`${BASE}${path}`, options);
  if (res.status === 401 || res.status === 403) {
    clearSession();
    window.location.href = '/login';
    throw new Error('Session expired');
  }
  return res;
}

export const api = {
  login: (username, password) =>
    fetch(`${BASE}/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    }),

  register: (username, password, tenant_name) =>
    fetch(`${BASE}/v1/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, tenant_name }),
    }),

  listKeys: () => authFetch('/v1/auth/keys/list'),

  createKey: (name) =>
    authFetch('/v1/auth/keys/create', {
      method: 'POST',
      body: JSON.stringify({ name: name || undefined }),
    }),

  revokeKey: (id) => authFetch(`/v1/auth/keys/revoke/${id}`, { method: 'DELETE' }),

  chat: (message, sessionId = null) =>
    authFetch('/v1/rag/chat', {
      method: 'POST',
      body: JSON.stringify({
        messages: [{ role: 'user', content: message }],
        session_id: sessionId,
        use_rag: true
      }),
    }),

  sovereignChat: (messages, options = {}) =>
    authFetch('/v1/sovereign/chat/completions', {
      method: 'POST',
      body: JSON.stringify({
        messages,
        ...options,
      }),
    }),

  sovereignResponses: (input, options = {}) =>
    authFetch('/v1/sovereign/responses', {
      method: 'POST',
      body: JSON.stringify({
        input,
        ...options,
      }),
    }),

  evaluateCases: (cases, options = {}) =>
    authFetch('/v1/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        cases,
        ...options,
      }),
    }),

  listBenchmarkRuns: () => authFetch('/v1/benchmarks'),

  startBenchmarkRun: (payload) =>
    authFetch('/v1/benchmarks', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getBenchmarkRun: (runId) => authFetch(`/v1/benchmarks/${runId}`),

  getBenchmarkCases: (runId, params = {}) => {
    const query = new URLSearchParams(
      Object.entries(params).reduce((acc, [key, value]) => {
        if (value !== undefined && value !== null && value !== '') acc[key] = value;
        return acc;
      }, {})
    );
    const suffix = query.toString() ? `?${query.toString()}` : '';
    return authFetch(`/v1/benchmarks/${runId}/cases${suffix}`);
  },

  getTrace: (traceId) => authFetch(`/v1/traces/${traceId}`),

  getReadiness: () => authFetch('/v1/health/readiness'),
};
