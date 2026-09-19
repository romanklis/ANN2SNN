// Thin fetch wrapper around the ANN2SNN Flask API.
//
// The dashboard is served *by* the Flask app at "/", so the default API base is
// the page origin (same-origin, no CORS). It can be overridden for the Vite dev
// server with `?api=http://localhost:8080` or `window.ANN2SNN_API_BASE`.
//
// No simulation maths lives here: every trajectory/error/spike value comes from
// the Python engine.

function resolveBase() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("api") || window.ANN2SNN_API_BASE;
  if (override) return String(override).replace(/\/+$/, "");
  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    return "";
  }
  return "http://localhost:8080";
}

export const API_BASE = resolveBase();

export class ApiError extends Error {
  constructor(message, { status = 0, kind = "http", payload = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.kind = kind; // "network" | "http"
    this.payload = payload;
  }
}

async function request(path, { method = "GET", body = null, signal = null, raw = false } = {}) {
  const url = `${API_BASE}${path}`;
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (err) {
    if (err && err.name === "AbortError") throw err;
    throw new ApiError(
      `Cannot reach the ANN2SNN API at ${API_BASE || window.location.origin}${path}. ` +
        `Is the Flask server running?`,
      { kind: "network" }
    );
  }

  if (raw && res.ok) return res;

  let data = null;
  const text = await res.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: text.slice(0, 300) };
    }
  }
  if (!res.ok) {
    const detail = data && data.error ? data.error : `HTTP ${res.status}`;
    throw new ApiError(detail, { status: res.status, kind: "http", payload: data });
  }
  return data;
}

/** GET /api/controllers — catalogue + benchmark geometry. */
export function fetchControllers(signal) {
  return request("/api/controllers", { signal });
}

/** GET /api/health — liveness + versions. */
export function fetchHealth(signal) {
  return request("/api/health", { signal });
}

/**
 * POST /api/simulate — run one controller on the orbit.
 * @param {object} opts {controller, seed, steps, radius, freq, spike_format}
 */
export function simulate(opts, signal) {
  return request("/api/simulate", { method: "POST", body: compact(opts), signal });
}

/** POST /api/benchmark — run several controllers on one shared reference. */
export function benchmark(opts, signal) {
  return request("/api/benchmark", { method: "POST", body: compact(opts), signal });
}

/** POST /api/train — start behavioural distillation. */
export function startTraining(opts = {}) {
  return request("/api/train", { method: "POST", body: compact(opts) });
}

/** GET /api/train/<id> — distillation progress. */
export function trainingStatus(jobId) {
  return request(`/api/train/${encodeURIComponent(jobId)}`);
}

/** POST /api/export/mp4 — server-rendered MP4; returns a Blob. */
export async function exportMp4(opts, signal) {
  const res = await request("/api/export/mp4", {
    method: "POST",
    body: compact(opts),
    signal,
    raw: true,
  });
  return res.blob();
}

/** POST /api/sessions — create an interactive session. */
export function newSession(opts, signal) {
  return request("/api/sessions", { method: "POST", body: compact(opts), signal });
}

/** POST /api/sessions/<id>/step */
export function stepSession(sessionId, { n = 1, action = null } = {}, signal) {
  return request(`/api/sessions/${encodeURIComponent(sessionId)}/step`, {
    method: "POST",
    body: compact({ n, action }),
    signal,
  });
}

/** GET /api/sessions/<id>/trajectory */
export function sessionTrajectory(sessionId, signal) {
  return request(`/api/sessions/${encodeURIComponent(sessionId)}/trajectory`, { signal });
}

function compact(obj) {
  const out = { ...obj };
  Object.keys(out).forEach((k) => out[k] === undefined && delete out[k]);
  return out;
}
