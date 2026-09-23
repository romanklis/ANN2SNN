// ANN2SNN — extended view.
//
// Sibling of the single-screen dashboard (`src/main.js`): the same engine and the
// same API, but with the full-information scope.  One `POST /api/benchmark` with
// `trace_level: "full"` returns the base traces plus the estimator internals
// (per-channel samples as the filter consumed them, innovations, covariance) and
// the environment side-channels (applied command, disturbance, impulse frames).
//
// Everything here is a *view*: it owns one frame cursor, draws a stack of lane
// charts, and reports the numbers at that cursor for every selected controller.
// No simulation maths lives in the browser.

import "./style.css";
import * as api from "./api.js";
import { Stage } from "./stage.js";
import { SpikeRaster } from "./raster.js";
import { LaneChart } from "./laneschart.js";

const $ = (id) => document.getElementById(id);

const ANN = "flylike_ann";
const SNN = "snn_transferred";
const REF_COLOR = "#aeb9c6";
const FIX_COLOR = "#b48cff";
const DIST_COLOR = "#ff6b6b";
const MAX_CONTROLLERS = 4;

//: stable colours for the hero pair, then a palette for anything else
const CTRL_COLORS = {
  flylike_ann: "#ffb454",
  snn_transferred: "#7ee0c0",
  pid: "#4da3ff",
  pid_no_ff: "#b48cff",
  dense_ann: "#dd8452",
  random_ann: "#93a1b8",
};
const PALETTE = ["#4da3ff", "#dd8452", "#7ee0c0", "#b48cff", "#ffb454", "#93a1b8"];

const el = {
  runPill: $("run-pill"),
  successPill: $("success-pill"),
  apiPill: $("api-pill"),
  trainedPill: $("trained-pill"),
  clock: $("clock"),
  frameReadout: $("frame-readout"),
  banner: $("error-banner"),
  exampleSelect: $("example-select"),
  envSelect: $("env-select"),
  seedInput: $("seed-input"),
  stepsInput: $("steps-input"),
  pickerBody: $("picker-body"),
  pickerCount: $("picker-count"),
  exportJson: $("export-json-btn"),
  stage: $("stage"),
  spikes: $("spikes"),
  spikeTitle: $("spike-title"),
  playBtn: $("play-btn"),
  stepBack: $("step-back-btn"),
  stepFwd: $("step-fwd-btn"),
  scrub: $("scrub"),
  allToggle: $("all-toggle"),
  cmdUnit: $("cmd-unit"),
  readout: $("readout"),
  readoutNote: $("readout-note"),
  tableMetrics: $("table-metrics"),
  tableEstimator: $("table-estimator"),
  tableEnv: $("table-env"),
  tableConfig: $("table-config"),
  tableRobust: $("table-robust"),
  tableSpikes: $("table-spikes"),
};

const stage = new Stage(el.stage);
const raster = new SpikeRaster(el.spikes);
const lanes = {
  pos: new LaneChart($("lane-pos")),
  vel: new LaneChart($("lane-vel")),
  track: new LaneChart($("lane-track")),
  est: new LaneChart($("lane-est")),
  cmd: new LaneChart($("lane-cmd")),
  dist: new LaneChart($("lane-dist")),
  innov: new LaneChart($("lane-innov")),
  cov: new LaneChart($("lane-cov")),
};

const state = {
  catalogue: null,
  examples: {},
  envPresets: {},
  example: "ball",
  envPreset: "clean",
  seed: 42,
  steps: 500,
  controllers: [ANN, SNN],
  payload: null,
  reference: null,
  T: 0,
  dt: 0.02,
  dragging: false,
  robustness: null,
};

// ------------------------------------------------------------------ helpers
function param(name) {
  return new URLSearchParams(window.location.search).get(name);
}
function syncURL() {
  const params = new URLSearchParams();
  params.set("example", state.example);
  params.set("env", state.envPreset);
  params.set("seed", String(state.seed));
  params.set("steps", String(state.steps));
  params.set("controllers", state.controllers.join(","));
  const next = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState(null, "", next);
}
function exampleSpec() {
  return state.examples[state.example] || {};
}
function exampleDefaults() {
  const d = exampleSpec().defaults || {};
  return { radius: d.radius ?? 0.15, freq: d.freq ?? 0.5 };
}
function exampleExtent() {
  const hi = exampleSpec().bounds_high || [];
  const e = hi.length ? Math.max(...hi.map((v) => Math.abs(v))) : 0.25;
  return e || 0.25;
}
function axisNames(d) {
  return ["x", "y", "z"].slice(0, d);
}
function color(name) {
  if (CTRL_COLORS[name]) return CTRL_COLORS[name];
  // stable across selection changes: keyed on the catalogue order
  const catalog = (state.catalogue && state.catalogue.catalog) || [];
  const idx = catalog.findIndex((c) => c.name === name);
  return PALETTE[(idx < 0 ? 0 : idx) % PALETTE.length];
}
function label(name) {
  return (state.catalogue && state.catalogue[name]?.label) || name;
}
function fmt(v, digits = 3) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(digits);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}
function showBanner(html) {
  el.banner.innerHTML = html;
  el.banner.classList.remove("hidden");
}
function hideBanner() {
  el.banner.classList.add("hidden");
  el.banner.innerHTML = "";
}
function setPill(node, text, cls) {
  node.textContent = text;
  node.className = `pill ${cls}`;
}
function shortKind(kind) {
  return { altitude: "BARO", flow_velocity: "FLOW", position_fix: "FIX", imu: "IMU" }[kind] || kind;
}
/** HTML table from a matrix: `rows[0]` is the header when `head` is true. */
function tableHtml(rows, { head = true, cls = "xtable" } = {}) {
  if (!rows || !rows.length) return '<div class="x-empty">no data</div>';
  return `<table class="${cls}">` + rows.map((cells, i) => {
    const tag = head && i === 0 ? "th" : "td";
    const body = cells.map((c, j) => {
      const isNum = i > 0 && j > 0;
      return `<${tag}${isNum ? ' class="num mono"' : ""}>${c}</${tag}>`;
    }).join("");
    return `<tr>${body}</tr>`;
  }).join("") + "</table>";
}

// ------------------------------------------------------------------- setup
function readQuery() {
  state.example = param("example") || "ball";
  state.envPreset = param("env") || "clean";
  state.seed = Number(param("seed") ?? 42) || 42;
  state.steps = Number(param("steps") ?? 500) || 500;
  const ctrls = param("controllers");
  if (ctrls) state.controllers = ctrls.split(",").map((s) => s.trim()).filter(Boolean);
}

function populateControls() {
  const cat = state.catalogue;
  const examples = cat.examples || [];
  state.examples = Object.fromEntries(examples.map((e) => [e.name, e]));
  if (!state.examples[state.example] && examples.length) state.example = cat.default_example || examples[0].name;

  el.exampleSelect.innerHTML = examples
    .map((e) => `<option value="${escapeHtml(e.name)}">${escapeHtml(e.label)}</option>`).join("");
  el.exampleSelect.value = state.example;
  el.exampleSelect.disabled = false;

  const presets = cat.embodiment_presets || {};
  const order = ["clean", "noisy", "delayed", "perturbed", "heavy", "embodied", "randomized"];
  const names = order.filter((n) => n in presets)
    .concat(Object.keys(presets).filter((n) => !order.includes(n)));
  el.envSelect.innerHTML = names.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join("");
  if (!names.includes(state.envPreset)) state.envPreset = names[0] || "clean";
  el.envSelect.value = state.envPreset;
  el.envSelect.disabled = false;

  el.seedInput.value = String(state.seed);
  el.stepsInput.value = String(state.steps);

  const catalog = cat.catalog || [];
  const known = catalog.map((c) => c.name);
  state.controllers = state.controllers.filter((c) => known.includes(c));
  if (!state.controllers.length) state.controllers = [ANN, SNN].filter((c) => known.includes(c));
  el.pickerBody.innerHTML = catalog.map((c) => {
    const checked = state.controllers.includes(c.name);
    return `<label class="x-check" title="${escapeHtml(c.description || c.label || "")}">
      <input type="checkbox" value="${escapeHtml(c.name)}"${checked ? " checked" : ""} />
      <span class="x-chip" style="background:${color(c.name)}"></span>${escapeHtml(c.label || c.name)}</label>`;
  }).join("");
  el.pickerBody.querySelectorAll("input[type=checkbox]").forEach((box) => {
    box.addEventListener("change", () => {
      const on = [...el.pickerBody.querySelectorAll("input:checked")].map((b) => b.value);
      if (!on.length) { box.checked = true; return; }
      if (on.length > MAX_CONTROLLERS) {
        box.checked = false;
        showBanner(`<strong>At most ${MAX_CONTROLLERS} controllers.</strong> Uncheck one first.`);
        return;
      }
      hideBanner();
      state.controllers = on;
      applyPickerCap();
      syncURL();
      load();
    });
  });
  applyPickerCap();
}

function applyPickerCap() {
  const boxes = [...el.pickerBody.querySelectorAll("input[type=checkbox]")];
  const checked = boxes.filter((b) => b.checked).length;
  boxes.forEach((b) => { if (!b.checked) b.disabled = checked >= MAX_CONTROLLERS; });
  el.pickerCount.textContent = `${checked} of ${boxes.length} selected`;
}

// -------------------------------------------------------------------- load
async function load() {
  setPill(el.apiPill, "loading…", "pill-idle");
  hideBanner();
  const { radius, freq } = exampleDefaults();
  try {
    const body = await api.benchmark({
      controllers: state.controllers,
      steps: state.steps,
      seed: state.seed,
      radius,
      freq,
      include_trace: true,
      trace_level: "full",
      spike_format: "events",
      embodiment: state.envPreset,
      example: state.example,
    });
    state.payload = body;
    state.reference = body.reference || null;
    state.dt = body.reference?.dt ?? 0.02;
    const first = body.results[state.controllers[0]] || {};
    state.T = Math.max(...state.controllers.map((c) => body.results[c]?.trajectory?.length || 0));
    if (!state.T) throw new Error("empty trajectory");

    buildStage();
    buildLanes();
    renderSpikeRaster();
    renderTables();
    el.scrub.max = String(Math.max(0, state.T - 1));
    el.scrub.disabled = false;
    [el.playBtn, el.stepBack, el.stepFwd].forEach((b) => (b.disabled = false));
    el.exportJson.disabled = false;
    el.cmdUnit.textContent = `[${exampleSpec().units?.command || ""}]`;
    setPill(el.trainedPill, state.catalogue.trained ? "trained" : "untrained",
      state.catalogue.trained ? "pill-ok" : "pill-idle");
    setPill(el.apiPill, "API ok", "pill-ok");
    syncURL();
    applyFrame(0);
    stage.setLoop(true);
    stage.play();
    updatePlayIcon();
    loadRobustness();
  } catch (err) {
    setPill(el.apiPill, "API unreachable", "pill-bad");
    const base = api.API_BASE || window.location.origin;
    showBanner(`<strong>Could not load ${base}/api/benchmark.</strong> ${escapeHtml(err.message || "")}`);
  }
}

function selected() {
  return state.controllers.map((c) => [c, state.payload.results[c] || {}]);
}

// ------------------------------------------------------------------- stage
function buildStage() {
  const spec = exampleSpec();
  const fixes = [];
  const series = selected().map(([name, r]) => {
    (r.fix_events || []).forEach((e) => fixes.push(e.pos));
    return {
      name,
      color: color(name),
      trajectory: r.trajectory || [],
      estimates: r.estimates,
      ring: name === SNN,
    };
  });
  const measurements = (state.payload.results[SNN] || {}).measurements
    || (selected()[0]?.[1] || {}).measurements || null;
  stage.setScene({
    reference: state.reference,
    series,
    plateHalf: exampleExtent(),
    measurements,
    renderer: spec.renderer || "plate",
    successLabel: spec.labels?.success || "IN BOUNDS",
    boundsHigh: spec.bounds_high || null,
    fixes,
  });
}

// ------------------------------------------------------------------- lanes
function wrapFor(key) {
  return $(`wrap-${key}`);
}
function sizeLane(key, rowCount) {
  const wrap = wrapFor(key);
  if (wrap) wrap.style.height = `${Math.max(64, rowCount * 46 + 30)}px`;
}

function buildLanes() {
  const spec = exampleSpec();
  const d = spec.pos_dim || 2;
  const axes = axisNames(d);
  const verts = state.T;
  const ref = state.reference || { pos: [], vel: [], acc: [] };
  const rref = (i, a) => (ref.pos?.[i] ? ref.pos[i][a] : null);

  // ---- position / velocity / errors / command / disturbance / covariance ---
  const posRows = axes.map((a, ai) => ({
    label: `p${a}`, unit: "m", zeroLine: false,
    series: [{ label: "r", color: REF_COLOR, values: Array.from({ length: verts }, (_, i) => rref(i, ai)), dash: true }]
      .concat(selected().flatMap(([name, r]) => ([
        { label: name, color: color(name), values: (r.trajectory || []).map((row) => row[ai]) },
        { label: "x̂", color: color(name), values: (r.estimates || []).map((row) => row[ai]), dash: true, width: 1.2 },
      ]))),
  }));
  sizeLane("pos", posRows.length);
  lanes.pos.setData({ rows: posRows, tMax: state.T * state.dt });

  const velRows = axes.map((a, ai) => ({
    label: `v${a}`, unit: "m/s", zeroLine: true,
    series: [{ label: "ṙ", color: REF_COLOR, values: Array.from({ length: verts }, (_, i) => (ref.vel?.[i] ? ref.vel[i][ai] : null)), dash: true }]
      .concat(selected().flatMap(([name, r]) => ([
        { label: name, color: color(name), values: (r.trajectory || []).map((row) => row[d + ai]) },
        { label: "x̂", color: color(name), values: (r.estimates || []).map((row) => row[d + ai]), dash: true, width: 1.2 },
      ]))),
  }));
  sizeLane("vel", velRows.length);
  lanes.vel.setData({ rows: velRows, tMax: state.T * state.dt });

  const trackRows = axes.map((a, ai) => ({
    label: `e${a}`, unit: "cm", zeroLine: true,
    series: selected().map(([name, r]) => ({
      label: name, color: color(name),
      values: Array.from({ length: verts }, (_, i) => {
        const rp = ref.pos?.[i]?.[ai];
        const p = r.trajectory?.[i]?.[ai];
        return rp == null || p == null ? null : (rp - p) * 100;
      }),
    })),
  }));
  sizeLane("track", trackRows.length);
  lanes.track.setData({ rows: trackRows, tMax: state.T * state.dt });

  const estRows = axes.map((a, ai) => ({
    label: `ê${a}`, unit: "cm", zeroLine: true,
    series: selected().map(([name, r]) => ({
      label: name, color: color(name),
      values: Array.from({ length: verts }, (_, i) => {
        const e = r.estimates?.[i]?.[ai];
        const p = r.trajectory?.[i]?.[ai];
        return e == null || p == null ? null : (e - p) * 100;
      }),
    })),
  }));
  sizeLane("est", estRows.length);
  lanes.est.setData({ rows: estRows, tMax: state.T * state.dt });

  const cmdRows = axes.map((a, ai) => ({
    label: `u${a}`, unit: spec.units?.command || "", zeroLine: true,
    series: selected().flatMap(([name, r]) => ([
      { label: name, color: color(name), values: (r.tilts || []).map((row) => row[ai]) },
      {
        label: "applied", color: color(name), width: 1.1,
        values: (r.applied || []).map((row) => (row ? row[ai] : null)), dash: true,
      },
    ])),
  }));
  sizeLane("cmd", cmdRows.length);
  lanes.cmd.setData({ rows: cmdRows, tMax: state.T * state.dt });

  const distRows = axes.map((a, ai) => ({
    label: `d${a}`, unit: "m/s²", zeroLine: true,
    series: [{ label: "r̈", color: REF_COLOR, values: Array.from({ length: verts }, (_, i) => (ref.acc?.[i] ? ref.acc[i][ai] : null)), dash: true }]
      .concat(selected().map(([name, r]) => ({
        label: name, color: DIST_COLOR,
        values: (r.disturbances || []).map((row) => (row ? row[ai] : null)),
      }))),
    markers: selected().flatMap(([, r]) => r.impulse_frames || []).length
      ? { frames: [...new Set(selected().flatMap(([, r]) => r.impulse_frames || []))], color: "#e4572e" }
      : null,
  }));
  sizeLane("dist", distRows.length);
  lanes.dist.setData({ rows: distRows, tMax: state.T * state.dt });

  // ---- innovations: one row per sensor-channel component --------------------
  const innovDefs = channelComponents(spec, d);
  const innovRows = innovDefs.map((def) => ({
    label: def.label, unit: def.unit, zeroLine: true,
    series: selected().map(([name, r]) => ({
      label: name, color: color(name),
      values: Array.from({ length: verts }, (_, i) => {
        const series = r.innovations?.[def.kind];
        const v = series?.[i];
        return v == null ? null : v[def.index];
      }),
    })),
  }));
  sizeLane("innov", Math.max(1, innovRows.length));
  lanes.innov.setData({ rows: innovRows, tMax: state.T * state.dt });

  // ---- covariance σ per state component ------------------------------------
  const covLabels = axes.concat(axes.map((a) => `v${a}`));
  const covRows = covLabels.map((lbl, ci) => ({
    label: `σ ${lbl}`, unit: "", zeroLine: false, yMin: 0,
    series: selected().map(([name, r]) => ({
      label: name, color: color(name),
      values: (r.covariance_diag || []).map((row) => (row == null || row[ci] == null ? null : Math.sqrt(row[ci]))),
    })),
  }));
  sizeLane("cov", covRows.length);
  lanes.cov.setData({ rows: covRows, tMax: state.T * state.dt });
}

/** One entry per component of every sensor channel that produces innovations. */
function channelComponents(spec, d) {
  const pos = axisNames(d);
  const vel = pos.map((p) => `v${p}`);
  const out = [];
  for (const ch of (spec.sensor && spec.sensor.channels) || []) {
    if (ch.kind === "imu") continue;                 // prediction input, no residual
    const unit = ch.kind === "flow_velocity" ? "m/s" : "m";
    ch.axes.forEach((axis, index) => {
      const name = axis < d ? pos[axis] : vel[axis - d];
      out.push({ kind: ch.kind, index, label: `${shortKind(ch.kind)} ${name}`, unit });
    });
  }
  return out;
}

// ------------------------------------------------------------------ spikes
function spikingController() {
  const withSpikes = selected().find(([, r]) => r.spikes);
  return withSpikes ? withSpikes[0] : (state.controllers.includes(SNN) ? SNN : null);
}
function renderSpikeRaster() {
  const name = spikingController();
  const r = name ? state.payload.results[name] : null;
  const spikes = r?.spikes || null;
  raster.setData({ spikes, T: state.T, maxNeurons: 200, label: name || "SNN" });
  const total = spikes?.shape?.[1] ?? state.catalogue?.n_neurons ?? 0;
  el.spikeTitle.innerHTML = spikes
    ? `Spike activity <span class="unit">${escapeHtml(label(name))} · first ${raster.N} of ${total} neurons</span>`
    : "Spike activity";
}

// --------------------------------------------------------------- readout
function readoutSpec(d) {
  const pos = axisNames(d);
  const rows = [];
  rows.push({ label: "t [s]", digits: 3, get: () => state.frameK * state.dt });
  rows.push({ label: "in bounds", get: (r) => (r.success ? (r.success[state.frameK] ? "yes" : "no") : "—") });
  pos.forEach((a, ai) => {
    rows.push({ label: `p_${a} [m]`, digits: 4, get: (r) => r.trajectory?.[state.frameK]?.[ai] });
    rows.push({ label: `r_${a} [m]`, digits: 4, get: () => state.reference?.pos?.[state.frameK]?.[ai] });
    rows.push({ label: `e_${a} [cm]`, digits: 3, get: (r) => {
      const rp = state.reference?.pos?.[state.frameK]?.[ai];
      const p = r.trajectory?.[state.frameK]?.[ai];
      return rp == null || p == null ? null : (rp - p) * 100;
    } });
    rows.push({ label: `x̂_${a} [m]`, digits: 4, get: (r) => r.estimates?.[state.frameK]?.[ai] });
    rows.push({ label: `ê_${a} [cm]`, digits: 3, get: (r) => {
      const e = r.estimates?.[state.frameK]?.[ai];
      const p = r.trajectory?.[state.frameK]?.[ai];
      return e == null || p == null ? null : (e - p) * 100;
    } });
  });
  pos.forEach((a, ai) => {
    rows.push({ label: `v_${a} [m/s]`, digits: 4, get: (r) => r.trajectory?.[state.frameK]?.[d + ai] });
    rows.push({ label: `v̂_${a} [m/s]`, digits: 4, get: (r) => r.estimates?.[state.frameK]?.[d + ai] });
  });
  pos.forEach((a, ai) => {
    rows.push({ label: `u_${a} cmd`, digits: 4, get: (r) => r.tilts?.[state.frameK]?.[ai] });
    rows.push({ label: `u_${a} applied`, digits: 4, get: (r) => r.applied?.[state.frameK]?.[ai] });
  });
  pos.concat(pos.map((a) => `v${a}`)).forEach((a, ci) => {
    rows.push({ label: `σ_${a}`, digits: 5, get: (r) => {
      const c = r.covariance_diag?.[state.frameK];
      return c == null ? null : Math.sqrt(c[ci]);
    } });
  });
  for (const def of channelComponents(exampleSpec(), d)) {
    rows.push({ label: `innov ${def.label}`, digits: 5, get: (r) => {
      const v = r.innovations?.[def.kind]?.[state.frameK];
      return v == null ? null : v[def.index];
    } });
  }
  return rows;
}

function renderReadout() {
  const spec = exampleSpec();
  const d = spec.pos_dim || 2;
  const rows = readoutSpec(d);
  const cells = [["quantity", ...state.controllers.map((c) => label(c))]];
  for (const row of rows) {
    cells.push([
      escapeHtml(row.label),
      ...selected().map(([, r]) => {
        const v = row.get(r);
        if (v == null) return "—";
        if (typeof v === "number") return fmt(v, row.digits ?? 3);
        return escapeHtml(String(v));
      }),
    ]);
  }
  el.readout.innerHTML = tableHtml(cells);
  el.readoutNote.textContent = `frame ${state.frameK} · ${state.T - 1}`;
}

// ---------------------------------------------------------------- tables
function renderTables() {
  renderMetrics();
  renderEstimator();
  renderEnv();
  renderConfig();
  renderSpikeStats();
}

function renderMetrics() {
  const defs = [
    ["mean error [cm]", (r) => fmt(r.metrics?.mean_error_cm, 3)],
    ["rms error [cm]", (r) => fmt(r.metrics?.rms_error_cm, 3)],
    ["max error [cm]", (r) => fmt(r.metrics?.max_error_cm, 3)],
    ["final error [cm]", (r) => fmt(r.metrics?.final_error_cm, 3)],
    ["settling step", (r) => (r.metrics?.settling_step == null ? "—" : String(r.metrics.settling_step))],
    ["in bounds [%]", (r) => fmt(r.on_plate_pct, 1)],
    ["x̂ RMSE [cm]", (r) => fmt(r.metrics?.estimation_pos_rmse_cm, 3)],
    ["x̂ RMSE per axis [cm]", (r) => (r.metrics?.estimation_pos_rmse_cm_axes || []).map((v) => fmt(v, 2)).join(" / ") || "—"],
    ["x̂ max drift [cm]", (r) => fmt(r.metrics?.estimation_max_pos_err_cm, 3)],
    ["first fix frame", (r) => (r.metrics?.estimation_first_fix_step == null ? "—" : String(r.metrics.estimation_first_fix_step))],
    ["impulses", (r) => (r.metrics?.impulse_count == null ? "—" : String(r.metrics.impulse_count))],
    ["max recovery [steps]", (r) => (r.metrics?.max_recovery_step == null ? "—" : String(r.metrics.max_recovery_step))],
    ["disturbance RMS", (r) => fmt(r.metrics?.disturbance_rms, 4)],
    ["trained", (r) => (r.trained ? "yes" : "no")],
    ["spiking", (r) => (r.spiking ? "yes" : "no")],
    ["sample counts", (r) => Object.entries(r.estimator?.measurement_counts || {}).map(([k, v]) => `${shortKind(k)} ${v}`).join(" · ") || "—"],
  ];
  const rows = [["metric", ...state.controllers.map((c) => label(c))]];
  for (const [name, get] of defs) rows.push([escapeHtml(name), ...selected().map(([, r]) => get(r))]);
  el.tableMetrics.innerHTML = tableHtml(rows);
}

function renderEstimator() {
  const cells = [["field", ...state.controllers.map((c) => label(c))]];
  const fields = [
    ["description", (r) => r.estimator?.description],
    ["measurement", (r) => r.estimator?.measurement],
    ["multi-channel", (r) => (r.estimator?.multi_channel ? "yes" : "no")],
    ["seeded from launch", (r) => (r.estimator?.seeded_from_launch ? "yes" : "no")],
    ["process noise", (r) => fmt(r.estimator?.process_noise, 4)],
    ["assumed meas noise", (r) => fmt(r.estimator?.meas_noise, 6)],
    ["configured delay", (r) => fmt(r.estimator?.delay, 0)],
    ["fix frames", (r) => String((r.estimator?.fix_frames || []).length)],
  ];
  for (const [name, get] of fields) cells.push([escapeHtml(name), ...selected().map(([, r]) => escapeHtml(get(r) ?? "—"))]);
  let html = tableHtml(cells);

  // resolved channel table (one block per controller)
  for (const [name, r] of selected()) {
    const channels = r.estimator?.sensor?.channels || [];
    if (!channels.length) continue;
    const rows = [["channel", "axes", "σ", "latency [frames]", "gate [m]"]];
    for (const ch of channels) {
      rows.push([
        `${escapeHtml(shortKind(ch.kind))}`,
        ch.axes.join(","),
        fmt(ch.sigma, 4),
        String(ch.latency),
        ch.gate_range == null ? "—" : fmt(ch.gate_range, 2),
      ]);
    }
    html += `<div class="x-sub">${escapeHtml(label(name))} channels</div>` + tableHtml(rows);
    html += `<div class="x-empty">innovations recorded: ${
      Object.entries(r.estimator?.measurement_counts || {}).map(([k, v]) => `${shortKind(k)} ${v}`).join(" · ") || "none"
    }</div>`;
  }
  el.tableEstimator.innerHTML = html;
}

function renderEnv() {
  const cells = [["field", ...state.controllers.map((c) => label(c))]];
  const fields = [
    ["preset", (r) => r.env?.preset],
    ["clean", (r) => (r.env?.clean ? "yes" : "no")],
    ["sensor channels", (r) => ((r.env?.sensor?.channels || []).map((c) => shortKind(c.kind)).join(" · ") || "camera")],
    ["noise scale", (r) => fmt(r.env?.sensor_noise_scale, 2)],
    ["extra sensor delay", (r) => fmt(r.env?.sensor_delay, 0)],
    ["actuator gain / bias", (r) => `${fmt(r.env?.actuator_gain, 3)} / ${fmt(r.env?.actuator_bias, 3)}`],
    ["actuator delay", (r) => fmt(r.env?.actuator_delay, 0)],
    ["damping", (r) => fmt(r.env?.damping, 3)],
    ["c scale", (r) => fmt(r.env?.c_scale, 3)],
    ["process noise", (r) => fmt(r.env?.process_noise, 4)],
    ["impulse interval / std", (r) => `${fmt(r.env?.impulse_interval, 0)} / ${fmt(r.env?.impulse_std, 3)}`],
    ["impulses fired", (r) => fmt(r.env?.impulse_count, 0)],
    ["checkpoint fixes", (r) => fmt(r.env?.fix_count, 0)],
    ["randomized", (r) => (r.env?.randomize ? "yes" : "no")],
    ["seed", (r) => fmt(r.env?.seed, 0)],
  ];
  for (const [name, get] of fields) cells.push([escapeHtml(name), ...selected().map(([, r]) => escapeHtml(String(get(r) ?? "—")))]);
  el.tableEnv.innerHTML = tableHtml(cells);
}

function renderConfig() {
  const cat = state.catalogue || {};
  const cfg = state.payload.config || {};
  const ref = state.reference || {};
  const rows = [
    ["field", "value"],
    ["example", escapeHtml(state.example)],
    ["label", escapeHtml(exampleSpec().label || "")],
    ["environment", escapeHtml(state.envPreset)],
    ["seed", String(state.seed)],
    ["steps", String(state.steps)],
    ["dt [s]", fmt(ref.dt, 4)],
    ["trajectory", `${escapeHtml(ref.kind || "—")} · radius ${fmt(ref.radius, 3)} · freq ${fmt(ref.freq, 3)}`],
    ["trace level", escapeHtml(cfg.trace_level || "short")],
    ["include trace", cfg.include_trace === false ? "no" : "yes"],
    ["record spikes", cfg.record_spikes === false ? "no" : "yes"],
    ["pos dim", String(exampleSpec().pos_dim ?? "—")],
    ["policy dims", `${exampleSpec().n_in ?? "—"} → ${exampleSpec().n_out ?? "—"}`],
    ["engine version", escapeHtml(cat.engine_version || "—")],
    ["torch", escapeHtml(cat.torch_version || "—")],
    ["neurons", String(cat.n_neurons ?? "—")],
    ["micro steps", String(cat.micro_steps ?? "—")],
    ["weights", cat.trained ? "trained bundle loaded" : "untrained"],
  ];
  el.tableConfig.innerHTML = tableHtml(rows);
}

function renderSpikeStats() {
  const name = spikingController();
  const r = name ? state.payload.results[name] : null;
  const spikes = r?.spikes;
  if (!spikes || spikes.format !== "events") {
    el.tableSpikes.innerHTML = '<div class="x-empty">no spiking controller selected</div>';
    return;
  }
  const [T, N] = spikes.shape || [state.T, 0];
  const events = spikes.data || [];
  const counts = new Array(N).fill(0);
  for (const [k, n] of events) if (n < N) counts[n] += 1;
  const active = counts.filter((c) => c > 0).length;
  const top = counts.map((c, n) => [c, n]).sort((a, b) => b[0] - a[0]).slice(0, 8);
  const rate = T && state.dt ? events.length / (T * state.dt) : 0;
  const rows = [
    ["field", "value"],
    ["controller", escapeHtml(label(name))],
    ["neurons", String(N)],
    ["frames", String(T)],
    ["total spikes", String(events.length)],
    ["active neurons", `${active} (${fmt(100 * active / Math.max(1, N), 1)} %)`],
    ["mean rate [Hz]", fmt(rate, 1)],
    ["per-active-neuron [Hz]", fmt(rate / Math.max(1, active), 2)],
    ["busiest neuron", top.length ? `#${top[0][1]} · ${top[0][0]} spikes` : "—"],
  ];
  let html = tableHtml(rows);
  if (top.some(([c]) => c > 0)) {
    html += '<div class="x-sub">busiest neurons</div>' +
      tableHtml([["neuron", "spikes"], ...top.filter(([c]) => c > 0).map(([c, n]) => [`#${n}`, String(c)])]);
  }
  el.tableSpikes.innerHTML = html;
}

async function loadRobustness() {
  try {
    const res = await api.robustness({
      controllers: state.controllers.slice(0, 3),
      axis: "preset",
      steps: 120,
      seed: state.seed,
      example: state.example,
    });
    state.robustness = res;
    const rows = [["preset", ...state.controllers.slice(0, 3).map((c) => label(c))]];
    for (const cell of res.cells || []) {
      rows.push([
        escapeHtml(String(cell.point)),
        ...state.controllers.slice(0, 3).map((c) => fmt(cell.per_controller?.[c]?.mean_error_cm, 2)),
      ]);
    }
    el.tableRobust.innerHTML = tableHtml(rows);
  } catch {
    el.tableRobust.innerHTML = '<div class="x-empty">robustness unavailable</div>';
  }
}

// ----------------------------------------------------------------- cursor
function applyFrame(k) {
  state.frameK = Math.max(0, Math.min(k, Math.max(0, state.T - 1)));
  state.lastFrame = state.frameK;
  Object.values(lanes).forEach((lane) => lane.setFrame(state.frameK));
  raster.setFrame(state.frameK);
  el.clock.textContent = `t = ${(state.frameK * state.dt).toFixed(2)} s`;
  el.frameReadout.textContent = `frame ${state.frameK} / ${Math.max(0, state.T - 1)}`;
  if (!state.dragging) el.scrub.value = String(state.frameK);
  const primary = state.payload?.results[state.controllers.includes(SNN) ? SNN : state.controllers[0]] || {};
  const ok = primary.success ? primary.success[state.frameK] : true;
  setPill(el.successPill, ok ? `● ${exampleSpec().labels?.success || "IN BOUNDS"}` : "● OUT OF BOUNDS",
    ok ? "pill-ok" : "pill-bad");
  renderReadout();
  updatePlayIcon();
  setPill(el.runPill, stage.playing ? "● PLAYING" : "○ PAUSED", stage.playing ? "pill-ok" : "pill-idle");
}
function updatePlayIcon() {
  el.playBtn.textContent = stage.playing ? "❚❚" : "▶";
}

// ------------------------------------------------------------- listeners
el.playBtn.addEventListener("click", () => { stage.toggle(); updatePlayIcon(); });
el.stepBack.addEventListener("click", () => { stage.pause(); stage.setFrame(state.frameK - 1); });
el.stepFwd.addEventListener("click", () => { stage.pause(); stage.setFrame(state.frameK + 1); });
el.scrub.addEventListener("input", () => {
  state.dragging = true;
  stage.pause();
  stage.setFrame(Number(el.scrub.value));
});
el.scrub.addEventListener("change", () => { state.dragging = false; });
el.allToggle.addEventListener("change", () => {
  Object.values(lanes).forEach((lane) => { lane.showAll = el.allToggle.checked; });
});
el.exampleSelect.addEventListener("change", () => { state.example = el.exampleSelect.value; syncURL(); load(); });
el.envSelect.addEventListener("change", () => { state.envPreset = el.envSelect.value; syncURL(); load(); });
el.seedInput.addEventListener("change", () => {
  state.seed = Math.max(0, Number(el.seedInput.value) || 0); syncURL(); load();
});
el.stepsInput.addEventListener("change", () => {
  state.steps = Math.min(2000, Math.max(10, Number(el.stepsInput.value) || 500)); syncURL(); load();
});
el.exportJson.addEventListener("click", () => {
  if (!state.payload) return;
  const blob = new Blob([JSON.stringify(state.payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ann2snn_extended_${state.example}_seed${state.seed}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
});
stage.onFrame = (k) => applyFrame(k);

// ---------------------------------------------------------------- bootstrap
(async function boot() {
  try {
    state.catalogue = await api.fetchControllers();
    const cat = state.catalogue;
    state.examples = Object.fromEntries((cat.examples || []).map((e) => [e.name, e]));
    readQuery();
    populateControls();
    await load();
  } catch (err) {
    setPill(el.apiPill, "API unreachable", "pill-bad");
    showBanner(`<strong>Cannot reach the API.</strong> ${escapeHtml(err.message || "")}`);
  }
})();
