// ANN2SNN — "ANN → SNN closed-loop control experiment" dashboard.
//
// One POST /api/benchmark runs all five controllers on the same orbit. The hero
// stage shows only the experiment pair (fly-like ANN and the transferred SNN)
// plus the target orbit; the side column shows the controller pipeline, the SNN
// spike activity, the control output and the tracking error; the footer reports
// the quantitative transfer result. A single Play/Pause drives the shared cursor.

import "./style.css";
import * as api from "./api.js";
import { Stage } from "./stage.js";
import { SpikeRaster } from "./raster.js";
import { SignalChart } from "./signals.js";
import { Pipeline } from "./pipeline.js";
import { StageRecorder, startDisplayRecording, canvasRecordingSupported } from "./record.js";

const $ = (id) => document.getElementById(id);

const ANN = "flylike_ann";
const SNN = "snn_transferred";
const ANN_COLOR = "#ffb454";
const SNN_COLOR = "#7ee0c0";
const BENCH = { steps: 500, seed: 42, radius: 0.15, freq: 0.5 };

const el = {
  runPill: $("run-pill"),
  spikingPill: $("spiking-pill"),
  apiPill: $("api-pill"),
  trainedPill: $("trained-pill"),
  capturePill: $("capture-pill"),
  envSelect: $("env-select"),
  profilePill: $("profile-pill"),
  robustStrip: $("robust-strip"),
  clock: $("clock"),
  neuronCount: $("neuron-count"),
  frameCount: $("frame-count"),
  banner: $("error-banner"),
  stage: $("stage"),
  spikes: $("spikes"),
  controlOut: $("control-out"),
  tracking: $("tracking"),
  spikeTitle: $("spike-title"),
  controllerPipeline: $("controller-pipeline"),
  playBtn: $("play-btn"),
  recordBtn: $("record-btn"),
  recordTabBtn: $("record-tab-btn"),
  exportBtn: $("export-btn"),
  resultAnn: $("result-ann"),
  resultSnn: $("result-snn"),
  resultDelta: $("result-delta"),
  resultNote: $("result-note"),
};

const stage = new Stage(el.stage);
const raster = new SpikeRaster(el.spikes);
const controlChart = new SignalChart(el.controlOut);
const trackingChart = new SignalChart(el.tracking);
const pipeline = new Pipeline(el.controllerPipeline, { onPhase: () => updateStatus() });
const recorder = new StageRecorder(stage, { filename: "ann2snn.webm", onState: onCaptureState });

const state = {
  catalog: {},
  results: {},
  stats: null,
  reference: null,
  lastFrame: 0,
  displayRecorder: null,
  envPreset: "embodied",
  profile: "robust",
  envSpecs: {},
};

// ------------------------------------------------------------------ helpers
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
function fmt(v, digits = 2) {
  return v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(digits);
}
function signed(v, digits = 2) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(digits)}`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function metaFor(name) {
  return state.catalog[name] || {};
}
function updatePlayIcon() {
  el.playBtn.textContent = stage.playing ? "❚❚" : "▶";
}
function updateStatus() {
  setPill(el.runPill, stage.playing ? "● RUNNING" : "○ PAUSED",
    stage.playing ? "pill-ok" : "pill-idle");
  const spiking = pipeline.spiking && stage.total > 0;
  setPill(el.spikingPill, spiking ? "● SPIKING" : "○ SPIKING",
    spiking ? "pill-ok" : "pill-idle");
}
function updateClock(k) {
  const dt = state.reference?.dt ?? 0.02;
  el.clock.textContent = `t = ${(k * dt).toFixed(2)} s`;
}

// ------------------------------------------------------------- environment
function populateEnvSelect(cat) {
  if (!el.envSelect || el.envSelect.dataset.ready === "1") return;
  const presets = cat.embodiment_presets || {};
  const order = ["clean", "noisy", "delayed", "perturbed", "heavy", "embodied", "randomized"];
  const names = order.filter((n) => n in presets)
    .concat(Object.keys(presets).filter((n) => !order.includes(n)));
  if (!names.length) return;
  el.envSelect.innerHTML = names
    .map((n) => `<option value="${n}">${escapeHtml(n)}</option>`)
    .join("");
  if (!names.includes(state.envPreset)) state.envPreset = names[0];
  el.envSelect.value = state.envPreset;
  el.envSelect.disabled = false;
  el.envSelect.dataset.ready = "1";
}
function applyProfile() {
  state.profile = state.envPreset === "clean" ? "clean" : "robust";
  setPill(el.profilePill, state.profile, state.profile === "robust" ? "pill-ok" : "pill-idle");
}
async function loadRobustness() {
  if (!el.robustStrip) return;
  try {
    const res = await api.robustness({
      controllers: [ANN, SNN], axis: "preset", steps: 120, seed: BENCH.seed,
    });
    el.robustStrip.innerHTML =
      '<span class="rb-head">ROBUSTNESS · mean cm (ANN / SNN)</span>' +
      res.cells
        .map((c) => {
          const a = c.per_controller?.[ANN]?.mean_error_cm;
          const s = c.per_controller?.[SNN]?.mean_error_cm;
          const sel = c.embodiment?.preset === state.envPreset ? " sel" : "";
          return `<span class="rb-cell${sel}"><span class="rb-name">${escapeHtml(String(c.point))}</span>` +
            `<b class="ann">${fmt(a, 1)}</b><b class="snn">${fmt(s, 1)}</b></span>`;
        })
        .join("");
  } catch {
    el.robustStrip.innerHTML = "";
  }
}

// -------------------------------------------------------------------- load
async function loadOnce() {
  setPill(el.apiPill, "connecting…", "pill-idle");
  hideBanner();
  try {
    const cat = await api.fetchControllers();
    state.catalog = Object.fromEntries((cat.catalog || []).map((c) => [c.name, c]));
    state.envSpecs = cat.embodiment_presets || {};
    populateEnvSelect(cat);
    applyProfile();

    const body = await api.benchmark({
      controllers: [ANN, SNN],
      steps: BENCH.steps,
      seed: BENCH.seed,
      radius: BENCH.radius,
      freq: BENCH.freq,
      include_trace: true,
      spike_format: "events",
      embodiment: state.envPreset,
      profile: state.profile,
    });
    state.results = body.results || {};
    state.stats = body.stats || null;
    state.reference = body.reference || null;

    const ann = state.results[ANN] || {};
    const snn = state.results[SNN] || {};
    const dt = body.reference?.dt ?? cat.dt ?? 0.02;
    const T = Math.max(ann.trajectory?.length || 0, snn.trajectory?.length || 0);
    const tMax = T * dt;

    // hero: the experiment pair only (ANN first, so the SNN ball draws on top)
    stage.setScene({
      reference: body.reference,
      series: [
        { name: ANN, color: ANN_COLOR, trajectory: ann.trajectory || [], estimates: ann.estimates },
        { name: SNN, color: SNN_COLOR, trajectory: snn.trajectory || [],
          estimates: snn.estimates, ring: true },
      ],
      plateHalf: body.stats?.plate_half_m ?? cat.plate_half ?? 0.25,
      measurements: snn.measurements || ann.measurements || null,
    });
    stage.setLoop(true);

    // spike activity (first 200 of N neurons)
    const spikes = snn.spikes;
    raster.setData({ spikes, T: snn.trajectory?.length || 0, maxNeurons: 200, label: "SNN" });
    raster.setFrame(0);
    const totalN = spikes?.shape?.[1] ?? cat.n_neurons ?? 0;
    el.spikeTitle.innerHTML = spikes
      ? `Spike activity <span class="unit">SNN · first ${raster.N} of ${totalN} neurons</span>`
      : "Spike activity";
    el.neuronCount.textContent = `${cat.n_neurons ?? totalN} neurons`;
    el.frameCount.textContent = `${BENCH.steps} frames`;

    // control output (SNN plate tilt) + tracking (ANN vs SNN error)
    controlChart.setData({
      series: [
        { label: "θx", color: "#4da3ff", values: (snn.tilts || []).map((t) => t[0]), dash: false },
        { label: "θy", color: "#dd8452", values: (snn.tilts || []).map((t) => t[1]), dash: true },
      ],
      zeroLine: true,
      tMax,
    });
    trackingChart.setData({
      series: [
        { label: "ANN", color: ANN_COLOR, values: ann.tracking_error || [], dash: false },
        { label: "SNN", color: SNN_COLOR, values: snn.tracking_error || [], dash: true },
      ],
      zeroLine: false,
      tMax,
    });
    controlChart.setFrame(0);
    trackingChart.setFrame(0);

    buildResultBar(ann, snn);

    setPill(el.trainedPill, cat.trained ? "trained" : "untrained", cat.trained ? "pill-ok" : "pill-idle");
    setPill(el.apiPill, "API ok", "pill-ok");
    el.playBtn.disabled = false;
    [el.recordBtn, el.recordTabBtn, el.exportBtn].forEach((b) => (b.disabled = false));
    if (!canvasRecordingSupported()) setPill(el.capturePill, "no capture", "pill-bad");

    updateClock(0);
    pipeline.replay();
    stage.play();
    updatePlayIcon();
    updateStatus();
    setTimeout(loadRobustness, 0);
  } catch (err) {
    setPill(el.apiPill, "API unreachable", "pill-bad");
    const base = api.API_BASE || window.location.origin;
    showBanner(
      `<strong>API unreachable.</strong> Could not load <code>${base}/api/benchmark</code>. ` +
        `Start the Flask server and reload.`
    );
  }
}

// ------------------------------------------------------------------ panels
function buildResultBar(ann, snn) {
  const stats = state.stats?.per_controller || {};
  const annErr = stats[ANN]?.mean_error_cm ?? ann.metrics?.mean_error_cm;
  const snnErr = stats[SNN]?.mean_error_cm ?? snn.metrics?.mean_error_cm;
  el.resultAnn.textContent = fmt(annErr);
  el.resultSnn.textContent = fmt(snnErr);
  const delta = (annErr != null && snnErr != null) ? snnErr - annErr : null;
  el.resultDelta.textContent = signed(delta);
  el.resultDelta.parentElement.title = "difference (SNN − ANN)";
  const est = stats[SNN]?.estimation_pos_rmse_cm;
  el.resultNote.textContent =
    `closed-loop radial tracking error · Δ = SNN − ANN · env ${state.envPreset} · policy ${state.profile}` +
    (est != null ? ` · x̂ RMSE ${fmt(est)} cm` : "");
}

// ---------------------------------------------------------------- capture
function onCaptureState(stateName, detail = "") {
  const cls = stateName === "recording" ? "pill-bad" : stateName === "done" ? "pill-ok" : "pill-idle";
  setPill(el.capturePill, detail ? `${stateName} · ${detail}` : stateName, cls);
  if (stateName === "done" || stateName === "error" || stateName === "unsupported") {
    stage.setLoop(true);
  }
}
function toggleRecord() {
  if (recorder.recording) {
    recorder.stop();
    return;
  }
  recorder.filename = `ann2snn_experiment_seed${BENCH.seed}.webm`;
  pipeline.replay();
  recorder.start();
}
async function toggleTabRecord() {
  if (state.displayRecorder) {
    state.displayRecorder.stop();
    state.displayRecorder = null;
    return;
  }
  const rec = await startDisplayRecording({
    filename: `ann2snn_tab_seed${BENCH.seed}.webm`,
    onState: onCaptureState,
  });
  if (rec) {
    state.displayRecorder = rec;
    stage.setFrame(0);
    stage.setLoop(false);
    pipeline.replay();
    stage.play();
    rec.addEventListener("stop", () => { state.displayRecorder = null; });
  }
}
async function exportMp4() {
  el.exportBtn.disabled = true;
  setPill(el.capturePill, "rendering MP4…", "pill-bad");
  try {
    const blob = await api.exportMp4({
      controller: SNN,
      seed: BENCH.seed,
      steps: BENCH.steps,
      radius: BENCH.radius,
      freq: BENCH.freq,
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ann2snn_${SNN}_seed${BENCH.seed}.mp4`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    setPill(el.capturePill, `MP4 ${(blob.size / 1e6).toFixed(1)} MB`, "pill-ok");
  } catch (err) {
    setPill(el.capturePill, "export failed", "pill-bad");
    showBanner(`<strong>MP4 export failed.</strong> ${escapeHtml(err.message || String(err))}`);
  } finally {
    el.exportBtn.disabled = false;
  }
}

// --------------------------------------------------------------- listeners
el.playBtn.addEventListener("click", () => {
  stage.toggle();
  updatePlayIcon();
  updateStatus();
});
el.recordBtn.addEventListener("click", toggleRecord);
el.recordTabBtn.addEventListener("click", toggleTabRecord);
el.exportBtn.addEventListener("click", exportMp4);
el.envSelect.addEventListener("change", () => {
  state.envPreset = el.envSelect.value;
  applyProfile();
  loadOnce();
});

stage.onFrame = (k) => {
  raster.setFrame(k);
  controlChart.setFrame(k);
  trackingChart.setFrame(k);
  updateClock(k);
  if (k === 0 && state.lastFrame > 0) pipeline.replay();   // capture the reveal each loop
  state.lastFrame = k;
  updatePlayIcon();
  updateStatus();
};

loadOnce();
