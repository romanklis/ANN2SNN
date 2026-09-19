// In-page video recording.
//
// Primary mode records the *stage canvas* (`canvas.captureStream(60)` +
// MediaRecorder) with no extra dependencies and no permissions: the result is a
// clean clip of the ball, the HUD, the error sparkline, the tilt bars and the
// spike raster. The secondary mode records the whole browser tab through
// `getDisplayMedia` (the browser asks for permission) so charts and controls are
// included too.

const MIME_CANDIDATES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
  "video/mp4",
];

export function pickMime() {
  if (typeof MediaRecorder === "undefined") return null;
  for (const m of MIME_CANDIDATES) {
    try {
      if (MediaRecorder.isTypeSupported(m)) return m;
    } catch {
      /* ignore */
    }
  }
  return "";
}

export function canvasRecordingSupported() {
  return (
    typeof MediaRecorder !== "undefined" &&
    typeof HTMLCanvasElement !== "undefined" &&
    !!HTMLCanvasElement.prototype.captureStream
  );
}

function extFor(mime) {
  return mime && mime.includes("mp4") ? "mp4" : "webm";
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

/**
 * Records the stage canvas while the animation runs.
 *
 * @param {Stage} stage
 * @param {object} opts {filename, fps, onState}
 */
export class StageRecorder {
  constructor(stage, { filename = "ann2snn.webm", fps = 60, onState = null } = {}) {
    this.stage = stage;
    this.filename = filename;
    this.fps = fps;
    this.onState = onState;
    this.recorder = null;
    this.chunks = [];
    this.recording = false;
    this._endedHandler = null;
  }

  get supported() {
    return canvasRecordingSupported();
  }

  _setState(state, detail = "") {
    if (this.onState) this.onState(state, detail);
  }

  start() {
    if (this.recording) return false;
    if (!this.supported) {
      this._setState("unsupported", "MediaRecorder/captureStream not available");
      return false;
    }
    const mime = pickMime();
    if (mime === null) {
      this._setState("unsupported", "MediaRecorder not available");
      return false;
    }
    let stream;
    try {
      stream = this.stage.canvas.captureStream(this.fps);
    } catch (err) {
      this._setState("error", String(err));
      return false;
    }

    this.chunks = [];
    try {
      this.recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    } catch (err) {
      this._setState("error", String(err));
      return false;
    }
    this.mime = this.recorder.mimeType || mime || "video/webm";

    this.recorder.ondataavailable = (e) => {
      if (e.data && e.data.size) this.chunks.push(e.data);
    };
    this.recorder.onstop = () => this._finish();
    this.recorder.onerror = (e) => this._setState("error", e.error ? String(e.error) : "recorder error");

    // Record exactly one pass from the start.
    this.stage.setFrame(0);
    this.stage.setLoop(false);
    this._endedHandler = () => setTimeout(() => this.stop(), 250);
    this.stage.onEnded = this._endedHandler;

    this.recorder.start(250);
    this.recording = true;
    this._setState("recording");
    this.stage.play();
    return true;
  }

  stop() {
    if (!this.recording || !this.recorder) return;
    this.stage.pause();
    try {
      this.recorder.stop();
    } catch {
      /* already stopped */
    }
    this.recording = false;
  }

  _finish() {
    const blob = new Blob(this.chunks, { type: this.mime || "video/webm" });
    this.chunks = [];
    this.recording = false;
    if (this.stage) this.stage.onEnded = null;
    this._setState("done", `${(blob.size / 1e6).toFixed(2)} MB`);
    downloadBlob(blob, this.filename);
  }
}

/**
 * Records the whole browser tab via `getDisplayMedia` (the page must be in the
 * shared surface). Returns the MediaRecorder or null if the user cancels.
 */
export async function startDisplayRecording({ filename = "ann2snn_tab.webm", onState = null } = {}) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
    if (onState) onState("unsupported", "getDisplayMedia not available");
    return null;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 30 },
      audio: false,
    });
  } catch (err) {
    if (onState) onState("cancelled", String(err));
    return null;
  }
  const mime = pickMime();
  const chunks = [];
  const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const stopAll = () => {
    stream.getTracks().forEach((t) => t.stop());
    if (onState) onState("done");
  };
  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size) chunks.push(e.data);
  };
  recorder.onstop = () => {
    const type = recorder.mimeType || mime || "video/webm";
    downloadBlob(new Blob(chunks, { type }), filename.replace(/\.webm$/, `.${extFor(type)}`));
    stopAll();
  };
  stream.getVideoTracks()[0].addEventListener("ended", () => recorder.stop());
  recorder.start(250);
  if (onState) onState("recording", "tab");
  return recorder;
}
