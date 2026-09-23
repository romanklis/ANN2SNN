// Controller-pipeline state machine.
//
// Purely cosmetic: it drives the `data-phase` attribute and the `.on` classes
// that CSS animates for the one-time reveal.  It never gates or alters the real
// benchmark trace.
//
// The phases are **data-driven**: every element carrying `data-phase-row="…"`
// inside the pipeline root is one phase, in DOM order.  That way the ball's
// `PLANT → CAMERA → KALMAN → POLICY` and the GPS-denied drone's
// `PLANT → SENSORS → KALMAN → POLICY` reveal the same way, with the sensor block
// holding one row per channel.

export const PHASES = ["plant", "sensors", "kalman", "policy"];

export class Pipeline {
  /** @param {HTMLElement} root @param {object} opts {onPhase} */
  constructor(root, { onPhase = null } = {}) {
    this.root = root;
    this.onPhase = onPhase;
    this._timers = [];
    this.rows = Array.from(root.querySelectorAll("[data-phase-row]"));
    this.names = this.rows.map((r) => r.dataset.phaseRow).filter(Boolean);
    if (!this.names.length) this.names = [...PHASES];
    this.phase = this.names[0];
    // Apply the initial state *without* firing onPhase: the caller's `onPhase`
    // may read the very object being constructed (TDZ) if invoked here.
    this._apply();
  }

  /** True once the controller is in the loop (steady state). */
  get spiking() {
    return this.phase === this.names[this.names.length - 1];
  }

  /** Paint the current phase onto the DOM (no callback). */
  _apply() {
    const idx = this.names.indexOf(this.phase);
    this.root.dataset.phase = this.phase;
    this.rows.forEach((row, i) => row.classList.toggle("on", i <= idx));
  }

  setPhase(phase) {
    if (!this.names.includes(phase)) phase = this.names[this.names.length - 1];
    this.phase = phase;
    this._apply();
    if (this.onPhase) this.onPhase(phase);
  }

  clear() {
    this._timers.forEach(clearTimeout);
    this._timers = [];
  }

  /** Scripted reveal, then settles on the steady final phase. */
  replay(stepMs = 600) {
    this.clear();
    this.setPhase(this.names[0]);
    this.names.slice(1).forEach((p, i) => {
      this._timers.push(setTimeout(() => this.setPhase(p), (i + 1) * stepMs));
    });
  }
}
