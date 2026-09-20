// Controller-pipeline state machine.
//
// Purely cosmetic: it drives the `data-phase` attribute and the `.on` classes
// that CSS animates for the one-time reveal
// (PLANT → CAMERA → KALMAN → POLICY). It never gates or alters the real
// benchmark trace.

export const PHASES = ["plant", "camera", "kalman", "policy"];

export class Pipeline {
  /** @param {HTMLElement} root @param {object} opts {onPhase} */
  constructor(root, { onPhase = null } = {}) {
    this.root = root;
    this.onPhase = onPhase;
    this._timers = [];
    this.rows = {
      plant: root.querySelector(".node-plant"),
      camera: root.querySelector(".node-camera"),
      kalman: root.querySelector(".node-kalman"),
      policy: root.querySelector(".node-policy"),
    };
    this.phase = "plant";
    // Apply the initial state *without* firing onPhase: the caller's `onPhase`
    // may read the very object being constructed (TDZ) if invoked here.
    this._apply();
  }

  /** True once the controller is in the loop (steady state). */
  get spiking() {
    return this.phase === "policy";
  }

  /** Paint the current phase onto the DOM (no callback). */
  _apply() {
    const idx = PHASES.indexOf(this.phase);
    this.root.dataset.phase = this.phase;
    for (let i = 0; i < PHASES.length; i++) {
      this.rows[PHASES[i]]?.classList.toggle("on", i <= idx);
    }
  }

  setPhase(phase) {
    if (!PHASES.includes(phase)) phase = "policy";
    this.phase = phase;
    this._apply();
    if (this.onPhase) this.onPhase(phase);
  }

  clear() {
    this._timers.forEach(clearTimeout);
    this._timers = [];
  }

  /** Scripted ~1.8 s reveal, then settles on the steady `policy` phase. */
  replay(stepMs = 600) {
    this.clear();
    this.setPhase("plant");
    PHASES.slice(1).forEach((p, i) => {
      this._timers.push(setTimeout(() => this.setPhase(p), (i + 1) * stepMs));
    });
  }
}
