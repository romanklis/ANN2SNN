// Controller-pipeline state machine.
//
// Purely cosmetic: it drives the `data-phase` attribute and the `.on` classes
// that CSS animates for the one-time reveal
// (ANN CONTROLLER → WEIGHT TRANSFER → SNN SPIKING → CLOSED LOOP). It never gates
// or alters the real benchmark trace.

export const PHASES = ["ann", "transfer", "snn", "loop"];

export class Pipeline {
  /** @param {HTMLElement} root @param {object} opts {onPhase} */
  constructor(root, { onPhase = null } = {}) {
    this.root = root;
    this.onPhase = onPhase;
    this._timers = [];
    this.nodes = {
      ann: root.querySelector(".node-ann"),
      snn: root.querySelector(".node-snn"),
      plant: root.querySelector(".node-plant"),
    };
    this.links = {
      transfer: root.querySelector(".link-transfer"),
      loop: root.querySelector(".link-loop"),
    };
    this.phase = "ann";
    // Apply the initial state *without* firing onPhase: the caller's `onPhase`
    // may read the very object being constructed (TDZ) if invoked here.
    this._apply();
  }

  get spiking() {
    return this.phase === "snn" || this.phase === "loop";
  }

  /** Paint the current phase onto the DOM (no callback). */
  _apply() {
    const idx = PHASES.indexOf(this.phase);
    const at = (name) => idx >= PHASES.indexOf(name);
    this.root.dataset.phase = this.phase;
    this.nodes.ann?.classList.toggle("on", at("ann"));
    this.links.transfer?.classList.toggle("on", at("transfer"));
    this.nodes.snn?.classList.toggle("on", at("snn"));
    this.links.loop?.classList.toggle("on", at("loop"));
    this.nodes.plant?.classList.toggle("on", at("loop"));
  }

  setPhase(phase) {
    if (!PHASES.includes(phase)) phase = "loop";
    this.phase = phase;
    this._apply();
    if (this.onPhase) this.onPhase(phase);
  }

  clear() {
    this._timers.forEach(clearTimeout);
    this._timers = [];
  }

  /** Scripted ~1.8 s reveal, then settles on the steady `loop` phase. */
  replay(stepMs = 600) {
    this.clear();
    this.setPhase("ann");
    PHASES.slice(1).forEach((p, i) => {
      this._timers.push(setTimeout(() => this.setPhase(p), (i + 1) * stepMs));
    });
  }
}
