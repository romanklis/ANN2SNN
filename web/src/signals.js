// Canvas 2D line-chart renderer for the experiment's signal lanes.
//
// Used for the control output (plate tilt θx/θy) and the radial tracking error.
// Dependency-free, shared frame cursor with the stage, DPR-aware.

const COLORS = {
  bg: "#0b1018",
  grid: "#1d2637",
  axis: "#2a3547",
  zero: "#3a4757",
  text: "#cfe0f7",
  muted: "#93a1b8",
  cursor: "#e4572e",
};

const FONT_SM = "10px ui-sans-serif, system-ui, sans-serif";
const FONT_MONO = "10px ui-monospace, SFMono-Regular, Menlo, monospace";

export class SignalChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.series = [];
    this.zeroLine = false;
    this.tMax = 0;
    this.frame = 0;
    this._resize = this._resize.bind(this);
    if (window.ResizeObserver) {
      this._ro = new ResizeObserver(this._resize);
      this._ro.observe(canvas.parentElement || canvas);
    } else {
      window.addEventListener("resize", this._resize);
    }
    this._resize();
  }

  /** @param {object} opts {series:[{label,color,values,dash}], zeroLine, tMax} */
  setData({ series = [], zeroLine = false, tMax = 0 } = {}) {
    this.series = series.filter((s) => Array.isArray(s.values) && s.values.length);
    this.zeroLine = !!zeroLine;
    this.tMax = tMax || 0;
    this.draw();
  }

  setFrame(k) {
    this.frame = Math.max(0, k | 0);
    this.draw();
  }

  get total() {
    let n = 0;
    for (const s of this.series) n = Math.max(n, s.values.length);
    return n;
  }

  _resize() {
    const parent = this.canvas.parentElement;
    const cssW = Math.max(120, parent ? parent.clientWidth : 320);
    const cssH = Math.max(60, parent ? parent.clientHeight : 120);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = cssW;
    this.H = cssH;
    this.draw();
  }

  _domain() {
    if (!this.series.length) return [-1, 1];
    let lo = Infinity, hi = -Infinity;
    for (const s of this.series) {
      for (const v of s.values) {
        const n = Number(v);
        if (!Number.isFinite(n)) continue;
        if (n < lo) lo = n;
        if (n > hi) hi = n;
      }
    }
    if (lo === Infinity) return [-1, 1];
    if (this.zeroLine) {
      const a = Math.max(Math.abs(lo), Math.abs(hi), 1e-3) * 1.15;
      return [-a, a];
    }
    return [Math.min(0, lo), Math.max(hi, 1e-3) * 1.15];
  }

  draw() {
    const ctx = this.ctx;
    if (!this.W) this._resize();
    ctx.clearRect(0, 0, this.W, this.H);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, this.W, this.H);

    const m = { l: 38, r: 12, t: 18, b: 16 };
    const w = this.W - m.l - m.r;
    const h = this.H - m.t - m.b;
    if (w <= 20 || h <= 16) return;

    if (!this.series.length) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = FONT_SM;
      ctx.textAlign = "center";
      ctx.fillText("no signal", this.W / 2, this.H / 2);
      ctx.textAlign = "left";
      return;
    }

    const [lo, hi] = this._domain();
    const n = Math.max(2, this.total);
    const X = (i) => m.l + (w * i) / (n - 1);
    const Y = (v) => m.t + h - ((v - lo) / (hi - lo)) * h;

    // horizontal grid + labels
    ctx.font = FONT_MONO;
    ctx.textAlign = "right";
    for (const f of [0, 0.5, 1]) {
      const v = lo + (hi - lo) * f;
      const y = Y(v);
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + w, y); ctx.stroke();
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(v.toFixed(Math.abs(hi - lo) >= 10 ? 0 : 2), m.l - 4, y + 3);
    }
    if (this.zeroLine && lo < 0 && hi > 0) {
      ctx.strokeStyle = COLORS.zero;
      ctx.beginPath(); ctx.moveTo(m.l, Y(0)); ctx.lineTo(m.l + w, Y(0)); ctx.stroke();
    }

    // series
    const k = Math.min(this.frame, n - 1);
    for (const s of this.series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.8;
      ctx.setLineDash(s.dash ? [5, 4] : []);
      ctx.beginPath();
      const lim = Math.min(k, s.values.length - 1);
      for (let i = 0; i <= lim; i++) {
        const x = X(i), y = Y(Number(s.values[i]));
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (lim >= 0) {
        ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.arc(X(lim), Y(Number(s.values[lim])), 2.6, 0, Math.PI * 2); ctx.fill();
      }
    }

    // playhead
    ctx.strokeStyle = COLORS.cursor;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(k), m.t); ctx.lineTo(X(k), m.t + h); ctx.stroke();

    // legend + time caption
    ctx.font = FONT_SM;
    ctx.textAlign = "left";
    let lx = m.l + 2;
    for (const s of this.series) {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, 4, 8, 3);
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(s.label, lx + 12, 9.5);
      lx += 16 + ctx.measureText(s.label).width;
    }
    ctx.textAlign = "right";
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(`${this.tMax.toFixed(2)} s`, m.l + w, this.H - 4);
    ctx.textAlign = "left";
    ctx.fillText("0", m.l - 2, this.H - 4);
  }
}
