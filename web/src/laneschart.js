// Multi-row Canvas 2D lane chart for the extended (full-information) dashboard.
//
// One canvas holds a stack of rows that share a single x-axis and one frame
// cursor: e.g. the per-axis position lanes (`r`, `p`, `p̂`) for several
// controllers, or the per-axis command lanes (commanded vs applied).  Rows carry
// their own y-scale, `null` values are drawn as gaps, and a row may overlay
// event markers (impulse kicks, checkpoint fixes).
//
// Dependency-free, like `signals.js` / `raster.js` / `stage.js`: no chart library
// is bundled.  It is intentionally separate from `signals.js` so the hero page's
// two-lane chart keeps its exact behaviour.

const COLORS = {
  bg: "#0b1018",
  grid: "#1d2637",
  axis: "#2a3547",
  zero: "#3a4757",
  text: "#cfe0f7",
  muted: "#93a1b8",
  cursor: "#e4572e",
  bounds: "rgba(74, 222, 128, 0.10)",
};

const FONT_SM = "10px ui-sans-serif, system-ui, sans-serif";
const FONT_MONO = "10px ui-monospace, SFMono-Regular, Menlo, monospace";

export class LaneChart {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.rows = [];
    this.tMax = 0;
    this.frame = 0;
    this.all = false;
    this._resize = this._resize.bind(this);
    if (window.ResizeObserver) {
      this._ro = new ResizeObserver(this._resize);
      this._ro.observe(canvas.parentElement || canvas);
    } else {
      window.addEventListener("resize", this._resize);
    }
    this._resize();
  }

  /**
   * @param {object} opts
   *   rows: [{ label, unit, series: [{label,color,values,dash,width}],
   *            zeroLine, markers: {frames:[], color}, yMin, yMax }]
   *   tMax: seconds covered by the series (for the time axis)
   */
  setData({ rows = [], tMax = 0 } = {}) {
    this.rows = rows.filter((r) => r && Array.isArray(r.series) && r.series.length);
    this.tMax = tMax || 0;
    this.draw();
  }

  setFrame(k) {
    this.frame = Math.max(0, k | 0);
    this.draw();
  }

  /** Draw every trace in full instead of only up to the cursor. */
  set showAll(v) {
    this.all = !!v;
    this.draw();
  }
  get showAll() {
    return this.all;
  }

  get total() {
    let n = 0;
    for (const row of this.rows) {
      for (const s of row.series) n = Math.max(n, s.values.length);
      if (row.markers) for (const f of row.markers.frames) n = Math.max(n, f + 1);
    }
    return n;
  }

  _resize() {
    const parent = this.canvas.parentElement;
    const cssW = Math.max(160, parent ? parent.clientWidth : 480);
    const cssH = Math.max(60, parent ? parent.clientHeight : 160);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = cssW;
    this.H = cssH;
    this.draw();
  }

  _domain(row) {
    let lo = row.yMin, hi = row.yMax;
    if (lo == null || hi == null) {
      let a = Infinity, b = -Infinity;
      for (const s of row.series) {
        for (const v of s.values) {
          const n = Number(v);
          if (v == null || !Number.isFinite(n)) continue;
          if (n < a) a = n;
          if (n > b) b = n;
        }
      }
      if (a === Infinity) { a = -1; b = 1; }
      if (row.zeroLine) {
        const m = Math.max(Math.abs(a), Math.abs(b), 1e-6) * 1.15;
        lo = lo ?? -m;
        hi = hi ?? m;
      } else {
        lo = lo ?? Math.min(0, a);
        hi = hi ?? (Math.max(b, 1e-6) * 1.15);
      }
    }
    if (hi - lo < 1e-12) hi = lo + 1e-12;
    return [lo, hi];
  }

  draw() {
    const ctx = this.ctx;
    if (!this.W) this._resize();
    ctx.clearRect(0, 0, this.W, this.H);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, this.W, this.H);

    if (!this.rows.length) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = FONT_SM;
      ctx.textAlign = "center";
      ctx.fillText("no lane selected", this.W / 2, this.H / 2);
      ctx.textAlign = "left";
      return;
    }

    const m = { l: 52, r: 14, t: 12, b: 16 };
    const gap = 10;
    const plotW = this.W - m.l - m.r;
    const plotH = this.H - m.t - m.b - gap * (this.rows.length - 1);
    if (plotW <= 30 || plotH <= 20) return;
    const rowH = plotH / this.rows.length;

    const total = Math.max(2, this.total);
    const k = this.all ? total - 1 : Math.min(this.frame, total - 1);
    const X = (i) => m.l + (plotW * i) / (total - 1);

    this.rows.forEach((row, ri) => {
      const top = m.t + ri * (rowH + gap);
      const [lo, hi] = this._domain(row);
      const Y = (v) => top + rowH - ((v - lo) / (hi - lo)) * rowH;

      // row band + frame
      ctx.fillStyle = "rgba(255,255,255,0.015)";
      ctx.fillRect(m.l, top, plotW, rowH);

      // y grid (3 lines) + labels
      ctx.font = FONT_MONO;
      ctx.textAlign = "right";
      for (const f of [0, 0.5, 1]) {
        const v = lo + (hi - lo) * f;
        const y = Y(v);
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + plotW, y); ctx.stroke();
        ctx.fillStyle = COLORS.muted;
        const span = Math.abs(hi - lo);
        ctx.fillText(v.toFixed(span >= 10 ? 1 : span >= 1 ? 2 : 3), m.l - 5, y + 3);
      }
      if (row.zeroLine && lo < 0 && hi > 0) {
        ctx.strokeStyle = COLORS.zero;
        ctx.beginPath(); ctx.moveTo(m.l, Y(0)); ctx.lineTo(m.l + plotW, Y(0)); ctx.stroke();
      }

      // row caption: label [unit]
      ctx.font = FONT_SM;
      ctx.textAlign = "left";
      ctx.fillStyle = COLORS.text;
      const unit = row.unit ? ` [${row.unit}]` : "";
      ctx.fillText(`${row.label}${unit}`, m.l + 4, top + 9);

      // series
      ctx.save();
      ctx.beginPath();
      ctx.rect(m.l, top - 1, plotW, rowH + 2);
      ctx.clip();
      for (const s of row.series) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.width ?? 1.6;
        ctx.setLineDash(s.dash ? [5, 4] : []);
        ctx.beginPath();
        let pen = false;
        const lim = Math.min(k, s.values.length - 1);
        for (let i = 0; i <= lim; i++) {
          const v = s.values[i];
          if (v == null || !Number.isFinite(Number(v))) { pen = false; continue; }
          const x = X(i), y = Y(Number(v));
          if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
        }
        ctx.stroke();
        ctx.setLineDash([]);
        // head dot at the last finite sample drawn so far
        for (let i = lim; i >= 0; i--) {
          const v = s.values[i];
          if (v == null || !Number.isFinite(Number(v))) continue;
          ctx.fillStyle = s.color;
          ctx.beginPath();
          ctx.arc(X(i), Y(Number(v)), 2.4, 0, Math.PI * 2);
          ctx.fill();
          break;
        }
      }
      // event markers (impulse kicks, checkpoint fixes)
      if (row.markers && row.markers.frames) {
        ctx.strokeStyle = row.markers.color || COLORS.cursor;
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 1;
        for (const f of row.markers.frames) {
          if (f > k) continue;
          const x = X(f);
          ctx.beginPath(); ctx.moveTo(x, top + 2); ctx.lineTo(x, top + rowH - 2); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      ctx.restore();

      // legend for this row
      let lx = m.l + 4;
      const ly = top + rowH - 4;
      for (const s of row.series) {
        ctx.fillStyle = s.color;
        ctx.fillRect(lx, ly - 3, 8, 3);
        ctx.fillStyle = COLORS.muted;
        ctx.fillText(s.label, lx + 11, ly);
        lx += 15 + ctx.measureText(s.label).width;
      }
    });

    // shared playhead across every row
    ctx.strokeStyle = COLORS.cursor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X(k), m.t);
    ctx.lineTo(X(k), this.H - m.b);
    ctx.stroke();

    // shared time axis
    ctx.font = FONT_MONO;
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "left";
    ctx.fillText("0", m.l - 2, this.H - 4);
    ctx.textAlign = "right";
    const t = (k / Math.max(1, total - 1)) * (this.tMax || 0);
    ctx.fillText(`${t.toFixed(2)} / ${(this.tMax || 0).toFixed(2)} s`, m.l + plotW, this.H - 4);
    ctx.textAlign = "left";
  }
}
