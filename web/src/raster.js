// Canvas spike-train raster for the spiking model.
//
// Replaces the old Plotly raster with a dependency-free Canvas 2D renderer that
// animates with the shared frame cursor. A static T×N bitmap is built once from
// the spike payload, then columns up to the cursor are blitted each frame.

const COLORS = {
  bg: "#0b1018",
  spike: "#7ee0c0",
  cursor: "#e4572e",
  text: "#cfe0f7",
  muted: "#93a1b8",
  grid: "#1d2637",
};

const FONT_SM = "11px ui-sans-serif, system-ui, sans-serif";

export class SpikeRaster {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.T = 0;
    this.N = 0;
    this.label = "";
    this.bitmap = null;      // offscreen canvas (T x N)
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

  /** @param {object} opts {spikes, T, maxNeurons, label} */
  setData({ spikes, T = 0, maxNeurons = 200, label = "SNN" } = {}) {
    this.label = label;
    this.T = T;
    this.bitmap = null;
    this.N = 0;

    if (!spikes || spikes.format === "none" || spikes.format === "counts") {
      this.draw();
      return;
    }
    const totalN = spikes.shape?.[1] || 0;
    const N = Math.min(totalN, maxNeurons);
    if (N <= 0 || T <= 0) {
      this.draw();
      return;
    }

    const off = document.createElement("canvas");
    off.width = T;
    off.height = N;
    const octx = off.getContext("2d");
    octx.fillStyle = COLORS.spike;

    if (spikes.format === "events") {
      for (const [t, n] of spikes.data) {
        if (t >= 0 && t < T && n >= 0 && n < N) octx.fillRect(t, n, 1, 1);
      }
    } else if (spikes.format === "full") {
      const rows = spikes.data || [];
      const lim = Math.min(rows.length, T);
      for (let t = 0; t < lim; t++) {
        const row = rows[t];
        const cols = Math.min(row.length, N);
        for (let n = 0; n < cols; n++) if (row[n]) octx.fillRect(t, n, 1, 1);
      }
    } else {
      this.draw();
      return;
    }

    this.bitmap = off;
    this.N = N;
    this.draw();
  }

  setFrame(k) {
    this.frame = Math.max(0, k | 0);
    this.draw();
  }

  _resize() {
    const parent = this.canvas.parentElement;
    const cssW = Math.max(160, parent ? parent.clientWidth : 320);
    const cssH = Math.max(80, parent ? parent.clientHeight : 160);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = cssW;
    this.H = cssH;
    this.draw();
  }

  draw() {
    const ctx = this.ctx;
    if (!this.W) this._resize();
    ctx.clearRect(0, 0, this.W, this.H);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, this.W, this.H);

    const m = { l: 30, r: 12, t: 20, b: 16 };
    const w = this.W - m.l - m.r;
    const h = this.H - m.t - m.b;
    if (w <= 10 || h <= 10) return;

    if (!this.bitmap) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = FONT_SM;
      ctx.textAlign = "center";
      ctx.fillText("non-spiking model", this.W / 2, this.H / 2);
      ctx.textAlign = "left";
      return;
    }

    // frame the raster and copy the columns up to the cursor
    const srcW = Math.max(1, Math.min(this.T, this.frame + 1));
    ctx.imageSmoothingEnabled = false;
    const dstW = (w * srcW) / this.T;
    ctx.drawImage(this.bitmap, 0, 0, srcW, this.N, m.l, m.t, dstW, h);
    ctx.imageSmoothingEnabled = true;

    // playhead
    const cx = m.l + dstW;
    ctx.strokeStyle = COLORS.cursor;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, m.t); ctx.lineTo(cx, m.t + h); ctx.stroke();

    // labels
    ctx.fillStyle = COLORS.muted;
    ctx.font = FONT_SM;
    ctx.textAlign = "right";
    ctx.fillText(String(this.N), m.l - 6, m.t + 8);
    ctx.fillText("0", m.l - 6, m.t + h);
    ctx.textAlign = "left";
    ctx.fillText(`${this.label} · first ${this.N} neurons`, m.l, m.t - 7);
    ctx.textAlign = "right";
    ctx.fillText(`${this.T} frames`, m.l + w, m.t - 7);
    ctx.textAlign = "left";
  }
}
