// Canvas stage: every model's ball rolling on the same plate, drawn as an
// isometric 3D scene — a solid plate with depth and small shaded spheres.
//
// The stage never computes physics — it replays the traces returned by
// POST /api/benchmark. A single frame cursor drives all series, so the balls are
// directly comparable. Canvas 2D only, so `canvas.captureStream` can record it.

const COLORS = {
  bg: "#0b1018",
  plate: "#eef2f7",
  plateSide: "#c3ccd8",
  plateEdge: "#3a4757",
  grid: "#dbe3ec",
  axis: "#b9c4d1",
  orbit: "#aeb9c6",
  ref: "#e4572e",
  text: "#cfe0f7",
  muted: "#93a1b8",
  off: "#ff4d4d",
};

const FONT_SM = "11px ui-sans-serif, system-ui, sans-serif";
const COS30 = Math.cos(Math.PI / 6);
const SIN30 = Math.sin(Math.PI / 6);

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const n = parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function mix(hex, to, amt) {
  const [r, g, b] = hexToRgb(hex);
  const a = Math.max(0, Math.min(1, amt));
  const f = (v) => Math.round(v + (to - v) * a);
  return `rgb(${f(r)},${f(g)},${f(b)})`;
}
const lighten = (hex, a) => mix(hex, 255, a);
const darken = (hex, a) => mix(hex, 0, a);

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");

    this.reference = null;       // { pos, radius, dt, ... }
    this.series = [];            // [{name,label,color,trajectory}]
    this.plateHalf = 0.25;

    this.frame = 0;
    this.playing = false;
    this.loop = true;
    this._lastTs = 0;
    this._raf = null;

    this.onFrame = null;         // (frame, total) => void
    this.onEnded = null;         // () => void

    this._resize = this._resize.bind(this);
    this._tick = this._tick.bind(this);
    if (window.ResizeObserver) {
      this._ro = new ResizeObserver(this._resize);
      this._ro.observe(canvas.parentElement || canvas);
    } else {
      window.addEventListener("resize", this._resize);
    }
    this._resize();
  }

  // ------------------------------------------------------------------ data
  setScene({ reference, series = [], plateHalf = 0.25 }) {
    this.reference = reference || null;
    this.series = series.filter((s) => s.trajectory && s.trajectory.length);
    this.plateHalf = plateHalf || 0.25;
    this.frame = 0;
    this._lastTs = 0;
    this.draw();
    this._emit();
  }

  get total() {
    let n = 0;
    for (const s of this.series) n = Math.max(n, s.trajectory.length);
    return n;
  }

  // -------------------------------------------------------------- playback
  play() {
    if (this.total < 2 || this.playing) return;
    if (this.frame >= this.total - 1) this.frame = 0;
    this.playing = true;
    this._lastTs = 0;
    this._raf = requestAnimationFrame(this._tick);
  }

  pause() {
    this.playing = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  }

  toggle() {
    if (this.playing) this.pause();
    else this.play();
    return this.playing;
  }

  setFrame(k) {
    const t = this.total;
    this.frame = Math.max(0, Math.min(Math.max(0, t - 1), Math.round(k)));
    this.draw();
    this._emit();
  }

  setLoop(v) {
    this.loop = !!v;
  }

  reset() {
    this.pause();
    this.setFrame(0);
  }

  _tick(ts) {
    if (!this.playing) return;
    if (!this._lastTs) this._lastTs = ts;
    const dtMs = Math.min(100, ts - this._lastTs);
    this._lastTs = ts;
    const dt = this.reference?.dt ?? 0.02;
    this.frame += (dtMs / 1000) / dt;      // real-time playback
    if (this.frame >= this.total - 1) {
      this.frame = this.total - 1;
      this.draw();
      this._emit();
      if (this.loop) {
        this.frame = 0;
        this._raf = requestAnimationFrame(this._tick);
      } else {
        this.pause();
        if (this.onEnded) this.onEnded();
      }
      return;
    }
    this.draw();
    this._emit();
    this._raf = requestAnimationFrame(this._tick);
  }

  _emit() {
    if (this.onFrame) this.onFrame(Math.floor(this.frame), this.total);
  }

  // --------------------------------------------------------------- layout
  _resize() {
    const parent = this.canvas.parentElement;
    const cssW = Math.max(240, parent ? parent.clientWidth : 640);
    const cssH = Math.max(200, parent ? parent.clientHeight : 480);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = cssW;
    this.H = cssH;
    this.draw();
  }

  // Isometric frame: world (x, y) on the plate -> screen, with room reserved for
  // the plate thickness and the sphere radius.
  _iso() {
    const L = this.plateHalf;
    const halfW = 2 * L * COS30;      // max |isoX| over the plate corners
    const halfH = 2 * L * SIN30;      // max |isoY|
    const mX = 34, mTop = 30, mBottom = 30;
    const unit = Math.max(20, Math.min(
      (this.W - 2 * mX) / (2 * halfW),
      (this.H - mTop - mBottom) / (2 * halfH),
    ));
    const ballR = Math.max(5, L * 0.12 * unit);
    const thickness = Math.max(7, L * 0.14 * unit);
    const availH = this.H - mTop - mBottom - thickness;
    const cx = this.W / 2;
    const cy = mTop + availH / 2;
    return { L, unit, ballR, thickness, cx, cy };
  }

  _project(iso, x, y) {
    return {
      x: iso.cx + (x - y) * COS30 * iso.unit,
      y: iso.cy + (x + y) * SIN30 * iso.unit,
    };
  }

  // --------------------------------------------------------------- drawing
  draw() {
    const ctx = this.ctx;
    if (!this.W) this._resize();
    ctx.clearRect(0, 0, this.W, this.H);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, this.W, this.H);

    const iso = this._iso();
    const L = iso.L;
    const P = (x, y) => this._project(iso, x, y);

    // ---- plate extrusion (two visible front faces) ----------------------- #
    const corners = [[L, L], [L, -L], [-L, -L], [-L, L]].map(([x, y]) => P(x, y));
    const th = iso.thickness;
    ctx.fillStyle = COLORS.plateSide;
    for (const [a, b] of [[0, 1], [0, 3]]) {
      ctx.beginPath();
      ctx.moveTo(corners[a].x, corners[a].y);
      ctx.lineTo(corners[b].x, corners[b].y);
      ctx.lineTo(corners[b].x, corners[b].y + th);
      ctx.lineTo(corners[a].x, corners[a].y + th);
      ctx.closePath();
      ctx.fill();
    }

    // ---- plate top + grid ------------------------------------------------ #
    ctx.beginPath();
    corners.forEach((c, i) => (i ? ctx.lineTo(c.x, c.y) : ctx.moveTo(c.x, c.y)));
    ctx.closePath();
    ctx.fillStyle = COLORS.plate;
    ctx.fill();

    ctx.save();
    ctx.clip();
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const g = -L + (2 * L * i) / 4;
      let a = P(g, -L), b = P(g, L);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      a = P(-L, g); b = P(L, g);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    ctx.strokeStyle = COLORS.axis;
    let a0 = P(0, -L), b0 = P(0, L);
    ctx.beginPath(); ctx.moveTo(a0.x, a0.y); ctx.lineTo(b0.x, b0.y); ctx.stroke();
    a0 = P(-L, 0); b0 = P(L, 0);
    ctx.beginPath(); ctx.moveTo(a0.x, a0.y); ctx.lineTo(b0.x, b0.y); ctx.stroke();
    ctx.restore();

    ctx.beginPath();
    corners.forEach((c, i) => (i ? ctx.lineTo(c.x, c.y) : ctx.moveTo(c.x, c.y)));
    ctx.closePath();
    ctx.strokeStyle = COLORS.plateEdge;
    ctx.lineWidth = 2;
    ctx.stroke();

    if (!this.series.length) {
      const c = P(0, 0);
      ctx.fillStyle = COLORS.muted;
      ctx.font = FONT_SM;
      ctx.textAlign = "center";
      ctx.fillText("Loading solutions…", c.x, c.y);
      ctx.textAlign = "left";
      return;
    }

    const clipHalf = this.plateHalf;
    const clip = (v) => Math.max(-clipHalf, Math.min(clipHalf, v));
    const k = Math.floor(this.frame);

    // ---- reference orbit (on the plate plane) ---------------------------- #
    const ref = this.reference?.pos || [];
    if (ref.length) {
      ctx.strokeStyle = COLORS.orbit;
      ctx.setLineDash([8, 6]);
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ref.forEach((q, i) => {
        const s = P(q[0], q[1]);
        if (i) ctx.lineTo(s.x, s.y); else ctx.moveTo(s.x, s.y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // ---- trails (projected onto the plate) ------------------------------- #
    ctx.globalAlpha = 0.4;
    ctx.lineWidth = 1.6;
    for (const s of this.series) {
      const traj = s.trajectory;
      ctx.strokeStyle = s.color;
      ctx.beginPath();
      for (let i = 0; i <= k && i < traj.length; i++) {
        const p = P(clip(traj[i][0]), clip(traj[i][1]));
        if (i) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // ---- moving target: flat disc on the plate --------------------------- #
    const cur = ref[k];
    if (cur) {
      const p = P(cur[0], cur[1]);
      const ry = iso.ballR * 0.95 * (SIN30 / COS30);
      ctx.fillStyle = "rgba(228,87,46,0.16)";
      ctx.beginPath();
      ctx.ellipse(p.x, p.y, iso.ballR * 0.95, ry, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = COLORS.ref;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // ---- balls: strict painting order (series order, no depth sort) ------- #
    // The ANN and the transferred SNN nearly coincide, so sorting by depth made
    // the top ball flip colour frame-to-frame. Painting in series order keeps
    // the SNN (teal + ring) always on top; the ANN is drawn 1 px larger so a
    // thin amber rim stays visible behind it.
    const balls = [];
    for (const s of this.series) {
      if (k >= s.trajectory.length) continue;
      balls.push({ s, x: s.trajectory[k][0], y: s.trajectory[k][1], ring: !!s.ring });
    }

    for (const b of balls) {
      const inside = Math.abs(b.x) <= this.plateHalf && Math.abs(b.y) <= this.plateHalf;
      const p = P(clip(b.x), clip(b.y));
      const r = b.ring ? iso.ballR : iso.ballR + 1;
      const color = inside ? b.s.color : COLORS.off;

      // contact shadow on the plate
      ctx.fillStyle = "rgba(0,0,0,0.22)";
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + r * 0.62, r * 1.02, r * 0.44, 0, 0, Math.PI * 2);
      ctx.fill();

      // shaded sphere
      const g = ctx.createRadialGradient(
        p.x - r * 0.35, p.y - r * 0.45, r * 0.12,
        p.x, p.y, r * 1.18
      );
      g.addColorStop(0, lighten(color, 0.6));
      g.addColorStop(0.55, color);
      g.addColorStop(1, darken(color, 0.4));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.stroke();

      // specular highlight
      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.beginPath();
      ctx.arc(p.x - r * 0.34, p.y - r * 0.42, r * 0.22, 0, Math.PI * 2);
      ctx.fill();

      // transfer marker: the SNN ball carries a light ring so it stays visible
      // where it overlaps the ANN ball (the transfer is near-lossless)
      if (b.ring) {
        ctx.strokeStyle = "#eaf6ff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 1.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // ---- clock ------------------------------------------------------------ #
    const t = (k * (this.reference?.dt ?? 0.02)).toFixed(2);
    ctx.fillStyle = COLORS.text;
    ctx.font = FONT_SM;
    ctx.textAlign = "right";
    ctx.fillText(`t = ${t} s  ·  frame ${k}/${Math.max(0, this.total - 1)}`, this.W - 14, this.H - 12);
    ctx.textAlign = "left";
  }
}
