# ANN2SNN dashboard — frontend workspace

Vanilla-JS **single-screen** dashboard presenting the ANN2SNN ball-and-plate as an
**ANN → SNN closed-loop control experiment**, bundled with **Vite** and served as
static files by the Flask backend (`server/app.py`) at `/`. It talks to that API
exclusively — **no simulation maths is re-implemented in JS**.

Stack: `vite` only. The plate, spike raster and signal lanes are dependency-free
**Canvas 2D** renderers, so the view can be recorded via `canvas.captureStream()`.
(The `plotly` entry in `package.json` is no longer imported or bundled.)

## Layout

```
web/
├── package.json          # scripts: build / dev / preview
├── package-lock.json     # committed -> `npm ci` works
├── vite.config.js        # root = web/, output = ../server/static, base "./"
├── .npmrc                # pins the npm cache onto the workspace volume
├── index.html            # header + hero stage + pipeline/spikes/control/tracking + result bar
└── src/
    ├── main.js           # load benchmark, hero ANN/SNN, panels, result bar, capture
    ├── api.js            # fetch wrapper for the Flask API
    ├── stage.js          # Canvas 2D isometric plate + ANN/SNN spheres
    ├── raster.js         # Canvas 2D spike-train raster
    ├── signals.js        # Canvas 2D line charts (control output + tracking)
    ├── pipeline.js       # controller-pipeline phase state machine
    ├── record.js         # MediaRecorder (canvas + whole-tab)
    └── style.css         # single-viewport grid (no scrolling)
```

## What it shows

One `POST /api/benchmark` runs the experiment pair (`flylike_ann`,
`snn_transferred`) on the same orbit; the page tells one story:

* **Hero stage (left)** — isometric 3D plate with only the experiment pair: the
  **fly-like ANN** ball (amber) and the **SNN transferred** ball (teal, light
  ring), plus the dashed target orbit. The SNN is transferred from the ANN, so the
  balls nearly coincide. They are painted in strict series order (ANN first, SNN
  always on top) so the visible colour never flips; the ANN is 1 px larger so a
  thin amber rim stays visible.
* **Controller pipeline (right top)** — four compact rows for the closed loop:
  `PLANT (ball+plate) → CAMERA (y = [x,y]+v) → KALMAN (x̂) → POLICY (π(e), e = x̂ − r)`,
  with the `FLY-LIKE ANN → SNN TRANSFERRED` name on the POLICY row and a footnote
  that the weight transfer is **offline** (not part of the loop). A scripted reveal
  runs on load/loop/record.
* **Spike activity (right)** — first 200 of `<N>` SNN neurons, animated with the
  shared cursor.
* **Control output (right)** — SNN plate tilt `θx`/`θy` (rad).
* **Tracking (right)** — radial error: ANN solid, SNN dashed.
* **Result bar (footer)** — `TRANSFER EXPERIMENT`: ANN error · SNN error ·
  `Δ = SNN − ANN` (neutral, no winner).
* **Environment selector (header)** — `Clean · Noisy · Delayed · Perturbed · Heavy ·
  Embodied · Randomized` (default **Embodied**) with a `clean`/`robust` policy pill;
  embodied presets use the robust policy (auto-distilled). A footer **robustness
  strip** shows ANN/SNN mean error across presets. See `docs/EMBODIMENT.md`.
* **Single Play/Pause** — one button drives the shared cursor (looping). Fixed
  defaults: seed 42, 500 steps (10 s), r 0.15, f 0.5 Hz.

## Build

```bash
cd web
npm ci            # or: npm install  (lockfile is committed)
npm run build     # -> ../server/static  (index.html + assets/)
```

Then start the backend from the repo root:

```bash
gunicorn -b 0.0.0.0:8080 --workers 2 --timeout 300 server.wsgi:application
```

## Development

```bash
npm run dev       # Vite dev server on :5173
```

The dev server runs on a different origin than Flask, so open the page with an
explicit API base: `http://localhost:5173/?api=http://localhost:8080` (or set
`window.ANN2SNN_API_BASE`). The Flask CORS allow-list already includes `:5173`.

## Recording

Three small header icons preserve the video workflow without cluttering the view:

* **●** records the hero stage canvas to a `.webm` clip (one animation pass).
* **▣** records the whole browser tab through `getDisplayMedia` (asks permission)
  — use this to capture the panels too.
* **⤓** asks the backend to render a deterministic H.264 MP4 of the SNN with
  matplotlib + ffmpeg (`tools/render.py`).
