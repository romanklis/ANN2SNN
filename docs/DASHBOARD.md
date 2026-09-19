# Interactive dashboard & demo videos

Browser dashboard over the canonical `sim_engine` package: run any controller,
watch the ball animate on the plate, compare solutions side by side, distil the
learned brains, and record polished demo videos.

```
server/   Flask API over sim_engine.api.EngineService + built frontend
web/      Vite + vanilla JS + Plotly frontend (stage canvas, charts, recorder)
tools/    matplotlib + ffmpeg MP4 renderer and CLI
```

## Run it

The stack is designed to run inside the DinD `ann2snn` image
(`registry:5000/openclaw-agent:ann2snn`, which bakes torch 2.4.1+CPU, Flask, Node
20 and ffmpeg). That image's default entrypoint is the OpenClaw agent adapter, so
override it and mount the repo at `/workspace`.

The dind container bind-mounts `${OPENCLAW}/workspaces` (host) at `/workspaces`
(dind), so stage the repo there first:

```bash
OPENCLAW=~/projects/openclaw-contained          # adjust to your checkout
mkdir -p "$OPENCLAW/workspaces/ann2snn-dashboard"
tar -C /path/to/ANN2SNN --exclude=.git --exclude='new elements' \
    --exclude=__pycache__ -cf - . \
  | tar -C "$OPENCLAW/workspaces/ann2snn-dashboard" -xf -

DIND="docker exec openclaw-docker-dind docker run --rm --entrypoint sh --user root \
      -v /workspaces/ann2snn-dashboard:/workspace -w /workspace \
      registry:5000/openclaw-agent:ann2snn"

$DIND -c 'python3 -m pip install -e ".[dashboard,video,dev]"'   # already baked; optional
$DIND -c 'sh -lc "cd web && npm ci && npm run build"'           # -> server/static/
$DIND -c 'sh -lc "python3 -m pytest tests server/tests tools/tests -q"'
$DIND -c 'sh -lc "python3 tools/generate_videos.py"'            # videos/*.mp4
$DIND -c 'sh -lc "gunicorn -b 0.0.0.0:8080 --workers 2 --timeout 300 server.wsgi:application"'
```

Then open `http://localhost:8080` (forward the port from the dind container as
usual). On the host, the `Makefile` targets (`make web-build`, `make test`,
`make serve`, `make videos`) are the same commands once you are *inside* the
image with the repo at `/workspace`.

**One command**: inside the image with the repo at `/workspace`, `make up`
installs nothing extra, builds the frontend and serves the dashboard; plain `make`
does the same (it is the default target). `make help` lists everything.

### Standalone Docker image (no DinD / OpenClaw)

External users can build a single self-contained image from the repo's
`Dockerfile`. It is multi-stage: Node builds the Vite frontend, then a
`python:3.11-slim` CPU runtime installs the engine + CPU-only torch + Flask +
ffmpeg and serves the bundle.

```bash
make up-docker              # plain docker build + run, auto-picks a free host port
make docker-run             # identical (both build then run)
```

`make up-docker` (and `make docker-run`) starts at `PORT` (default 8080) and walks
upward until it finds a **free host port**, prints the chosen URL, streams the
logs and stops the container on Ctrl-C. On a busy host override the start:
`make up-docker PORT=9000`. It uses plain `docker build` + `docker run`, so **no
`docker compose` and no BuildKit/buildx are required**.

An optional Compose workflow is available, but Compose uses a fixed port
(`${PORT:-8080}`) so it fails if that host port is busy:

```bash
mkdir -p videos
make compose-up PORT=9000     # compose, foreground
make compose-down             # stop it
```

Raw commands if you prefer:

```bash
docker build -t ann2snn-dashboard .
docker run --rm -p 8080:8080 -v "$PWD/videos:/app/videos" ann2snn-dashboard
# open http://localhost:8080  (or map a free host port, e.g. -p 9000:8080)
```

Or with Compose directly (remember a free host port):

```bash
mkdir -p videos
PORT=9000 docker compose up --build
```

Useful one-offs (the image's entrypoint is gunicorn, so pass a command to replace
it):

```bash
# render the demo videos into ./videos
docker run --rm -v "$PWD/videos:/app/videos" ann2snn-dashboard \
  python3 tools/generate_videos.py --outdir /app/videos

# run the test suite inside the image
docker run --rm ann2snn-dashboard python3 -m pytest tests server/tests tools/tests -q

# shell
docker run --rm -it --entrypoint sh ann2snn-dashboard
```

`Makefile` shortcuts: `make up` (local), `make up-docker` / `make docker-run`
(auto free-port container), `make docker-build`, `make docker-videos`,
`make docker-shell`, `make compose-up`, `make compose-down`. The image runs as the
non-root `app` user with a `/api/health` `HEALTHCHECK`; weights are cached in the
`/app/data` volume and MP4s are written to `/app/videos`.

Environment variables:

| Var | Default | Purpose |
|---|---|---|
| `ANN2SNN_WEIGHTS` | `./weights.pt` | distilled weights bundle loaded at startup / written by `/api/train` |
| `ANN2SNN_WEB_ROOT` | `server/static` | directory holding the built `index.html` + `assets/` |
| `ANN2SNN_THREADS` | `2` | torch/OpenMP thread cap |
| `ANN2SNN_CORS_ORIGINS` | localhost dev ports | comma-separated CORS allow-list (`*` = all) |
| `ANN2SNN_SEED` | `42` | default engine seed |
| `ANN2SNN_N_NEURONS` | `1000` | hidden neurons per brain |
| `ANN2SNN_AUTO_TRAIN` | `1` | distil + cache weights on first use when none exist |
| `PORT` | `8080` | Flask dev server port (`make serve-dev`) |

> **Balancing out of the box.** The learned controllers (`flylike_ann`,
> `snn_transferred`, `dense_ann`) are useless with random weights — the ball rolls
> off the plate. So on the first request the server behaviourally distils them
> from the PD teacher (~8 s for 100 epochs on CPU), caches the bundle at
> `ANN2SNN_WEIGHTS`, and reuses it afterwards. The standalone Docker image also
> pre-distils at build time, so it starts trained. Set `ANN2SNN_AUTO_TRAIN=0` to
> skip this and run the honest untrained baselines instead (or use the UI's
> **Distill** button / `tools/generate_videos.py`).

## API

All responses are JSON; the API never leaks torch tensors.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | liveness + versions + `trained` |
| `GET` | `/api/controllers` | catalogue (`catalog`) + geometry/defaults |
| `GET` | `/api/engine` | full `Engine.describe()` |
| `POST` | `/api/simulate` | single controller (or `{"controllers": [...]}`) |
| `POST` | `/api/benchmark` | multi-controller comparison on one reference |
| `POST` | `/api/train` | start behavioural distillation (background job) |
| `GET` | `/api/train/<job_id>` | `{state, stage, epoch, epochs, loss, error}` |
| `POST` | `/api/export/mp4` | render + download one MP4 (bounded steps) |
| `POST` | `/api/sessions` | create an interactive stepping session |
| `POST` | `/api/sessions/<id>/step` | advance (`{n?, action?}`) |
| `POST` | `/api/sessions/<id>/reset` | reset the plant + controller |
| `PUT` | `/api/sessions/<id>/controller` | swap the brain |
| `GET` | `/api/sessions/<id>/trajectory` | recorded trace |
| `DELETE` | `/api/sessions/<id>` | drop the session |

`POST /api/simulate` body: `{controller, seed=42, steps=250, radius=0.15,
freq=0.5, spike_format="events"}`. The response is normalised for the UI:
`trajectory` (T×4), `tilts` (T×2), `error_cm` (T), `target` (T×2),
`reference{pos,vel,acc,dt,radius,freq}`, `metrics{mean/rms/max/final_error_cm,
settling_step}`, `meta.controller`, and `spikes{format,shape,data}` for spiking
brains. `steps` is clamped to `[1, 2000]`; `epochs` to `[1, 200]`.

`GET /api/controllers` also returns `plate_half`, `auto_train`, the engine-level
`training` block (`final_loss`, `config`, `epochs_logged`) and, per catalogue
entry, a `guide` describing **how the model was obtained**, its **value/role** and
**how it was trained** (`guide.training.{method, teacher, transferred_from,
epochs, seed, final_loss}`). Training facts survive a loaded weights bundle, so a
pre-distilled image reports them too.

`POST /api/benchmark` (and the multi form of `/api/simulate`) additionally returns
a `stats` block for the overall comparison:

```jsonc
"stats": {
  "plate_half_m": 0.25,
  "ranking": ["pid", ...],
  "per_controller": {"pid": {"mean_error_cm": ..., "rms_error_cm": ..., "max_error_cm": ...,
                             "final_error_cm": ..., "settling_step": ..., "on_plate_pct": 100.0}},
  "summary": {"n": 5, "best": "pid", "best_mean_error_cm": ..., "worst": ...,
              "mean_of_means_cm": ..., "mean_rms_cm": ..., "spread_cm": ...,
              "mean_on_plate_pct": ..., "trained": true}
}
```

## Frontend behaviour

The dashboard is a **single screen with no scrolling** that presents the project as
an **ANN → SNN closed-loop control experiment**. One `POST /api/benchmark` runs
the experiment pair (`flylike_ann`, `snn_transferred`) on the same orbit and feeds
every panel.

* **Hero stage (left, ≥60 % width)** — isometric 3D plate showing only the
  experiment pair: the **fly-like ANN** ball (amber) and the **SNN transferred**
  ball (teal, light ring) plus the dashed target orbit. The SNN is transferred
  from the ANN, so the balls nearly coincide. They are painted in strict series
  order — ANN first, SNN always on top — so the visible colour never flips, and the
  ANN is drawn 1 px larger so a thin amber rim stays visible. Off-plate balls are
  clamped red at the plate edge; an isometric legend keys
  target / FLY-LIKE ANN / SNN TRANSFERRED.
* **Controller pipeline (right, top)** — `FLY-LIKE ANN` → *WEIGHT TRANSFER* →
  `SNN TRANSFERRED` → *CLOSE LOOP* → `BALL + PLATE`. On load (and on each loop /
  record start) it runs a ~1.8 s scripted reveal and then settles on
  `CLOSED LOOP · SPIKING`. The copy is deliberately plain ("SNN transferred from
  the fly-like ANN").
* **Spike activity (right)** — the SNN's spikes on a Canvas 2D raster that
  advances with the shared frame cursor; titled
  `first 200 of <N> neurons` (`N` is the engine's neuron count).
* **Control output (right)** — the SNN's plate tilt `θx`/`θy` in rad (Canvas 2D
  line chart, zero line).
* **Tracking (right)** — radial tracking error in cm vs time: `flylike_ann` solid,
  `snn_transferred` dashed.
* **Single Play/Pause** — one round button drives the whole shared cursor
  (looping). Fixed defaults: seed 42, 250 steps, radius 0.15 m, 0.5 Hz.
* **Header** — `● RUNNING`, live `t = … s`, `● SPIKING`, neuron/frame counts, seed,
  API/trained badges.
* **Result bar (footer)** — `TRANSFER EXPERIMENT`: fly-like ANN error ·
  SNN error · `Δ = SNN − ANN` (neutral wording, no winner).
* **Capture (header icons)** — `●` records the stage-only WebM; `▣` records the
  whole browser tab; `⤓` downloads a server-rendered MP4 of the SNN.

## Demo-video CLI

```bash
python3 tools/generate_videos.py                  # distil + render all clips
python3 tools/generate_videos.py --no-train       # fast untrained baselines
python3 tools/generate_videos.py --weights weights.pt
python3 tools/generate_videos.py --only pid --reuse-cache
python3 tools/generate_videos.py --jobs 4 --stride 2 --dpi 90
```

Outputs into `videos/`: `01_random_ann.mp4`, `02_flylike_ann.mp4`,
`03_snn_transferred.mp4`, `04_pid.mp4`, `all_controllers_sequence.mp4`,
`metrics.json`, `status.json`. Each clip is written to `*.part.mp4`, verified
with `ffprobe` and atomically renamed.

## Tests

```bash
python3 -m pytest tests server/tests tools/tests -q
```

* `server/tests/test_api.py` — health, catalogue, simulate (all controllers,
  aliases, spike formats, determinism, geometry), multi/benchmark, validation,
  sessions, distillation job, MP4 export (skipped without ffmpeg), CORS.
* `server/tests/test_static_dashboard.py` — build integrity, static serving,
  `/api/*` not shadowed, dashboard contract (skips until the bundle is built).
* `tools/tests/test_render_smoke.py` — tiny MP4 render + `ffprobe` (skipped
  without ffmpeg).

## Validation notes

The learned brains are untrained by default and diverge; after distillation the
reference numbers (250 steps, seed 42) are approximately: pid 2.426 cm,
snn_transferred 6.371, flylike_ann 6.427, dense_ann 7.828.
