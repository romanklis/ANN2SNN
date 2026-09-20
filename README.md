# ANN2SNN — Ball-and-Plate Simulation Engine

A refactor of the ball-and-plate control benchmark into a proper, importable
Python package (`sim_engine`) with a CLI and a reusable API so the same engine
can drive a web backend and an interactive frontend.

The benchmark pits **four brains** against the same closed-loop task — keep a
rolling ball on a prescribed 2D orbit by tilting a plate:

| Brain | Class | Kind |
|---|---|---|
| **PID controlled** | `ClassicalPDController` | analytic PD + acceleration feed-forward |
| **Random ANN** | `DenseNNController` (untrained) | `4 → 1000 ReLU → 2` |
| **Fly-like ANN** | `ConnectomeANNController` | sparse recurrent connectome, Dale's law, balanced E/I |
| **SNN transferred** | `LosslessConnectomeSNN` | IF spiking transfer of the connectome ANN |
| **Dense ANN (distilled)** | `DenseNNController` (trained) | behavioral distillation of the PID teacher |

The closed loop is the canonical control/estimation separation — the controller
never sees the true state:

```
x_{k+1} = f(x_k, u_k) + w_k
y_k     = h(x_k) + v_k            h(x) = [x, y]     (camera: position only)
x̂_k     = E(y_{0:k}, u_{0:k-1})                    E = Kalman filter
e_k     = x̂_k − r_k
u_ff,k  = −â_ref,k / C            â_ref reconstructed from x̂ (known rolling gain C)
u_k     = π_θ([e_k, u_ff,k])      6-D policy input
```

The learned arms receive the same feed-forward channel the PID reference uses
(`u_ff`, reconstructed from the Kalman estimate — never the true state), so the
comparison is like-for-like.

`π_ANN → π_SNN` is an **offline** weight transfer; both networks are inserted into
this identical loop, so the ANN/SNN comparison holds the plant and reference fixed.
The body/environment (sensor noise, sensor/actuator delay, damping, scaled rolling
gain, perturbations) is configurable via presets — see [`docs/EMBODIMENT.md`](docs/EMBODIMENT.md).

---

## Install

```bash
pip install -r requirements.txt      # torch, numpy, matplotlib
# or, to install the engine itself as a package:
pip install -e .
```

CPU-only is fine — everything runs on CPU.

## Quick start

```python
from sim_engine import Engine

engine = Engine()                       # untrained brains (instant)
report = engine.run_benchmark(["pid", "random_ann", "flylike_ann", "snn_transferred"])
print(report["ranking"])
```

To reproduce the full result, distil the networks from the PID teacher first:

```python
from sim_engine import Engine, EngineConfig, TrainingConfig

cfg = EngineConfig(train_on_init=True)
engine = Engine(cfg)                    # ~12 s on CPU for 100 epochs
report = engine.run_benchmark(["pid", "flylike_ann", "snn_transferred", "random_ann"])
```

Typical CPU numbers (10-second, 500-frame orbit, clean environment, 1000 neurons, seed 42):

```
1. pid              mean  1.447 cm
2. dense_ann        mean  1.631 cm
3. flylike_ann      mean  1.797 cm
4. snn_transferred  mean  2.029 cm
5. pid_no_ff        mean  7.556 cm     # PD without the acceleration feed-forward
6. random_ann       mean 634.607 cm    # diverges over the longer horizon
```

All controllers receive the same 6-D policy input `[e, u_ff]`, where `u_ff` is the
feed-forward command reconstructed from the Kalman estimate. For reference,
`pid_no_ff` is the same PD law with that channel removed — the ≈7.6 cm
no-feed-forward limit the learned arms used to sit at, which is why the earlier
ranking looked like a distillation shortfall. It was an input-information gap.

The **SNN transfer** is a rate-coded approximation of the connectome ANN (binary
spikes through the recurrence, 10 micro-steps/frame, mean output rate as the
command). It is *empirically* faithful — over 5 seeds (500 frames, clean) the ANN
tracks `1.675 ± 0.063 cm` and the SNN `1.931 ± 0.082 cm`, a ≈0.26 cm gap — and is
not claimed to be an identity. Run `python -m sim_engine benchmark --seeds
42,1,2,3,4` for mean ± spread; a single seed is not evidence.

## Command line

```bash
python -m sim_engine list                       # catalogue of brains
python -m sim_engine describe                   # full engine config as JSON
python -m sim_engine benchmark --controllers pid,random_ann,flylike_ann,snn_transferred \
        --steps 500 --out results.json
python -m sim_engine train --epochs 100 --save weights.pt
python -m sim_engine session --controller snn_transferred --frames 20 --verbose
```

`python -m sim_engine benchmark --train` distils the ANNs in-memory before
evaluating; `--weights weights.pt` reuses a saved bundle.

## Interactive dashboard & demo videos

A browser dashboard over `sim_engine` lives in `server/` (Flask API) + `web/`
(Vite + Plotly frontend), with a matplotlib/ffmpeg MP4 renderer in `tools/`.
It animates the ball on the plate, ranks **all solutions** on one shared orbit
(leaderboard + aggregate statistics), shows a **model guide** (how each brain was
obtained and trained), distils and caches the learned brains, and records demo
clips (in-page WebM or server MP4).

```bash
make install          # one-time: pip install -e ".[dashboard,video,dev]"
make up               # build the frontend + serve the dashboard on :8080
```

Other useful targets: `make test`, `make videos-train`
(`videos/01_random_ann.mp4 … all_controllers_sequence.mp4`), `make help`.

For anyone without that environment, the same solution comes up in a container:

```bash
make up-docker        # plain docker build + run, auto-picking a free host port
make docker-run       # same (build + run), or use: make compose-up PORT=9000
```

No `docker compose` or BuildKit/buildx is required; `make up-docker` finds a free
host port and prints the URL. See
[`docs/DASHBOARD.md`](docs/DASHBOARD.md) for the API, frontend behaviour,
recording workflow, the DinD image recipe and the standalone Docker image.

## The reusable API (for the web backend)

`sim_engine.api.EngineService` is the transport-agnostic seam a Flask/FastAPI
layer wraps. It takes plain dicts/lists and returns JSON-serializable dicts —
no torch objects cross the boundary.

```python
from sim_engine.api import EngineService

svc = EngineService({"steps": 500, "train_on_init": True})

svc.controllers()                       # catalogue for the UI
svc.benchmark(["pid", "snn_transferred"])   # batch closed-loop report

info = svc.new_session("flylike_ann")   # interactive stepping
sid  = info["session_id"]
obs  = svc.step(sid)                    # advance one 20 ms frame
obs  = svc.step(sid, n=10)              # or ten frames
svc.reset(sid)
svc.set_controller(sid, "snn_transferred")
svc.trajectory(sid)                     # whole recorded trace
```

See [`docs/API.md`](docs/API.md) for the full surface and response shapes.

## Package layout

```
sim_engine/
├── physics.py          # step_physics(), constants, BallPlatePlant
├── reference.py        # orbit reference trajectory (pos/vel/acc)
├── config.py           # Plant/Network/Benchmark/Training/Engine configs
├── controllers/
│   ├── base.py         # BaseController contract + error_vector()
│   ├── classical.py    # ClassicalPDController
│   ├── dense.py        # DenseNNController
│   ├── connectome.py   # ConnectomeTopology, ConnectomeANNController
│   └── snn.py          # LosslessConnectomeSNN
├── training.py         # behavioral distillation + weight save/load
├── environment.py      # embodied env: sensing, actuation, body, perturbations
├── estimators.py       # Kalman filter (position-only camera -> full state)
├── robustness.py       # environmental difficulty sweeps
├── benchmark.py        # closed-loop trajectory evaluation + metrics
├── registry.py         # name -> controller factory (demo brains)
├── engine.py           # Engine + SimulationSession
├── api.py              # reusable JSON-in/JSON-out API (EngineService)
├── serialization.py    # tensor/numpy -> JSON helpers
└── cli.py / __main__.py
examples/run_benchmark.py

server/                  # Flask API over sim_engine.api.EngineService
web/                     # Vite + Plotly dashboard (builds into server/static)
tools/                   # matplotlib + ffmpeg MP4 renderer and CLI
docs/DASHBOARD.md        # dashboard/API/video documentation
Makefile                 # install / test / web-build / serve / videos
```

### Design rules

* **`physics.py` owns every constant.** Controllers import `DT`, `MAX_TILT`,
  `C_CONST`, … from there — nothing is re-hardcoded.
* **One controller contract.** Every brain implements
  `reset()` + `act(state, ref) -> tilt`, so `benchmark.run_closed_loop` is
  controller-agnostic and a new brain is a drop-in.
* **Topology is data, not code.** `ConnectomeTopology` is a seeded, immutable
  anatomy (edge list + E/I polarity + signs); only synaptic *magnitudes* are
  trainable, which is what makes Dale's law meaningful.
* **No web framework in the engine.** `api.py` is pure dict-in/dict-out.

## Physics

Ball of radius `r` rolling without slipping on a plate tilted by
`(theta_x, theta_y)`:

```
x'' = -(5/7) g sin(theta_x) ≈ -C theta_x        C = (5/7) g = 7.007 m/s²
y'' = -(5/7) g sin(theta_y) ≈ -C theta_y
```

Integrated explicitly at `DT = 0.02 s` (50 Hz), with the plate angle saturated
to `MAX_TILT = 0.25 rad`. Control inputs are the tracking error
`[ex, ey, evx, evy]`; the reference orbit has radius `0.15 m` at `0.5 Hz`.
