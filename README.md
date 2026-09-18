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

Typical CPU numbers (250-frame orbit):

```
1. pid              mean  2.426 cm
2. snn_transferred  mean  6.369 cm
3. flylike_ann      mean  6.427 cm
4. random_ann       mean 74.890 cm
```

The **SNN transfer is essentially lossless**: it tracks within a few
hundredths of a centimetre of the connectome ANN it was transferred from.

## Command line

```bash
python -m sim_engine list                       # catalogue of brains
python -m sim_engine describe                   # full engine config as JSON
python -m sim_engine benchmark --controllers pid,random_ann,flylike_ann,snn_transferred \
        --steps 250 --out results.json
python -m sim_engine train --epochs 100 --save weights.pt
python -m sim_engine session --controller snn_transferred --frames 20 --verbose
```

`python -m sim_engine benchmark --train` distils the ANNs in-memory before
evaluating; `--weights weights.pt` reuses a saved bundle.

## The reusable API (for the web backend)

`sim_engine.api.EngineService` is the transport-agnostic seam a Flask/FastAPI
layer wraps. It takes plain dicts/lists and returns JSON-serializable dicts —
no torch objects cross the boundary.

```python
from sim_engine.api import EngineService

svc = EngineService({"steps": 250, "train_on_init": True})

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
├── benchmark.py        # closed-loop trajectory evaluation + metrics
├── registry.py         # name -> controller factory (demo brains)
├── engine.py           # Engine + SimulationSession
├── api.py              # reusable JSON-in/JSON-out API (EngineService)
├── serialization.py    # tensor/numpy -> JSON helpers
└── cli.py / __main__.py
examples/run_benchmark.py
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
