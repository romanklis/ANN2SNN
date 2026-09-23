# Examples (swappable plants & tasks)

An **example** is everything that is specific to *what is being controlled*: the
plant step, the task/reference, the nominal model the Kalman filter uses, the
error metric, the success bounds and the UI labels/units. Everything else — the
controller contract, closed-loop distillation, the ANN→SNN transfer, sessions,
the Flask API — is example-agnostic.

Three orthogonal axes:

```
Example     (Balancing ball · Hovering drone · …)      what plant/task is simulated
Environment (clean · noisy · delayed · perturbed · …)  the embodiment presets
Profile     (clean · robust)                           which policy (and weights)
```

## Built-ins

| Example | Plant | Command | Task | Sensor | State |
|---|---|---|---|---|---|
| `ball` (default) | ball rolling on a tilting plate | `[θx, θy]` rad, ≤ 0.25 | track a circular orbit (`R=0.15 m`, `0.5 Hz`) | position only | `[x, y, vx, vy]` |
| `drone` | 3-D point mass | `[ax, ay, az]` m/s², ≤ 3.0 | lissajous flight path (`A=0.6 m`, `0.25 Hz`) or hover | position only | `[x, y, z, vx, vy, vz]` |

Both use the same 6-D/9-D policy input `[error (2D), u_ff (D)]`, the same
delay-aware Kalman filter, the same closed-loop teacher (`PD + feed-forward`) and
the same distillation. The plate's command→acceleration gain is negative (`−C`),
the drone's is positive (`+1`); the *signed* plant gain is the single field that
carries that difference, and the teacher/labels/estimator all derive from it.

## Using an example

```bash
python -m sim_engine list                                   # brains
python -m sim_engine benchmark --example drone --steps 500  # 3-D lissajous
python -m sim_engine benchmark --example drone --radius 0.4 --freq 0.2
python -m sim_engine train --example drone --profile robust --save weights_drone.pt
python -m sim_engine session --example drone --controller snn_transferred --frames 20
```

```python
from sim_engine import Engine, EngineConfig, BenchmarkConfig

engine = Engine(EngineConfig(example="drone", train_on_init=True))
report = engine.run_benchmark(["pid", "flylike_ann", "snn_transferred"])
```

HTTP: `POST /api/benchmark {"example": "drone", "controllers": [...]}`;
`GET /api/controllers` returns `examples`, `example_names` and `default_example`.
Weight bundles are cached per `(example, profile, seed)`.

`--radius`/`--freq` mean "trajectory amplitude / frequency" for any example, so
the same knobs work for both.

## Drone specifics

* **Command**: acceleration (thrust-vector) in `[−3, 3]` m/s² per axis; the
  plant is a plain 3-D double integrator (`v' = v + u·dt`, `p' = p + v'·dt`).
* **References**: `lissajous_reference` (default; `x`/`y` ellipse plus a `z`
  oscillation) and `setpoint_reference` (minimum-jerk move to a hover point).
* **Bounds**: success is the corridor `‖p‖∞ ≤ 1 m`; the metric is `‖p − r‖` in cm.
* **No attitude**: there is deliberately no attitude, motor model or cascaded
  control. A full quadrotor (12-state, 4 motor inputs, nonlinear) would invalidate
  the linear Kalman filter and the analytic teacher this project is built on and
  is **out of scope**.

## Adding a new example

Write one module and register it — no engine change:

```python
from sim_engine.examples import ExampleSpec, register_example
from sim_engine.physics import DT

def _glider_step(state, command, *, dt, limit, gain, damping, disturbance):
    ...  # your plant step; return the next state

register_example(ExampleSpec(
    name="glider",
    label="Gliding wing",
    pos_dim=2,
    control_limit=0.4,
    plant_gain=-1.0,             # signed control -> acceleration
    init_state=(0.0, 0.3, 0.0, 0.0),
    step_fn=_glider_step,
    reference_fn=my_reference,
    bounds_low=(-1.0, -1.0),
    bounds_high=(1.0, 1.0),
    units={"command": "rad", "error": "cm", "position": "m", "velocity": "m/s"},
    labels={"plant": "GLIDER WING", "command": "ELEVATOR",
            "error": "TRACKING ERROR", "success": "IN AIR"},
    renderer="wing",
    defaults={"radius": 0.5, "freq": 0.3},
))
```

The dashboard renders two built-in stage views: `plate` (the isometric ball/plate)
and `quad` (the 3-D corridor flight view); any other `renderer` value falls back to
`plate`. New examples therefore show up in the selector and the charts immediately,
and only need a new stage view if their geometry differs from both.

The controller factories, the policy input, the Kalman filter and the teacher are
all sized from `pos_dim`, `plant_gain` and `control_limit`, so a new example gets
the same estimator/policy/distillation pipeline for free. `tests/test_examples.py`
shows the shape of the tests to add (plant response, reference bounds, filter
convergence, a short closed loop).
