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
| `drone_gps_denied` | 3-D point mass | `[ax, ay, az]` m/s², ≤ 3.0 | same lissajous path | **IMU + barometer + optical flow + checkpoint fixes** | `[x, y, z, vx, vy, vz]` |

All three use the same 6-D/9-D policy input `[error (2D), u_ff (D)]`, the same
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

## GPS-denied drone

Absolute position is the least realistic thing a drone can measure indoors, so
`drone_gps_denied` replaces the camera with an **onboard sensor suite** (declared on
the example as a `SensorSpec`, so it is data, not code):

| Channel | Kind | Measures | How it enters the filter | Noise | Latency |
|---|---|---|---|---|---|
| accelerometer | `imu` | body acceleration `a = u_eff + d` | **prediction input** (`B·a_meas`), not an update | 0.05 m/s² | 0 frames |
| barometer | `altitude` | absolute `z` | `H = e_z` on position | 0.02 m | 2 frames |
| optical flow | `flow_velocity` | horizontal velocity `(vx, vy)` (flow × altitude) | `H = [0, I]` on velocity | 0.03 m/s | 3 frames |
| checkpoint fix | `position_fix` | relative position `p − c` to the nearest beacon | `H = [I, 0]` on position | 0.05 m | 5 frames |

* The accelerometer is what makes an actuator gain/bias an *estimation* problem
  rather than a pure penalty: the filter propagates with the acceleration the body
  actually achieved instead of the command it was given. A measurement of the input
  would have `H ≈ 0` and update nothing, which is why it drives the prediction.
* Three ground beacons sit on a 0.9 m ring around the 0.6 m flight circle; a fix is
  only produced when the drone is within the gate (`‖p − c‖ ≤ 0.45 m`), which
  happens about seven times per 500-frame episode. Between fixes the estimate
  drifts and each fix snaps it back — the dashboard draws the belief `x̂` as a ghost
  marker so the sawtooth is visible.
* The launch pad seeds `p` (a surveyed take-off point), and the flow channel keeps
  the horizontal estimate observable in between: **inertial-only drifts without
  bound, flow bounds it, the fixes tighten it**. `tests/test_sensors.py` asserts that
  ordering (the observability ladder).
* Deliberate simplification: a fix is a *relative-position* fix (range **and**
  bearing, e.g. a fiducial or trilaterated anchors). A bare range `‖p − c‖` is
  nonlinear and would require an EKF, which would invalidate the linear-filter
  design. There is also no attitude state: a compass is assumed to resolve the IMU
  into the world frame, and there is no gravity, motor model or blade dynamics.

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
from sim_engine.sensors import SensorSpec, camera_channel

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
    sensor=SensorSpec(channels=(camera_channel(2),)),   # omit -> same camera
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
