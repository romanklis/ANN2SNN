# `sim_engine` reusable API

Transport-agnostic surface for driving the engine from a web backend. Import
`EngineService` and map HTTP routes onto it; nothing here imports a web
framework, and every return value is already JSON-serializable.

```python
from sim_engine.api import EngineService, list_controllers, run_benchmark, build_engine, config_from_dict
```

---

## 1. Configuration dict

All entry points accept a partial config dict; missing keys fall back to the
canonical defaults. Nested groups (`plant`, `network`, `benchmark`, `training`)
are merged shallowly; a few flat convenience keys are also accepted.

```jsonc
{
  "steps": 250,            // control frames in the orbit  (flat -> benchmark.steps)
  "radius": 0.15,          // orbit radius [m]
  "freq": 0.5,             // orbit frequency [Hz]
  "init_state": [-0.05, 0.05, 0.0, 0.0],
  "micro_steps": 10,       // SNN sub-steps per frame
  "n_neurons": 1000,       // hidden neurons per brain
  "epochs": 100,           // distillation epochs
  "seed": 42,
  "device": "cpu",         // "cpu" | "cuda" | "auto"
  "train_on_init": false,  // distil the ANNs at construction
  "weights_path": "weights.pt",
  "default_controllers": ["pid", "random_ann", "flylike_ann", "snn_transferred"],

  "plant":     {"dt": 0.02, "max_tilt": 0.25},
  "network":   {"synapses_per_neuron": 40, "inhibitory_fraction": 0.2},
  "benchmark": {"record_spikes": true},
  "training":  {"batch_size": 64, "lr": 0.008, "connectome_unroll": 3, "device": "cpu"}
}
```

`config_from_dict(d) -> EngineConfig`, `default_config() -> dict`,
`build_engine(d, train=None) -> Engine`.

---

## 2. Controller catalogue

`list_controllers()` (stateless) or `service.controllers()` (from a live engine)
returns:

```json
[
  {"name": "pid", "label": "PID controlled", "aliases": ["pd", "classical", "..."]},
  {"name": "random_ann", "label": "Random ANN", "aliases": ["random", "..."]},
  {"name": "flylike_ann", "label": "Fly-like ANN (connectome)", "aliases": ["connectome", "..."]},
  {"name": "snn_transferred", "label": "SNN transferred", "aliases": ["snn", "..."]},
  {"name": "dense_ann", "label": "Dense ANN (distilled)", "aliases": ["dense", "..."]}
]
```

`service.describe()` additionally returns each controller's architecture,
topology summary and a `trained` flag.

Names are resolved forgivingly (`-`/space → `_`, lowercase, alias map), so
`"SNN"`, `"snn"`, `"lossless-snn"` all resolve to `snn_transferred`.

---

## 3. Batch benchmark

`service.benchmark(controllers=[...], include_trace=True) -> dict`

```jsonc
{
  "reference": {"kind": "orbit", "steps": 250, "dt": 0.02, "radius": 0.15, "freq": 0.5,
                "pos": [[x,y], ...], "vel": [[vx,vy], ...], "acc": [[ax,ay], ...]},
  "init_state": [-0.05, 0.05, 0.0, 0.0],
  "config": { ... resolved config ... },
  "ranking": ["pid", "snn_transferred", "flylike_ann", "random_ann"],
  "results": {
    "pid": {
      "name": "pid",
      "metrics": {"mean_error_cm": 2.426, "rms_error_cm": 5.276,
                  "max_error_cm": 20.616, "final_error_cm": 0.468,
                  "settling_step": 58},
      "trajectory": [[x, y, vx, vy], ...],   // (T, 4)
      "tilts": [[tx, ty], ...],              // (T, 2)
      "tracking_error": [cm, ...],           // (T,)
      "spikes": [[0/1, ...], ...],           // (T, N) only for spiking brains
      "meta": {"controller": { ... describe() ... }}
    }
  }
}
```

Set `include_trace=False` to omit `trajectory`/`tilts`/`tracking_error`/`spikes`
(small payloads for listing metrics only).

---

## 4. Interactive sessions

For the animated frontend: one server-side session = one plant + one controller
+ a cursor into the reference orbit.

| Call | Returns |
|---|---|
| `service.new_session(controller, steps=..., radius=..., freq=..., init_state=...)` | session descriptor incl. `session_id` |
| `service.session_info(sid)` | descriptor |
| `service.step(sid, action=None, n=1)` | one observation (after `n` frames) |
| `service.reset(sid)` | observation at step 0 |
| `service.set_controller(sid, name)` | new descriptor |
| `service.trajectory(sid)` | recorded trace + `mean_error_cm` |
| `service.close_session(sid)` | frees it |

Session descriptor:

```json
{"session_id": "...", "controller": "snn_transferred", "label": "SNN transferred",
 "steps_total": 250, "step": 0, "dt": 0.02, "radius": 0.15, "freq": 0.5,
 "init_state": [-0.05, 0.05, 0.0, 0.0], "controller_info": { ... }}
```

Observation (one frame):

```json
{"session_id": "...", "controller": "snn_transferred", "t": 0.02, "step": 1,
 "done": false,
 "state":   [x, y, vx, vy],
 "target":  [x_ref, y_ref],
 "reference": {"pos": [x,y], "vel": [vx,vy], "acc": [ax,ay]},
 "tilt":    [theta_x, theta_y],
 "error_cm": 20.38,
 "spikes":  [0, 1, 0, ...]   // 1000 entries for spiking brains, else null
}
```

`step(sid, action=[tx, ty])` overrides the controller — useful for a manual
"joystick" mode in the UI. `step(sid, n=10)` advances ten frames and returns the
last observation (handy for coalescing animation ticks).

Session state lives in the `Engine`'s in-memory registry; the backend should
treat `session_id` as an opaque handle (drop idle sessions with
`close_session`).

---

## 5. Error handling

Unknown controller names raise `KeyError` from the registry and are re-wrapped
as `EngineError` when raised through `Engine.build_controller`. Unknown session
ids raise `EngineError`. Map both to HTTP 400/404 in the backend.

---

## 6. Suggested route map

| Method | Route | Service call |
|---|---|---|
| `GET` | `/api/controllers` | `service.controllers()` |
| `GET` | `/api/engine` | `service.describe()` |
| `POST` | `/api/benchmark` | `service.benchmark(body["controllers"])` |
| `POST` | `/api/sessions` | `service.new_session(body["controller"], **opts)` |
| `GET` | `/api/sessions/<sid>` | `service.session_info(sid)` |
| `POST` | `/api/sessions/<sid>/step` | `service.step(sid, action, n)` |
| `POST` | `/api/sessions/<sid>/reset` | `service.reset(sid)` |
| `PUT` | `/api/sessions/<sid>/controller` | `service.set_controller(sid, name)` |
| `GET` | `/api/sessions/<sid>/trajectory` | `service.trajectory(sid)` |
| `DELETE` | `/api/sessions/<sid>` | `service.close_session(sid)` |
