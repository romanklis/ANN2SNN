# Embodied control: noise · delay · perturbations · body dynamics

The original benchmark handed controllers the exact plant state. This layer makes
the **body and environment part of the problem**: controllers observe a noisy,
delayed state, their commands pass through a delayed/gained actuator, the ball is
subject to damping and a scaled rolling gain, and the plate is hit by
perturbations.

Everything is deterministic: given a seed, every controller evaluated in the same
cell faces the **same disturbance realisation** (same noise values, same impulse
timing/magnitudes), so comparisons are fair.

## Environment model (`sim_engine/environment.py`)

| group | knob | meaning |
|---|---|---|
| sensing | `sensor_noise_pos` / `sensor_noise_vel` | Gaussian σ on measured position [m] / velocity [m/s] |
| sensing | `sensor_delay` | observation delay [control frames] |
| actuation | `actuator_delay` | command delay [control frames] |
| actuation | `actuator_gain` / `actuator_bias` | scale / offset of the plate command |
| body | `damping` | velocity damping `b` [1/s] (`a -= b·v`) |
| body | `c_scale` | effective rolling gain `C = C_CONST · c_scale` (mass/inertia) |
| perturbation | `process_noise` | continuous acceleration jitter σ [m/s²] |
| perturbation | `impulse_interval` / `impulse_prob` / `impulse_std` | scheduled / random kicks [m/s] |
| randomisation | `randomize` + ranges | resample `c_scale`/`damping` each `reset()` |

The closed loop becomes:

```
obs  = env.observe(state)          # noisy, delayed
u    = controller.act(obs, ref_k)
u_eff = env.actuate(u)             # gained, delayed, clamped
state = env.step(state, u_eff, k)  # damping + C·scale + disturbance
```

The recorded `trajectory`/`tracking_error` are always the **true** plant state.

## Presets (`EMBODIMENT_PRESETS`)

`clean` (the original), `noisy`, `delayed` (sensor 3 / actuator 2 frames ≈ 60/40 ms),
`perturbed` (impulse every 60 frames, σ 0.15 m/s), `heavy` (damping 0.6, c_scale 0.8),
`embodied` (mild noise + delay + impulses + lossy body — the dashboard default),
`randomized` (seeded per-episode randomisation).

## Training profiles

* **`clean`** — the original i.i.d. error distillation of the PD teacher.
* **`robust`** — `training.distill_embodied` rolls the PD teacher **inside** the
  embodied environment (the teacher sees the same noisy/delayed observation and
  acts through the same actuator) and trains the dense ANN on pooled pairs and the
  recurrent connectome on **contiguous windows**, so the recurrence can in
  principle integrate the delayed observation stream.

Weight bundles are cached per `(profile, seed)`:
`weights.pt` (clean) and `weights_robust.pt` (robust).

## Robustness sweeps (`sim_engine/robustness.py`)

`sweep(build_controllers, axis)` evaluates controllers across one axis:
`preset` · `noise` · `delay` · `impulse`. Bounded (≤12 points, ≤5 controllers).

```bash
python -m sim_engine train --profile robust --embodiment embodied --save weights_robust.pt
python -m sim_engine benchmark --controllers pid,flylike_ann,snn_transferred --embodiment embodied
python -m sim_engine robustness --axis delay --out robustness.json
```

## API

`POST /api/simulate`, `/api/benchmark` accept `embodiment` (preset name, config
object, or `true` for `embodied`) and `profile` (`clean`|`robust`).
`POST /api/robustness {controllers, axis, steps, points?}` runs a sweep.
`GET /api/controllers` returns `embodiment_presets`, `profiles` and
`robustness_axes`. Bounds: noise ≤ 0.05 m, delay ≤ 20 frames, impulse σ ≤ 1 m/s,
damping ≤ 5, `c_scale ∈ [0.3, 2]`.

## Dashboard

The header has an **environment selector** (Clean · Noisy · Delayed · Perturbed ·
Heavy · Embodied · Randomized, default **Embodied**) and a `clean`/`robust`
policy pill. Selecting an embodied preset uses the **robust** policy
(auto-distilled and cached on first use). A thin **robustness strip** in the
footer shows the ANN/SNN mean error across every preset for the current policy.

## Fairness rule

`evaluate()` creates one `EmbodiedEnv` per controller with the **same seed**, and
the environment's RNG is driven only by the loop (not by the actions), so all
controllers in a run see an identical disturbance stream.
