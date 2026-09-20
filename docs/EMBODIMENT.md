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
| sensing | `sensor_noise_pos` | Gaussian σ on the measured position [m] |
| sensing | `sensor_delay` | observation delay [control frames] |
| estimation | `estimator` | always `kalman` — velocity is **not** measured |
| estimation | `estimate_process_noise` / `estimate_init_pos_var` / `estimate_init_vel_var` | filter tuning |
| actuation | `actuator_delay` | command delay [control frames] |
| actuation | `actuator_gain` / `actuator_bias` | scale / offset of the plate command |
| body | `damping` | velocity damping `b` [1/s] (`a -= b·v`) |
| body | `c_scale` | effective rolling gain `C = C_CONST · c_scale` (mass/inertia) |
| perturbation | `process_noise` | continuous **acceleration** jitter σ [m/s²] |
| perturbation | `impulse_interval` / `impulse_prob` | when a one-off kick is applied |
| perturbation | `impulse_std` | **velocity** kick σ [m/s] (applied directly to the ball's velocity) |
| randomisation | `randomize` + ranges | resample `c_scale`/`damping` each `reset()` |

The closed loop becomes the canonical estimation/control separation — the
controller **never sees the true state**:

```
x_{k+1} = f(x_k, u_k) + w_k
y_k     = h(x_k) + v_k            h(x) = [x, y]      (camera: position only)
x̂_k     = E(y_{0:k}, u_{0:k-1})                     E = Kalman filter
e_k     = x̂_k − r_k
u_ff,k  = −â_ref,k / C            â_ref reconstructed from x̂ (known rolling gain C)
u_k     = π_θ([e_k, u_ff,k])      6-D policy input (error + feed-forward)
obs  = env.measure(state)          # y: position-only, noisy, delayed
x̂    = kalman.update(y, u_prev)    # full-state estimate
u    = π(x̂ − r)                    # policy acts on the estimate
u_eff = env.actuate(u)             # gained, delayed, clamped
state = env.step(state, u_eff, k)  # damping + C·scale + disturbance
```

The Kalman filter uses the nominal plant model (`A`, `B`, `H=[I 0]`) and is
**delay-aware** (it corrects the buffered prior at the measurement's time index,
then re-propagates).

What the filter **does** model: the nominal double-integrator plant, the sensor
delay, and the measurement noise (`R` matched to `sensor_noise_pos`).
What it does **not** model: actuator delay, actuator gain/bias, damping and
`c_scale`. Those mismatches are deliberate — estimation degrades gracefully there,
which is the honest robustness signal.

The recorded `trajectory`/`tracking_error` are always the **true** plant state;
`measurements`, `estimates`, `estimation_pos_rmse_cm` and `estimation_vel_rmse`
expose the observation/estimation side. Distillation (`clean` **and** `robust`)
generates labels from closed-loop teacher rollouts through the same filter, so the
students learn `π(x̂ − r)`; the feed-forward channel `u_ff` is likewise
reconstructed from `x̂` (see `ReferenceAccelEstimator`), never from ground truth.

## Presets (`EMBODIMENT_PRESETS`)

`clean` (ideal sensing but the Kalman estimator still reconstructs velocity),
`noisy` (position σ 5 mm), `delayed` (sensor 3 / actuator 2 frames ≈ 60/40 ms),
`perturbed` (a 0.15 m/s **velocity** kick every 60 frames), `heavy` (damping 0.6,
c_scale 0.8), `embodied` (σ 4 mm + sensor 2 / actuator 1 frames + a 0.12 m/s kick
every 80 frames + damping 0.3 + c_scale 0.9 — the dashboard default), `randomized`
(seeded per-episode randomisation).

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
The hero stage draws the **camera measurement** (faint dot), the **true** ball,
and the controller's **estimate** (hollow ring); the pipeline includes
`CAMERA (y) → KALMAN (x̂)` before the controller, and the result bar reports the
estimation RMSE.

## Fairness rule

`evaluate()` creates one `EmbodiedEnv` per controller with the **same seed**, and
the environment's RNG is driven only by the loop (not by the actions), so all
controllers in a run see an identical disturbance stream.
