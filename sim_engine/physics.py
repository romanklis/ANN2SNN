"""Physical system dynamics: a ball rolling on a 2D tilting plate.

This module is the *single source of truth* for the plant model and for every
physical constant used across the benchmark.  It reproduces, bit-for-bit, the
differential equations of the original monolithic prototype
(``reference/prototype_ball_and_plate.py``):

    dvx/dt = -C * theta_x
    dvy/dt = -C * theta_y

integrated with an explicit (forward-Euler) fixed timestep ``DT``.

Because ``C_CONST = (5/7) * g`` is the acceleration of a *solid sphere rolling
without slipping*, the plant has no friction and no damping term of its own —
the controllers are the only thing keeping the ball on the reference orbit.
"""

from __future__ import annotations

import torch

# --------------------------------------------------------------------------- #
# Physical constants (import these rather than re-hardcoding numbers)
# --------------------------------------------------------------------------- #
GRAVITY: float = 9.81
"""Gravitational acceleration [m/s^2]."""

ROLLING_FACTOR: float = 5.0 / 7.0
"""Solid-sphere rolling-without-slipping factor (2/5 inverse)."""

C_CONST: float = ROLLING_FACTOR * GRAVITY
"""Rolling-ball gain ``~7.007`` [m/s^2 per rad of plate tilt]."""

DT: float = 0.02
"""Physics timestep [s] — 20 ms, i.e. a 50 Hz control loop."""

MAX_TILT: float = 0.25
"""Actuator saturation: maximum plate angle [rad] (~14.3 degrees)."""

PLATE_HALF: float = 0.25
"""Half-side of the physical plate [m]; a ball beyond this has left the plate."""

# --------------------------------------------------------------------------- #
# Network dimensions
# --------------------------------------------------------------------------- #
N_IN: int = 6
"""Controller input width: ``[ex, ey, evx, evy, uff_x, uff_y]``.

The last two are the feed-forward command ``u_ff = −â_ref/C``, where ``â_ref`` is
the reference acceleration reconstructed from the Kalman estimate (position-only
camera); the controller never receives the true state.
"""

N_OUT: int = 2
"""Controller output width: ``[theta_x, theta_y]``."""

N_NEURONS: int = 1000
"""Hidden-neuron budget shared by every brain (fair comparison)."""

SYNAPSES_PER_NEURON: int = 40
"""Fan-in of the sparse recurrent connectome."""

TOTAL_SYNAPSES: int = N_NEURONS * SYNAPSES_PER_NEURON
"""Total number of recurrent edges in the connectome."""

STATE_DIM: int = 4
"""Plant state width: ``[x, y, vx, vy]``."""


def clamp_action(
    tilt: torch.Tensor,
    max_tilt: float = MAX_TILT,
) -> torch.Tensor:
    """Saturate a 2-vector actuator command to the physical plate limits."""
    return torch.clamp(tilt, -max_tilt, max_tilt)


def step_physics(
    state: torch.Tensor,
    tilt: torch.Tensor,
    dt: float = DT,
    max_tilt: float = MAX_TILT,
    c_const: float = C_CONST,
    damping: float = 0.0,
    disturbance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance the ball-and-plate plant by one timestep.

    Parameters
    ----------
    state:
        ``[x, y, vx, vy]`` — ball position [m] and velocity [m/s].
    tilt:
        ``[theta_x, theta_y]`` — commanded plate angle [rad]; clamped to
        ``[-max_tilt, +max_tilt]`` (the command saturates, exactly as in the
        prototype).
    dt, max_tilt, c_const:
        Overrides for the physical constants; the defaults are the canonical
        values exported by this module.
    damping:
        Velocity damping ``b`` [1/s]; the body term is ``a -= b * v``.  The
        default ``0.0`` leaves the original frictionless model untouched.
    disturbance:
        Optional ``[dx, dy]`` external acceleration [m/s^2] applied this step
        (perturbations / process noise).  Default ``None`` = no disturbance.

    Returns
    -------
    torch.Tensor
        The next state ``[x, y, vx, vy]`` as a new tensor (the input is not
        mutated).
    """
    tilt_clamped = torch.clamp(tilt, -max_tilt, max_tilt)

    # dv/dt = -C * theta  (per axis)
    ax = -c_const * tilt_clamped[0]
    ay = -c_const * tilt_clamped[1]

    # Optional body damping (a -= b * v).  Guarded so damping=0 is bit-exact.
    if damping:
        ax = ax - damping * state[2]
        ay = ay - damping * state[3]

    # Optional external disturbance acceleration.
    if disturbance is not None:
        d = torch.as_tensor(disturbance, dtype=state.dtype, device=state.device)
        ax = ax + d[0]
        ay = ay + d[1]

    vx_next = state[2] + ax * dt
    vy_next = state[3] + ay * dt
    x_next = state[0] + vx_next * dt
    y_next = state[1] + vy_next * dt

    return torch.stack([x_next, y_next, vx_next, vy_next])


def rollout(
    state: torch.Tensor,
    tilts: torch.Tensor,
    dt: float = DT,
    max_tilt: float = MAX_TILT,
) -> torch.Tensor:
    """Batch helper: roll a plant forward over a pre-computed tilt sequence.

    ``tilts`` has shape ``(T, 2)``; the returned trajectory has shape
    ``(T, 4)`` and *excludes* the initial state (row ``k`` is the state after
    applying ``tilts[k]``).
    """
    traj = []
    cur = state
    for k in range(tilts.shape[0]):
        cur = step_physics(cur, tilts[k], dt=dt, max_tilt=max_tilt)
        traj.append(cur)
    return torch.stack(traj) if traj else torch.empty(0, STATE_DIM)


class BallPlatePlant:
    """Stateful convenience wrapper around :func:`step_physics`.

    Useful for interactive/web use where a caller wants an object it can
    ``reset()`` and ``step(action)`` one frame at a time.
    """

    def __init__(
        self,
        init_state=( -0.05, 0.05, 0.0, 0.0),
        dt: float = DT,
        max_tilt: float = MAX_TILT,
        c_const: float = C_CONST,
        damping: float = 0.0,
        device="cpu",
        dtype=torch.float32,
    ) -> None:
        self.init_state = torch.as_tensor(init_state, dtype=dtype, device=device)
        self.dt = dt
        self.max_tilt = max_tilt
        self.c_const = c_const
        self.damping = damping
        self.device = torch.device(device)
        self.dtype = dtype
        self.state = self.init_state.clone()
        self.t = 0

    def reset(self) -> torch.Tensor:
        """Return the plant to its initial state and reset the step counter."""
        self.state = self.init_state.clone()
        self.t = 0
        return self.state.clone()

    def step(self, tilt: torch.Tensor, disturbance: torch.Tensor | None = None) -> torch.Tensor:
        """Apply one actuator command and advance the plant by ``dt``."""
        tilt = torch.as_tensor(tilt, dtype=self.dtype, device=self.device)
        self.state = step_physics(
            self.state,
            tilt,
            dt=self.dt,
            max_tilt=self.max_tilt,
            c_const=self.c_const,
            damping=self.damping,
            disturbance=disturbance,
        )
        self.t += 1
        return self.state.clone()

    @property
    def position(self) -> torch.Tensor:
        return self.state[:2]
