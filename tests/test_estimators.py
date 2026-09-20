"""Kalman-filter tests: convergence, noise rejection and delay handling."""

from __future__ import annotations

import numpy as np
import torch

from sim_engine.estimators import KalmanFilter
from sim_engine.physics import DT, step_physics


def _roll(steps: int, u_seq: torch.Tensor, x0=(0.05, -0.05, 0.0, 0.0)) -> torch.Tensor:
    """True plant states ``(steps+1, 4)`` for a given tilt sequence."""
    x = torch.tensor(x0, dtype=torch.float32)
    out = [x.clone()]
    for k in range(steps):
        x = step_physics(x, u_seq[k], dt=DT)
        out.append(x.clone())
    return torch.stack(out)


def _run_kf(truth: torch.Tensor, u_seq: torch.Tensor, kf: KalmanFilter, delay: int = 0):
    kf.reset()
    est, u_prev = [], torch.zeros(2)
    n = len(u_seq)
    for k in range(n):
        src = k - delay
        if src >= 0:
            y = truth[src][:2].clone()
        else:
            y = truth[0][:2].clone()
        est.append(kf.update(y, u_prev).clone())
        u_prev = u_seq[k]
    return torch.stack(est)          # (n, 4), aligned to truth[0..n-1]


def test_kf_converges_on_noiseless_measurements():
    steps = 120
    u = torch.tensor([[0.05, -0.03]]).repeat(steps, 1)
    truth = _roll(steps, u)
    kf = KalmanFilter(dt=DT, process_noise=1e-3, meas_noise=1e-4)
    est = _run_kf(truth, u, kf)
    pos_err = (est[-1, :2] - truth[steps - 1, :2]).abs().max().item()
    vel_err = (est[-1, 2:] - truth[steps - 1, 2:]).abs().max().item()
    assert pos_err < 1e-2
    assert vel_err < 1e-1


def test_kf_rejects_measurement_noise():
    steps = 300
    sigma = 0.01
    u = torch.tensor([[0.03, -0.02]]).repeat(steps, 1)
    truth = _roll(steps, u)
    rng = np.random.default_rng(0)

    kf = KalmanFilter(dt=DT, process_noise=1e-2, meas_noise=sigma)
    kf.reset()
    u_prev = torch.zeros(2)
    est, raw = [], []
    for k in range(steps):
        y = truth[k][:2] + torch.tensor(rng.normal(0.0, sigma, size=2), dtype=torch.float32)
        raw.append(y.clone())
        est.append(kf.update(y, u_prev).clone())
        u_prev = u[k]
    est = torch.stack(est)[-150:]
    raw = torch.stack(raw)[-150:]
    ref = truth[:steps][-150:, :2]

    kf_rmse = float(torch.sqrt(((est[:, :2] - ref) ** 2).sum(dim=1).mean()))
    raw_rmse = float(torch.sqrt(((raw - ref) ** 2).sum(dim=1).mean()))
    assert kf_rmse < raw_rmse          # filtering beats the raw measurement
    assert kf_rmse < 2 * sigma


def test_kf_handles_sensor_delay():
    steps = 200
    delay = 3
    u = torch.tensor([[0.02, 0.01]]).repeat(steps, 1)
    truth = _roll(steps, u)
    kf = KalmanFilter(dt=DT, process_noise=1e-2, meas_noise=1e-3, delay=delay)
    est = _run_kf(truth, u, kf, delay=delay)
    err = (est[-100:, :2] - truth[:steps][-100:, :2]).norm(dim=1) * 100.0  # cm
    assert float(err.mean()) < 3.0      # delay-aware filter stays on top of the state
