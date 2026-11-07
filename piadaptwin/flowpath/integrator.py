from __future__ import annotations

from collections.abc import Callable

import torch

DerivFn = Callable[[float, torch.Tensor], torch.Tensor]

_C = (0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0)
_A = (
    (),
    (0.2,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)
_B5 = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0)
_B4 = (
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
)


def _dopri_step(
    func: DerivFn, t: float, z: torch.Tensor, h: float
) -> tuple[torch.Tensor, torch.Tensor]:
    k = [func(t, z)]
    for stage in range(1, 7):
        acc = z
        for j in range(stage):
            acc = acc + h * _A[stage][j] * k[j]
        k.append(func(t + _C[stage] * h, acc))
    y5 = z
    y4 = z
    for j in range(7):
        y5 = y5 + h * _B5[j] * k[j]
        y4 = y4 + h * _B4[j] * k[j]
    return y5, y4


def dopri5_integrate(
    func: DerivFn,
    z0: torch.Tensor,
    times: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    max_steps: int = 512,
) -> torch.Tensor:
    outputs = [z0]
    z = z0
    grid = [float(v) for v in times.tolist()]
    for idx in range(1, len(grid)):
        t_now = grid[idx - 1]
        t_end = grid[idx]
        span = t_end - t_now
        h = span
        steps = 0
        while t_now < t_end - 1e-9 and steps < max_steps:
            h = min(h, t_end - t_now)
            y5, y4 = _dopri_step(func, t_now, z, h)
            with torch.no_grad():
                scale = atol + rtol * torch.maximum(z.abs(), y5.abs())
                err = torch.sqrt(torch.mean(((y5 - y4) / scale) ** 2)).item()
            if err <= 1.0 or h <= 1e-6:
                t_now = t_now + h
                z = y5
                factor = 2.0 if err == 0.0 else min(2.0, 0.9 * err ** (-0.2))
                h = h * factor
            else:
                h = h * max(0.2, 0.9 * err ** (-0.25))
            steps += 1
        outputs.append(z)
    return torch.stack(outputs, dim=1)
