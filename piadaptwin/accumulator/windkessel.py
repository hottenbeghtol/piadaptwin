from __future__ import annotations

import torch


def _as_column(value: torch.Tensor | float, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.to(device=device)
        if out.ndim == 0:
            out = out.expand(batch, 1)
        elif out.ndim == 1:
            out = out.reshape(batch, 1)
        return out
    return torch.full((batch, 1), float(value), device=device)


def two_element_residual_terms(
    pressure: torch.Tensor,
    cardiac_output: torch.Tensor,
    compliance: torch.Tensor | float,
    resistance: torch.Tensor | float,
    dt: float,
) -> torch.Tensor:
    batch = pressure.shape[0]
    device = pressure.device
    comp = _as_column(compliance, batch, device)
    res = _as_column(resistance, batch, device)
    dpdt = (pressure[:, 2:] - pressure[:, :-2]) / (2.0 * dt)
    interior_p = pressure[:, 1:-1]
    interior_q = cardiac_output[:, 1:-1]
    return comp * dpdt + interior_p / res - interior_q


def windkessel_residual(
    pressure: torch.Tensor,
    cardiac_output: torch.Tensor,
    compliance: torch.Tensor | float,
    resistance: torch.Tensor | float,
    dt: float,
) -> torch.Tensor:
    terms = two_element_residual_terms(pressure, cardiac_output, compliance, resistance, dt)
    return torch.mean(terms.pow(2))


def windkessel_pressure_rollout(
    cardiac_output: torch.Tensor,
    compliance: torch.Tensor | float,
    resistance: torch.Tensor | float,
    dt: float,
    p0: torch.Tensor | float,
) -> torch.Tensor:
    batch, length = cardiac_output.shape
    device = cardiac_output.device
    comp = _as_column(compliance, batch, device)
    res = _as_column(resistance, batch, device)
    p_init = _as_column(p0, batch, device)
    pressures = [p_init]
    current = p_init
    for k in range(1, length):
        drift = (cardiac_output[:, k - 1 : k] - current / res) / comp
        current = current + dt * drift
        pressures.append(current)
    return torch.cat(pressures, dim=1)
