from __future__ import annotations

import torch
from torch import nn

from piadaptwin.manifold import PkpdConfig


class MarshCompartments(nn.Module):
    def __init__(self, config: PkpdConfig, v1: float, k10: float, ke0: float) -> None:
        super().__init__()
        self.k12 = config.k12
        self.k13 = config.k13
        self.k21 = config.k21
        self.k31 = config.k31
        self.v1 = v1
        self.k10 = k10
        self.ke0 = ke0

    def _derivative(self, state: torch.Tensor, infusion: torch.Tensor) -> torch.Tensor:
        c1 = state[:, 0]
        c2 = state[:, 1]
        c3 = state[:, 2]
        gamma = state[:, 3]
        outflow = (self.k10 + self.k12 + self.k13) * c1
        dc1 = -outflow + self.k21 * c2 + self.k31 * c3 + infusion / self.v1
        dc2 = self.k12 * c1 - self.k21 * c2
        dc3 = self.k13 * c1 - self.k31 * c3
        dgamma = self.ke0 * (c1 - gamma)
        return torch.stack([dc1, dc2, dc3, dgamma], dim=1)

    def forward(self, infusion: torch.Tensor, dt: float) -> torch.Tensor:
        batch, length = infusion.shape
        state = torch.zeros(batch, 4, device=infusion.device, dtype=infusion.dtype)
        trajectory = []
        for k in range(length):
            u = infusion[:, k]
            k1 = self._derivative(state, u)
            k2 = self._derivative(state + 0.5 * dt * k1, u)
            k3 = self._derivative(state + 0.5 * dt * k2, u)
            k4 = self._derivative(state + dt * k3, u)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            trajectory.append(state[:, 3])
        return torch.stack(trajectory, dim=1)


def effect_site_trajectory(
    infusion: torch.Tensor, config: PkpdConfig, v1: float, k10: float, ke0: float, dt: float
) -> torch.Tensor:
    model = MarshCompartments(config, v1, k10, ke0)
    return model(infusion, dt)
