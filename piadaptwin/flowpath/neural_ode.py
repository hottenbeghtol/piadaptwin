from __future__ import annotations

import torch
from torch import nn

from piadaptwin.flowpath.derivative import ResidualField
from piadaptwin.flowpath.integrator import dopri5_integrate


class NeuralODEBlock(nn.Module):
    def __init__(self, hidden: int, layers: int, rtol: float, atol: float) -> None:
        super().__init__()
        self.field = ResidualField(hidden, layers)
        self.rtol = rtol
        self.atol = atol

    def forward(self, z0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return dopri5_integrate(self.field, z0, times, rtol=self.rtol, atol=self.atol)
