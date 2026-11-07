from __future__ import annotations

import torch
from torch import nn


class ResidualField(nn.Module):
    def __init__(self, hidden: int, layers: int) -> None:
        super().__init__()
        self.input = nn.Linear(hidden + 1, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.output = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        time_col = torch.full((z.shape[0], 1), float(t), device=z.device, dtype=z.dtype)
        h = self.act(self.input(torch.cat([z, time_col], dim=1)))
        for block in self.blocks:
            h = h + self.act(block(h))
        return self.output(h)
