from __future__ import annotations

import torch
from torch import nn


class StateDecoder(nn.Module):
    def __init__(self, hidden: int, state_dim: int) -> None:
        super().__init__()
        self.state = nn.Linear(hidden, state_dim)
        self.flow = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.state(z)
        flow = torch.nn.functional.softplus(self.flow(z)).squeeze(-1)
        return states, flow
