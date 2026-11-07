from __future__ import annotations

import torch
from torch import nn


class WaveformEncoder(nn.Module):
    def __init__(
        self, in_channels: int, filters: int, kernel: int, layers: int, hidden: int
    ) -> None:
        super().__init__()
        pad = kernel // 2
        convs: list[nn.Module] = []
        prev = in_channels
        for _ in range(layers):
            convs.append(nn.Conv1d(prev, filters, kernel, stride=2, padding=pad))
            convs.append(nn.GroupNorm(1, filters))
            convs.append(nn.SiLU())
            prev = filters
        self.stack = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(filters, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stack(x)
        pooled = self.pool(features).squeeze(-1)
        return self.project(pooled)
