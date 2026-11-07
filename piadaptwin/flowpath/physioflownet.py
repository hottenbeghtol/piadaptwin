from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from piadaptwin.flowpath.decoder import StateDecoder
from piadaptwin.flowpath.encoder import WaveformEncoder
from piadaptwin.flowpath.neural_ode import NeuralODEBlock
from piadaptwin.manifold import FlowConfig


@dataclass
class FlowOutput:
    states: torch.Tensor
    pressure: torch.Tensor
    cardiac_output: torch.Tensor
    horizon_states: torch.Tensor
    dt_seconds: float


def _horizon_indices(
    horizons: tuple[int, ...], step_seconds: float, grid_points: int
) -> tuple[int, ...]:
    indices = []
    for horizon in horizons:
        idx = int(round(horizon / step_seconds)) - 1
        indices.append(max(0, min(grid_points - 1, idx)))
    return tuple(indices)


class PhysioFlowNet(nn.Module):
    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = WaveformEncoder(
            config.in_channels, config.enc_filters, config.kernel, config.enc_layers, config.hidden
        )
        self.dynamics = NeuralODEBlock(config.hidden, config.deriv_layers, config.rtol, config.atol)
        self.decoder = StateDecoder(config.hidden, config.state_dim)
        self.step_seconds = config.window_seconds / config.grid_points
        grid = (
            torch.linspace(self.step_seconds, config.window_seconds, config.grid_points)
            / config.window_seconds
        )
        self.times: torch.Tensor
        self.register_buffer("times", torch.cat([torch.zeros(1), grid]))
        self.horizon_index = _horizon_indices(
            config.horizons, self.step_seconds, config.grid_points
        )

    def forward(self, waveform: torch.Tensor) -> FlowOutput:
        batch = waveform.shape[0]
        z0 = self.encoder(waveform)
        trajectory = self.dynamics(z0, self.times)
        future = trajectory[:, 1:, :]
        flat = future.reshape(batch * future.shape[1], future.shape[2])
        states_flat, flow_flat = self.decoder(flat)
        states = states_flat.reshape(batch, future.shape[1], self.config.state_dim)
        flow = flow_flat.reshape(batch, future.shape[1])
        pressure = states[:, :, 2]
        horizon_states = states[:, self.horizon_index, :]
        return FlowOutput(
            states=states,
            pressure=pressure,
            cardiac_output=flow,
            horizon_states=horizon_states,
            dt_seconds=self.step_seconds,
        )
