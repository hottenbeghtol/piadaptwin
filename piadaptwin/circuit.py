from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from piadaptwin.flowpath.physioflownet import FlowOutput, PhysioFlowNet
from piadaptwin.governor.posterior import DiagonalPosterior
from piadaptwin.intake.batch import ClinicalBatch
from piadaptwin.manifold import THETA_NAMES, ExperimentConfig
from piadaptwin.metering.spectrodepthnet import MeterOutput, SpectroDepthNet


@dataclass
class CircuitOutput:
    flow: FlowOutput
    meter: MeterOutput


class PiAdapTwin(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.flow = PhysioFlowNet(config.flow)
        self.meter = SpectroDepthNet(
            config.meter, config.pkpd, config.flow.grid_points, config.use_spectral
        )
        self.posterior = DiagonalPosterior.from_prior(config.prior)
        prior_mu = torch.log(torch.tensor(config.prior.as_tuple(), dtype=torch.float32))
        self.prior_mu: torch.Tensor
        self.prior_logvar: torch.Tensor
        self.register_buffer("prior_mu", prior_mu)
        self.register_buffer("prior_logvar", torch.full((len(THETA_NAMES),), -2.0))
        self.use_neural_coupling = config.use_neural_coupling
        self.coupling: nn.Linear | None = None
        if config.use_neural_coupling:
            coupling_out = len(config.flow.horizons) * config.flow.state_dim
            self.coupling = nn.Linear(config.flow.grid_points, coupling_out)

    def theta_mean(self) -> dict[str, float]:
        if self.config.use_bayesadapt:
            values = self.posterior.physical_mean().detach()
        else:
            values = torch.exp(self.prior_mu)
        return {name: float(values[i]) for i, name in enumerate(THETA_NAMES)}

    def forward(self, batch: ClinicalBatch) -> CircuitOutput:
        theta = self.theta_mean()
        flow_out = self.flow(batch.waveform)
        meter_out = self.meter(
            batch.bis_window, batch.infusion, theta["V1"], theta["k10"], theta["ke0"], dt=1.0
        )
        if self.use_neural_coupling and self.coupling is not None:
            n_h = len(self.config.flow.horizons)
            delta = self.coupling(meter_out.gamma_hat).reshape(
                batch.waveform.shape[0], n_h, self.config.flow.state_dim
            )
            flow_out.horizon_states = flow_out.horizon_states + 0.1 * delta
        return CircuitOutput(flow=flow_out, meter=meter_out)


def build_circuit(config: ExperimentConfig) -> PiAdapTwin:
    return PiAdapTwin(config)
