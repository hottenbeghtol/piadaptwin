from __future__ import annotations

from dataclasses import dataclass

import torch

from piadaptwin.circuit import PiAdapTwin
from piadaptwin.intake.batch import ClinicalBatch
from piadaptwin.loading import (
    FlowLoss,
    MeterLoss,
    physioflownet_objective,
    spectrodepthnet_objective,
)
from piadaptwin.manifold import ExperimentConfig


@dataclass
class StepBreakdown:
    total: float
    flow: FlowLoss
    meter: MeterLoss


class StageTrainer:
    def __init__(
        self, circuit: PiAdapTwin, config: ExperimentConfig, lr: float, freeze_posterior: bool
    ) -> None:
        self.circuit = circuit
        self.config = config
        for param in circuit.posterior.parameters():
            param.requires_grad_(not freeze_posterior)
        trainable = [p for p in circuit.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=config.stages.weight_decay)
        self.grad_clip = config.stages.grad_clip

    def evaluate_losses(self, batch: ClinicalBatch) -> tuple[torch.Tensor, FlowLoss, MeterLoss]:
        out = self.circuit(batch)
        theta = self.circuit.theta_mean()
        flow_loss = physioflownet_objective(
            out.flow,
            batch.hemo_target,
            theta["C"],
            theta["R"],
            self.config.loss,
            self.config.use_windkessel,
        )
        meter_loss = spectrodepthnet_objective(
            out.meter, batch.bis_target, self.config.loss, self.config.use_pkpd
        )
        total = flow_loss.total + meter_loss.total
        return total, flow_loss, meter_loss

    def train_step(self, batch: ClinicalBatch) -> StepBreakdown:
        self.optimizer.zero_grad()
        total, flow_loss, meter_loss = self.evaluate_losses(batch)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.circuit.parameters(), self.grad_clip)
        self.optimizer.step()
        return StepBreakdown(total=float(total.detach()), flow=flow_loss, meter=meter_loss)
