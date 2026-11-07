from __future__ import annotations

from dataclasses import dataclass, field

import torch

from piadaptwin.accumulator.windkessel import windkessel_residual
from piadaptwin.circuit import PiAdapTwin, build_circuit
from piadaptwin.commissioning.trainer import StageTrainer
from piadaptwin.governor.bayesadapt import elbo_update
from piadaptwin.intake.batch import ClinicalBatch
from piadaptwin.intake.sources import build_stream
from piadaptwin.manifold import ExperimentConfig
from piadaptwin.runtime import resolve_device, set_seed


@dataclass
class ProtocolHistory:
    losses: list[float] = field(default_factory=list)
    deploy_elbo: list[float] = field(default_factory=list)


def _run_stage(
    trainer: StageTrainer, stream: list[ClinicalBatch], epochs: int, history: ProtocolHistory
) -> None:
    for _ in range(epochs):
        for batch in stream:
            step = trainer.train_step(batch)
            history.losses.append(step.total)


def deploy_step(circuit: PiAdapTwin, batch: ClinicalBatch, config: ExperimentConfig) -> float:
    with torch.no_grad():
        out = circuit(batch)
    pressure = out.flow.pressure
    cardiac = out.flow.cardiac_output
    dt = out.flow.dt_seconds

    def log_likelihood(samples: torch.Tensor) -> torch.Tensor:
        physical = torch.exp(samples)
        terms = []
        for s in range(physical.shape[0]):
            residual = windkessel_residual(pressure, cardiac, physical[s, 0], physical[s, 1], dt)
            terms.append(-residual)
        return torch.stack(terms).mean()

    elbo = elbo_update(
        circuit.posterior, circuit.prior_mu, circuit.prior_logvar, log_likelihood, config.governor
    )
    return float(elbo)


def run_protocol(
    config: ExperimentConfig, batches_per_epoch: int = 2, deploy_windows: int = 3
) -> tuple[PiAdapTwin, ProtocolHistory]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    circuit = build_circuit(config).to(device)
    stream = [b.to(device) for b in build_stream(config, batches_per_epoch, base_seed=config.seed)]
    history = ProtocolHistory()

    stage1 = StageTrainer(circuit, config, config.stages.stage1_lr, freeze_posterior=True)
    _run_stage(stage1, stream, config.stages.stage1_epochs, history)

    stage2 = StageTrainer(circuit, config, config.stages.stage1_lr, freeze_posterior=True)
    _run_stage(stage2, stream, config.stages.stage2_epochs, history)

    stage3 = StageTrainer(circuit, config, config.stages.stage3_lr, freeze_posterior=False)
    _run_stage(stage3, stream, config.stages.stage3_epochs, history)

    if config.use_bayesadapt:
        for i in range(deploy_windows):
            history.deploy_elbo.append(deploy_step(circuit, stream[i % len(stream)], config))

    return circuit, history
