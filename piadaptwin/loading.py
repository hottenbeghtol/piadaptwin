from __future__ import annotations

from dataclasses import dataclass

import torch

from piadaptwin.accumulator.windkessel import windkessel_residual
from piadaptwin.flowpath.physioflownet import FlowOutput
from piadaptwin.manifold import LossConfig
from piadaptwin.metering.residual import pkpd_residual
from piadaptwin.metering.spectrodepthnet import MeterOutput


@dataclass
class FlowLoss:
    total: torch.Tensor
    prediction: torch.Tensor
    windkessel: torch.Tensor
    smooth: torch.Tensor


@dataclass
class MeterLoss:
    total: torch.Tensor
    prediction: torch.Tensor
    pkpd: torch.Tensor


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def smoothness_loss(states: torch.Tensor) -> torch.Tensor:
    second = states[:, 2:, :] - 2.0 * states[:, 1:-1, :] + states[:, :-2, :]
    return torch.mean(second.pow(2))


def physioflownet_objective(
    flow_out: FlowOutput,
    hemo_target: torch.Tensor,
    compliance: torch.Tensor | float,
    resistance: torch.Tensor | float,
    loss_cfg: LossConfig,
    use_windkessel: bool,
) -> FlowLoss:
    pred = prediction_loss(flow_out.horizon_states, hemo_target)
    smooth = smoothness_loss(flow_out.states)
    if use_windkessel:
        residual = windkessel_residual(
            flow_out.pressure, flow_out.cardiac_output, compliance, resistance, flow_out.dt_seconds
        )
    else:
        residual = torch.zeros((), device=pred.device)
    total = pred + loss_cfg.alpha * residual + loss_cfg.beta * smooth
    return FlowLoss(total=total, prediction=pred, windkessel=residual, smooth=smooth)


def spectrodepthnet_objective(
    meter_out: MeterOutput,
    bis_target: torch.Tensor,
    loss_cfg: LossConfig,
    use_pkpd: bool,
    mu_override: float | None = None,
) -> MeterLoss:
    pred = prediction_loss(meter_out.bis_horizons, bis_target)
    if use_pkpd:
        residual = pkpd_residual(
            meter_out.gamma_hat,
            meter_out.gamma_pk,
            meter_out.bis_traj,
            meter_out.bis_pd,
            loss_cfg.lambda_bis,
        )
        weight = loss_cfg.mu if mu_override is None else mu_override
    else:
        residual = torch.zeros((), device=pred.device)
        weight = 0.0
    total = pred + weight * residual
    return MeterLoss(total=total, prediction=pred, pkpd=residual)
