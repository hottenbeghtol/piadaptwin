from __future__ import annotations

import os
import sys

import pyrallis
import torch

from piadaptwin.circuit import PiAdapTwin, build_circuit
from piadaptwin.commissioning.checkpoint import load_checkpoint, save_atomic
from piadaptwin.commissioning.protocol import run_protocol
from piadaptwin.gauges.detection import auprc, auroc
from piadaptwin.gauges.regression import horizon_mae, mae, rmse
from piadaptwin.intake.sources import build_stream
from piadaptwin.manifold import ExperimentConfig
from piadaptwin.runtime import get_logger, resolve_device

_VERBS = ("charge", "commission", "gauge", "forecast", "regulate", "export")
_USAGE = "usage: piadaptwin <charge|commission|gauge|forecast|regulate|export> [--config_path FILE]"


def _checkpoint_path(cfg: ExperimentConfig) -> str:
    return os.path.join(cfg.out_dir, "circuit.pt")


def _load_or_build(cfg: ExperimentConfig) -> PiAdapTwin:
    device = resolve_device(cfg.device)
    circuit = build_circuit(cfg).to(device)
    path = _checkpoint_path(cfg)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"trained checkpoint not found: {path}")
    payload = load_checkpoint(path)
    circuit.load_state_dict(payload["model"])
    return circuit


def _verb_charge(cfg: ExperimentConfig) -> None:
    logger = get_logger()
    batch = next(iter(build_stream(cfg, 1, base_seed=cfg.seed)))
    logger.info(
        "charged waveform=%s bis=%s infusion=%s targets=%s",
        tuple(batch.waveform.shape),
        tuple(batch.bis_window.shape),
        tuple(batch.infusion.shape),
        tuple(batch.hemo_target.shape),
    )


def _verb_commission(cfg: ExperimentConfig) -> None:
    logger = get_logger()
    circuit, history = run_protocol(cfg)
    save_atomic(
        _checkpoint_path(cfg),
        {"model": circuit.state_dict(), "seed": cfg.seed, "name": cfg.name},
    )
    logger.info(
        "commission done steps=%d first=%.4f last=%.4f deploy_elbo=%s",
        len(history.losses),
        history.losses[0] if history.losses else float("nan"),
        history.losses[-1] if history.losses else float("nan"),
        [round(v, 3) for v in history.deploy_elbo],
    )


def _verb_gauge(cfg: ExperimentConfig) -> None:
    logger = get_logger()
    circuit = _load_or_build(cfg)
    device = resolve_device(cfg.device)
    batch = next(iter(build_stream(cfg, 1, base_seed=cfg.seed + 99))).to(device)
    with torch.no_grad():
        out = circuit(batch)
    pred_map = out.flow.horizon_states[:, :, 2]
    prob = torch.clamp((65.0 - pred_map.amin(dim=1)) / 10.0 + 0.5, 0.0, 1.0)
    sbp_pred = out.flow.horizon_states[:, :, 0]
    sbp_true = batch.hemo_target[:, :, 0]
    per_horizon = [round(v, 3) for v in horizon_mae(sbp_pred, sbp_true)]
    logger.info(
        "gauge sbp_mae=%.3f bis_mae=%.3f auroc=%.3f auprc=%.3f horizons=%s",
        mae(sbp_pred, sbp_true),
        mae(out.meter.bis_horizons, batch.bis_target),
        auroc(prob, batch.hypotension_label),
        auprc(prob, batch.hypotension_label),
        per_horizon,
    )


def _verb_forecast(cfg: ExperimentConfig) -> None:
    logger = get_logger()
    circuit = _load_or_build(cfg)
    device = resolve_device(cfg.device)
    batch = next(iter(build_stream(cfg, 1, base_seed=cfg.seed + 7))).to(device)
    with torch.no_grad():
        out = circuit(batch)
    logger.info(
        "forecast sbp=%s bis=%s rmse=%.3f",
        [round(float(v), 1) for v in out.flow.horizon_states[0, :, 0]],
        [round(float(v), 1) for v in out.meter.bis_horizons[0]],
        rmse(out.flow.horizon_states[:, :, 0], batch.hemo_target[:, :, 0]),
    )


def _verb_regulate(cfg: ExperimentConfig) -> None:
    raise NotImplementedError("Treatment outcome evaluation requires independently observed data")


def _verb_export(cfg: ExperimentConfig) -> None:
    logger = get_logger()
    circuit = _load_or_build(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, "encoder.onnx")
    example_input = torch.zeros(1, cfg.flow.in_channels, cfg.data.samples_per_window)
    torch.onnx.export(
        circuit.flow.encoder, (example_input,), path, input_names=["waveform"], opset_version=17
    )
    logger.info("export wrote %s", path)


def main() -> None:
    rest = sys.argv[1:]
    if not rest or rest[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return
    verb = rest[0]
    if verb not in _VERBS:
        print(_USAGE)
        raise SystemExit(2)
    cfg = pyrallis.parse(config_class=ExperimentConfig, args=rest[1:])
    dispatch = {
        "charge": _verb_charge,
        "commission": _verb_commission,
        "gauge": _verb_gauge,
        "forecast": _verb_forecast,
        "regulate": _verb_regulate,
        "export": _verb_export,
    }
    dispatch[verb](cfg)


if __name__ == "__main__":
    main()
