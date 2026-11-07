from __future__ import annotations

import numpy as np
import torch


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def mae(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    return float(np.mean(np.abs(_to_numpy(pred) - _to_numpy(target))))


def rmse(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    diff = _to_numpy(pred) - _to_numpy(target)
    return float(np.sqrt(np.mean(diff**2)))


def horizon_mae(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> list[float]:
    p = _to_numpy(pred)
    t = _to_numpy(target)
    return [float(np.mean(np.abs(p[:, h] - t[:, h]))) for h in range(p.shape[1])]
