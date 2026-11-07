from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _prep(
    scores: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    s = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    y = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    return s.reshape(-1), y.reshape(-1)


def auroc(scores: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> float:
    s, y = _prep(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(scores: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> float:
    s, y = _prep(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))
