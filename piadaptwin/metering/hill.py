from __future__ import annotations

import torch


def hill_bis(
    gamma: torch.Tensor, e0: float, emax: float, hill_n: float, ec50: float
) -> torch.Tensor:
    positive = torch.clamp(gamma, min=0.0)
    numerator = positive.pow(hill_n)
    denominator = ec50**hill_n + numerator
    return e0 - emax * numerator / denominator
