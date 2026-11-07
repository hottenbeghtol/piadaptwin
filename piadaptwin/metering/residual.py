from __future__ import annotations

import torch


def pkpd_residual(
    gamma_hat: torch.Tensor,
    gamma_pk: torch.Tensor,
    bis_hat: torch.Tensor,
    bis_pd: torch.Tensor,
    lambda_bis: float,
) -> torch.Tensor:
    effect_term = torch.mean((gamma_hat - gamma_pk).pow(2))
    bis_term = torch.mean((bis_hat - bis_pd).pow(2))
    return effect_term + lambda_bis * bis_term
