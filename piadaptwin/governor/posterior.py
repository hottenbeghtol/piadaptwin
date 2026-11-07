from __future__ import annotations

import torch
from torch import nn

from piadaptwin.manifold import THETA_NAMES, ThetaVector


def diagonal_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    term = logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0
    return 0.5 * torch.sum(term)


class DiagonalPosterior(nn.Module):
    def __init__(self, mean: torch.Tensor, log_var: torch.Tensor) -> None:
        super().__init__()
        self.mu = nn.Parameter(mean.clone())
        self.logvar = nn.Parameter(log_var.clone())

    @classmethod
    def from_prior(cls, prior: ThetaVector, init_logvar: float = -2.0) -> DiagonalPosterior:
        mean = torch.log(torch.tensor(prior.as_tuple(), dtype=torch.float32))
        log_var = torch.full((len(THETA_NAMES),), init_logvar)
        return cls(mean, log_var)

    def sample(self, n_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        noise = torch.randn(n_samples, self.mu.shape[0], generator=generator)
        return self.mu.unsqueeze(0) + noise * std.unsqueeze(0)

    def physical_mean(self) -> torch.Tensor:
        return torch.exp(self.mu)

    def detached_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu.detach().clone(), self.logvar.detach().clone()
