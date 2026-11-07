from __future__ import annotations

from collections.abc import Callable

import torch

from piadaptwin.governor.posterior import DiagonalPosterior, diagonal_kl
from piadaptwin.manifold import GovernorConfig

LogLikFn = Callable[[torch.Tensor], torch.Tensor]


def prevent_collapse(
    posterior: DiagonalPosterior,
    prior_logvar: torch.Tensor,
    eps_sigma: float,
    noise_scale: float,
) -> None:
    with torch.no_grad():
        var = torch.exp(posterior.logvar)
        mask = var < eps_sigma
        if bool(mask.any()):
            bumped = var + noise_scale * torch.exp(prior_logvar)
            posterior.logvar[mask] = torch.log(bumped[mask])


def elbo_update(
    posterior: DiagonalPosterior,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    log_likelihood: LogLikFn,
    config: GovernorConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    optimizer = torch.optim.Adam(posterior.parameters(), lr=config.lr)
    last = torch.zeros(())
    for _ in range(config.steps):
        optimizer.zero_grad()
        samples = posterior.sample(config.n_samples, generator=generator)
        expected_ll = log_likelihood(samples)
        kl = diagonal_kl(posterior.mu, posterior.logvar, prior_mu, prior_logvar)
        elbo = expected_ll - kl
        (-elbo).backward()
        optimizer.step()
        last = elbo.detach()
    prevent_collapse(posterior, prior_logvar, config.eps_sigma, config.noise_scale)
    return last
