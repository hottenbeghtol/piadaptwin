import torch

from piadaptwin.governor.bayesadapt import elbo_update, prevent_collapse
from piadaptwin.governor.posterior import DiagonalPosterior, diagonal_kl
from piadaptwin.manifold import GovernorConfig, ThetaVector


def test_posterior_moves_toward_likelihood_optimum() -> None:
    torch.manual_seed(0)
    posterior = DiagonalPosterior.from_prior(ThetaVector())
    prior_mu, prior_logvar = posterior.detached_state()
    target = prior_mu + 0.4

    def log_likelihood(samples: torch.Tensor) -> torch.Tensor:
        return -10.0 * torch.mean((samples - target) ** 2)

    before = float((posterior.mu.detach() - target).abs().sum())
    cfg = GovernorConfig(n_samples=16, steps=60, lr=0.05)
    elbo_update(posterior, prior_mu, prior_logvar, log_likelihood, cfg)
    after = float((posterior.mu.detach() - target).abs().sum())
    assert after < before


def test_collapse_prevention_lifts_variance() -> None:
    posterior = DiagonalPosterior(torch.zeros(7), torch.full((7,), -20.0))
    prior_logvar = torch.full((7,), -2.0)
    prevent_collapse(posterior, prior_logvar, eps_sigma=1e-4, noise_scale=0.01)
    assert bool((torch.exp(posterior.logvar) >= 1e-4).all())


def test_kl_closed_form_matches_monte_carlo() -> None:
    torch.manual_seed(1)
    mu_q = torch.tensor([0.5, -0.2])
    logvar_q = torch.tensor([-0.5, 0.1])
    mu_p = torch.tensor([0.0, 0.0])
    logvar_p = torch.tensor([0.0, 0.0])
    closed = float(diagonal_kl(mu_q, logvar_q, mu_p, logvar_p))
    std_q = torch.exp(0.5 * logvar_q)
    samples = mu_q + torch.randn(200000, 2) * std_q
    log_q = -0.5 * (
        ((samples - mu_q) / std_q) ** 2 + logvar_q + torch.log(torch.tensor(2 * torch.pi))
    )
    std_p = torch.exp(0.5 * logvar_p)
    log_p = -0.5 * (
        ((samples - mu_p) / std_p) ** 2 + logvar_p + torch.log(torch.tensor(2 * torch.pi))
    )
    mc = float((log_q.sum(dim=1) - log_p.sum(dim=1)).mean())
    assert abs(closed - mc) < 0.05
