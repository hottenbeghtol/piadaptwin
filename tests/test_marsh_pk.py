import torch

from piadaptwin.manifold import PkpdConfig
from piadaptwin.metering.hill import hill_bis
from piadaptwin.metering.marsh_pk import effect_site_trajectory


def test_effect_site_rises_under_constant_infusion() -> None:
    infusion = torch.ones(2, 60) * 5.0
    gamma = effect_site_trajectory(infusion, PkpdConfig(), v1=4.27, k10=0.119, ke0=0.456, dt=1.0)
    assert float(gamma[:, -1].mean()) > float(gamma[:, 5].mean())
    assert bool((gamma >= 0.0).all())


def test_hill_monotone_decreasing_in_gamma() -> None:
    gamma = torch.linspace(0.0, 10.0, 50).reshape(1, -1)
    bis = hill_bis(gamma, e0=97.4, emax=92.4, hill_n=1.43, ec50=3.08)
    diffs = bis[:, 1:] - bis[:, :-1]
    assert bool((diffs <= 1e-6).all())
    assert float(bis[0, 0]) > float(bis[0, -1])
