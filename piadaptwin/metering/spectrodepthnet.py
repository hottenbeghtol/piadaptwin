from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from piadaptwin.manifold import MeterConfig, PkpdConfig
from piadaptwin.metering.hill import hill_bis
from piadaptwin.metering.marsh_pk import effect_site_trajectory
from piadaptwin.metering.patch_attention import PatchAttentionEncoder
from piadaptwin.metering.spectral import band_power_spectrogram


@dataclass
class MeterOutput:
    bis_horizons: torch.Tensor
    bis_traj: torch.Tensor
    gamma_hat: torch.Tensor
    gamma_pk: torch.Tensor
    bis_pd: torch.Tensor


def _resample(series: torch.Tensor, length: int) -> torch.Tensor:
    pooled = torch.nn.functional.adaptive_avg_pool1d(series.unsqueeze(1), length)
    return pooled.squeeze(1)


class SpectroDepthNet(nn.Module):
    def __init__(
        self,
        meter: MeterConfig,
        pkpd: PkpdConfig,
        grid_points: int,
        use_spectral: bool = True,
    ) -> None:
        super().__init__()
        self.meter = meter
        self.pkpd = pkpd
        self.grid_points = grid_points
        self.use_spectral = use_spectral
        self.n_fft = meter.stft_window * 4
        self.encoder: PatchAttentionEncoder | None = None
        self.scalar_path: nn.Sequential | None = None
        if use_spectral:
            self.encoder = PatchAttentionEncoder(
                meter.n_bands, meter.patch_seconds, meter.embed_dim, meter.n_heads, meter.depth
            )
        else:
            self.scalar_path = nn.Sequential(
                nn.Linear(4, meter.embed_dim),
                nn.SiLU(),
                nn.Linear(meter.embed_dim, meter.embed_dim),
            )
        self.bis_head = nn.Linear(meter.embed_dim, len(meter.horizons))
        self.gamma_head = nn.Linear(meter.embed_dim, grid_points)
        self.traj_head = nn.Linear(meter.embed_dim, grid_points)

    def _embed(self, bis_window: torch.Tensor) -> torch.Tensor:
        if self.use_spectral:
            assert self.encoder is not None
            spec = band_power_spectrogram(
                bis_window, self.n_fft, self.meter.stft_hop, self.meter.n_bands
            )
            return self.encoder(spec)
        assert self.scalar_path is not None
        stats = torch.stack(
            [
                bis_window.mean(dim=1),
                bis_window.std(dim=1),
                bis_window.amax(dim=1),
                bis_window.amin(dim=1),
            ],
            dim=1,
        )
        return self.scalar_path(stats)

    def forward(
        self,
        bis_window: torch.Tensor,
        infusion: torch.Tensor,
        v1: float,
        k10: float,
        ke0: float,
        dt: float,
    ) -> MeterOutput:
        embedding = self._embed(bis_window)
        bis_horizons = self.bis_head(embedding)
        bis_traj = self.traj_head(embedding)
        gamma_hat = torch.nn.functional.softplus(self.gamma_head(embedding))
        gamma_full = effect_site_trajectory(infusion, self.pkpd, v1, k10, ke0, dt)
        gamma_pk = _resample(gamma_full, self.grid_points)
        bis_pd = hill_bis(gamma_hat, self.pkpd.e0, self.pkpd.emax, self.pkpd.hill_n, self.pkpd.ec50)
        return MeterOutput(
            bis_horizons=bis_horizons,
            bis_traj=bis_traj,
            gamma_hat=gamma_hat,
            gamma_pk=gamma_pk,
            bis_pd=bis_pd,
        )
