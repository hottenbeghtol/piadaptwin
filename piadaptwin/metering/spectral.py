from __future__ import annotations

import torch


def band_power_spectrogram(
    signal: torch.Tensor, n_fft: int, hop: int, n_bands: int
) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=signal.device, dtype=signal.dtype)
    spectrum = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spectrum.abs().pow(2)
    groups = torch.tensor_split(power, n_bands, dim=1)
    bands = [group.mean(dim=1) for group in groups]
    stacked = torch.stack(bands, dim=1)
    return torch.log1p(stacked)
