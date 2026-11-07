from __future__ import annotations

import torch


def estimate_cardiac_output(ppg: torch.Tensor, fs: float, smooth_beats: int = 5) -> torch.Tensor:
    if ppg.ndim != 2:
        raise ValueError("ppg must be a (batch, length) tensor")
    diff = ppg[:, 1:] - ppg[:, :-1]
    upslope = torch.clamp(diff, min=0.0).mean(dim=1)
    amplitude = ppg.amax(dim=1) - ppg.amin(dim=1)
    proxy = amplitude * (upslope * fs)
    proxy = torch.clamp(proxy, min=1e-3)
    window = max(1, smooth_beats)
    kernel = torch.ones(window, device=ppg.device) / float(window)
    padded = torch.nn.functional.pad(proxy.reshape(1, 1, -1), (window - 1, 0), mode="replicate")
    smoothed = torch.nn.functional.conv1d(padded, kernel.reshape(1, 1, -1)).reshape(-1)
    return smoothed
