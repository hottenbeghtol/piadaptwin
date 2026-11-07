from __future__ import annotations

import torch


def slice_windows(stream: torch.Tensor, width: int, stride: int) -> torch.Tensor:
    if stream.shape[-1] < width:
        raise ValueError("stream shorter than window width")
    return stream.unfold(dimension=-1, size=width, step=stride)


def horizon_targets(future: torch.Tensor, horizons: tuple[int, ...], step: float) -> torch.Tensor:
    indices = [max(0, min(future.shape[1] - 1, int(round(h / step)) - 1)) for h in horizons]
    return future[:, indices, :]
