from __future__ import annotations

import numpy as np


def group_mae(errors: np.ndarray, group_ids: np.ndarray) -> dict[int, float]:
    err = np.abs(np.asarray(errors))
    gid = np.asarray(group_ids)
    result: dict[int, float] = {}
    for group in np.unique(gid):
        result[int(group)] = float(np.mean(err[gid == group]))
    return result


def variability_split(
    true_range: np.ndarray, errors: np.ndarray, low: float = 10.0, high: float = 25.0
) -> dict[str, float]:
    rng = np.asarray(true_range)
    err = np.abs(np.asarray(errors))
    stable = rng < low
    transitional = (rng >= low) & (rng <= high)
    severe = rng > high
    out: dict[str, float] = {}
    for name, mask in (("stable", stable), ("transitional", transitional), ("high", severe)):
        out[name] = float(np.mean(err[mask])) if bool(mask.any()) else float("nan")
    return out
