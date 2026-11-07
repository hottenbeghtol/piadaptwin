from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a) - np.asarray(b)
    if np.allclose(diff, 0.0):
        return 1.0
    result = wilcoxon(diff)
    return float(result.pvalue)


def bonferroni_threshold(n_comparisons: int, alpha: float = 0.05) -> float:
    return alpha / float(max(1, n_comparisons))


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    data = np.asarray(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(data, size=data.shape[0], replace=True)
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi
