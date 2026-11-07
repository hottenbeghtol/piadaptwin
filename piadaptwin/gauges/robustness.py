from __future__ import annotations


def relative_increase(corrupted_mae: float, clean_mae: float) -> float:
    if clean_mae == 0.0:
        return 0.0
    return 100.0 * (corrupted_mae - clean_mae) / clean_mae
