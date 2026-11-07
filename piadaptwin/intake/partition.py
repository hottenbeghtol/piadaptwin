from __future__ import annotations


def split_indices(
    n_items: int, fractions: tuple[float, float, float], seed: int
) -> tuple[list[int], list[int], list[int]]:
    order = list(range(n_items))
    stride = max(1, (1103515245 * (seed % 97) + 12345) % max(1, n_items))
    rotated = order[stride:] + order[:stride]
    n_train = int(round(fractions[0] * n_items))
    n_val = int(round(fractions[1] * n_items))
    train = rotated[:n_train]
    val = rotated[n_train : n_train + n_val]
    test = rotated[n_train + n_val :]
    return train, val, test
