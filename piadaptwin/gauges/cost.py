from __future__ import annotations

import time
from collections.abc import Callable

from torch import nn


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def measure_latency(fn: Callable[[], object], repeats: int = 10, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / float(repeats)
