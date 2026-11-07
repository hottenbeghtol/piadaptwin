from __future__ import annotations

CHANNEL_ORDER: tuple[str, ...] = ("abp", "ppg", "ecg", "etco2")
NEURAL_CHANNEL: str = "bis"
INFUSION_CHANNEL: str = "infusion"


def channel_index(name: str) -> int:
    return CHANNEL_ORDER.index(name)
