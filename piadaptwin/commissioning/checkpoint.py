from __future__ import annotations

import os
from typing import Any

import torch


def save_atomic(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    return payload
