from __future__ import annotations

from collections.abc import Iterator

from piadaptwin.intake.batch import ClinicalBatch
from piadaptwin.manifold import ExperimentConfig

KNOWN_SOURCES: tuple[str, ...] = ("vitaldb", "mimic3", "physionet2019")


def build_stream(cfg: ExperimentConfig, n_batches: int, base_seed: int) -> Iterator[ClinicalBatch]:
    source = cfg.data.source
    if source not in KNOWN_SOURCES:
        raise ValueError(f"unknown data source: {source}")
    if not cfg.data.manifest:
        raise FileNotFoundError(f"source '{source}' requires a data.manifest")
    raise NotImplementedError(f"dataset loader for '{source}' must be implemented")
