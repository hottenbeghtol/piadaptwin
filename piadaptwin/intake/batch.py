from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClinicalBatch:
    waveform: torch.Tensor
    bis_window: torch.Tensor
    infusion: torch.Tensor
    hemo_target: torch.Tensor
    bis_target: torch.Tensor
    theta: torch.Tensor
    hypotension_label: torch.Tensor
    map_future: torch.Tensor

    def to(self, device: torch.device) -> ClinicalBatch:
        return ClinicalBatch(
            waveform=self.waveform.to(device),
            bis_window=self.bis_window.to(device),
            infusion=self.infusion.to(device),
            hemo_target=self.hemo_target.to(device),
            bis_target=self.bis_target.to(device),
            theta=self.theta.to(device),
            hypotension_label=self.hypotension_label.to(device),
            map_future=self.map_future.to(device),
        )


