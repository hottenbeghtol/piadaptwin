from __future__ import annotations

import torch
from torch import nn


class PatchAttentionEncoder(nn.Module):
    def __init__(
        self, n_bands: int, patch_len: int, embed_dim: int, n_heads: int, depth: int
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(n_bands * patch_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=2 * embed_dim,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        batch, bands, frames = spectrogram.shape
        remainder = frames % self.patch_len
        if remainder != 0:
            pad = self.patch_len - remainder
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad), mode="replicate")
            frames = frames + pad
        n_patches = frames // self.patch_len
        reshaped = spectrogram.reshape(batch, bands, n_patches, self.patch_len)
        reshaped = reshaped.permute(0, 2, 1, 3).reshape(batch, n_patches, bands * self.patch_len)
        tokens = self.proj(reshaped)
        encoded = self.encoder(tokens)
        return encoded.mean(dim=1)
