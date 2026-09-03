"""Content-gated projection from SAT3D feature maps to JEPA tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ContentGatedTokenProjector(nn.Module):
    """Shared token projection with bounded acquisition conditioning.

    A pooled content summary controls how strongly condition scale/shift enters
    each latent channel.  The condition branch starts at zero, so MRI and PET
    initially share the same feature transform.
    """

    def __init__(self, input_dim: int, latent_dim: int, condition_dim: int) -> None:
        super().__init__()
        if min(int(input_dim), int(latent_dim), int(condition_dim)) < 1:
            raise ValueError("projector dimensions must be positive")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.normalization = nn.LayerNorm(self.input_dim)
        self.projection = nn.Linear(self.input_dim, self.latent_dim)
        self.condition_modulation = nn.Linear(self.condition_dim, 2 * self.latent_dim)
        self.content_gate = nn.Linear(self.input_dim + self.condition_dim, self.latent_dim)
        nn.init.zeros_(self.condition_modulation.weight)
        nn.init.zeros_(self.condition_modulation.bias)
        nn.init.zeros_(self.content_gate.weight)
        nn.init.constant_(self.content_gate.bias, -2.197224577)  # sigmoid = 0.1

    def forward(self, feature_map: Tensor, condition: Tensor) -> Tensor:
        if feature_map.ndim != 5 or feature_map.shape[1] != self.input_dim:
            raise ValueError(
                f"feature_map must be [B,{self.input_dim},Gx,Gy,Gz], got {tuple(feature_map.shape)}"
            )
        batch = feature_map.shape[0]
        if condition.shape != (batch, self.condition_dim):
            raise ValueError(f"condition must be [B,{self.condition_dim}]")
        raw_tokens = feature_map.flatten(2).transpose(1, 2)
        normalised = self.normalization(raw_tokens)
        base = self.projection(normalised)
        pooled_content = normalised.mean(dim=1)
        gate = torch.sigmoid(self.content_gate(torch.cat((pooled_content, condition), dim=-1)))
        scale, shift = self.condition_modulation(condition).chunk(2, dim=-1)
        gated_scale = gate * torch.tanh(scale)
        gated_shift = gate * shift
        result = base * (1.0 + gated_scale.unsqueeze(1)) + gated_shift.unsqueeze(1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Content-gated token projection produced non-finite values")
        return result


__all__ = ["ContentGatedTokenProjector"]
