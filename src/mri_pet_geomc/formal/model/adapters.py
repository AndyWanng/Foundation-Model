"""Near-identity modality stems and safe conditioned SAT3D adapters."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _group_count(channels: int) -> int:
    for groups in range(min(int(channels), 8), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class NearIdentityResidualStem(nn.Module):
    """A small 3D residual stem that is exactly identity at initialisation."""

    def __init__(self, *, hidden_channels: int = 8) -> None:
        super().__init__()
        if int(hidden_channels) < 1:
            raise ValueError("hidden_channels must be positive")
        hidden = int(hidden_channels)
        self.residual = nn.Sequential(
            nn.Conv3d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, volume: Tensor) -> Tensor:
        if volume.ndim != 5 or volume.shape[1] != 1 or not torch.is_floating_point(volume):
            raise ValueError("volume must be floating point [B,1,D,H,W]")
        result = volume + self.residual(volume)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Residual input stem produced non-finite values")
        return result


class MRIPETResidualStems(nn.Module):
    """Separate near-identity input adapters with one shared downstream encoder.

    ``modality_kind`` uses 0 for MRI and 1 for PET.  It is intentionally not a
    dataset/site id and therefore cannot leak corpus identity into the model.
    """

    MRI = 0
    PET = 1

    def __init__(self, *, hidden_channels: int = 8) -> None:
        super().__init__()
        self.mri = NearIdentityResidualStem(hidden_channels=hidden_channels)
        self.pet = NearIdentityResidualStem(hidden_channels=hidden_channels)

    def forward(self, volume: Tensor, modality_kind: Tensor) -> Tensor:
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError("volume must be [B,1,D,H,W]")
        kind = torch.as_tensor(modality_kind, device=volume.device, dtype=torch.long)
        if kind.shape != (volume.shape[0],) or torch.any((kind != self.MRI) & (kind != self.PET)):
            raise ValueError("modality_kind must be [B] containing only MRI=0 or PET=1")
        result = torch.empty_like(volume)
        for value, stem in ((self.MRI, self.mri), (self.PET, self.pet)):
            indices = torch.nonzero(kind == value, as_tuple=False).flatten()
            if indices.numel():
                selected = volume.index_select(0, indices)
                result.index_copy_(0, indices, stem(selected))
        return result


class ConditionedFeatureAdapter3D(nn.Module):
    """Bottleneck adapter on the audited SAT3D feature-map boundary.

    The current SAT3D wrapper safely exposes only its final feature map.  This
    adapter therefore avoids architecture-specific forward hooks.  Its final
    projection and FiLM layer are zero-initialised, making the wrapper exactly
    identity until optimisation learns an acquisition-conditioned correction.
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        *,
        bottleneck_dim: int | None = None,
    ) -> None:
        super().__init__()
        if min(int(channels), int(condition_dim)) < 1:
            raise ValueError("channels and condition_dim must be positive")
        self.channels = int(channels)
        self.condition_dim = int(condition_dim)
        hidden = int(bottleneck_dim or max(8, self.channels // 4))
        self.normalization = nn.GroupNorm(_group_count(self.channels), self.channels)
        self.down = nn.Conv3d(self.channels, hidden, kernel_size=1)
        self.activation = nn.SiLU()
        self.up = nn.Conv3d(hidden, self.channels, kernel_size=1)
        self.condition_modulation = nn.Linear(self.condition_dim, 2 * hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.condition_modulation.weight)
        nn.init.zeros_(self.condition_modulation.bias)

    def forward(self, feature_map: Tensor, condition: Tensor) -> Tensor:
        if feature_map.ndim != 5 or feature_map.shape[1] != self.channels:
            raise ValueError(
                f"feature_map must be [B,{self.channels},Gx,Gy,Gz], got {tuple(feature_map.shape)}"
            )
        if condition.shape != (feature_map.shape[0], self.condition_dim):
            raise ValueError(f"condition must be [B,{self.condition_dim}]")
        hidden = self.down(self.normalization(feature_map))
        scale, shift = self.condition_modulation(condition).chunk(2, dim=-1)
        hidden = hidden * (1.0 + torch.tanh(scale)[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        result = feature_map + self.up(self.activation(hidden))
        if not torch.isfinite(result).all():
            raise FloatingPointError("SAT3D feature adapter produced non-finite values")
        return result


class SafeConditionedEncoderWrapper(nn.Module):
    """Apply a shared encoder followed by an explicit final-stage adapter."""

    def __init__(self, encoder: nn.Module, adapter: ConditionedFeatureAdapter3D) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, volume: Tensor, condition: Tensor) -> Tensor:
        feature_map = self.encoder(volume)
        if not isinstance(feature_map, Tensor):
            raise TypeError("The SAT3D encoder must return one Tensor feature map")
        return self.adapter(feature_map, condition)


__all__ = [
    "ConditionedFeatureAdapter3D",
    "MRIPETResidualStems",
    "NearIdentityResidualStem",
    "SafeConditionedEncoderWrapper",
]
