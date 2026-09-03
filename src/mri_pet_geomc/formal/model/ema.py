"""Online/EMA acquisition encoder bundle for formal JEPA training."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .adapters import ConditionedFeatureAdapter3D, MRIPETResidualStems
from .condition import AcquisitionConditionEncoder, ConditionBatch
from .projector import ContentGatedTokenProjector


@dataclass(frozen=True)
class EncodedObservation:
    """Projected latent tokens and their acquisition condition."""

    tokens: Tensor
    condition: Tensor
    feature_grid: tuple[int, int, int]

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])


class FormalEncoderBundle(nn.Module):
    """Everything that must have an EMA copy for JEPA target encoding.

    The scope is intentionally explicit: MRI/PET input stems, acquisition
    condition encoder, shared SAT3D encoder, safe final-stage adapter and
    content-gated projector.  GeoMC, companion fusion and the ViT predictor are
    not members of this bundle and therefore cannot enter EMA accidentally.
    """

    def __init__(
        self,
        *,
        stems: MRIPETResidualStems,
        condition_encoder: AcquisitionConditionEncoder,
        encoder: nn.Module,
        feature_adapter: ConditionedFeatureAdapter3D,
        projector: ContentGatedTokenProjector,
    ) -> None:
        super().__init__()
        if feature_adapter.condition_dim != condition_encoder.output_dim:
            raise ValueError("Feature adapter and condition encoder dimensions differ")
        if projector.condition_dim != condition_encoder.output_dim:
            raise ValueError("Projector and condition encoder dimensions differ")
        if projector.input_dim != feature_adapter.channels:
            raise ValueError("SAT3D adapter channels and projector input dimensions differ")
        self.stems = stems
        self.condition_encoder = condition_encoder
        self.encoder = encoder
        self.feature_adapter = feature_adapter
        self.projector = projector

    @property
    def condition_dim(self) -> int:
        return int(self.condition_encoder.output_dim)

    @property
    def latent_dim(self) -> int:
        return int(self.projector.latent_dim)

    def forward(
        self,
        volume: Tensor,
        condition_batch: ConditionBatch,
        modality_kind: Tensor,
    ) -> EncodedObservation:
        condition_batch.validate()
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError("volume must be [B,1,D,H,W]")
        if condition_batch.batch_size != volume.shape[0]:
            raise ValueError("Volume and condition physical batch sizes differ")
        condition = self.condition_encoder(condition_batch)
        adapted_input = self.stems(volume, modality_kind)
        feature_map = self.encoder(adapted_input)
        if not isinstance(feature_map, Tensor) or feature_map.ndim != 5:
            raise TypeError("The shared SAT3D encoder must return one [B,C,Gx,Gy,Gz] Tensor")
        feature_map = self.feature_adapter(feature_map, condition)
        tokens = self.projector(feature_map, condition)
        return EncodedObservation(
            tokens=tokens,
            condition=condition,
            feature_grid=tuple(int(value) for value in feature_map.shape[-3:]),
        )


class OnlineEMABundle(nn.Module):
    """Paired online/target bundles with strict parameter and buffer EMA."""

    EMA_SCOPE: tuple[str, ...] = (
        "stems",
        "condition_encoder",
        "encoder",
        "feature_adapter",
        "projector",
    )
    EMA_EXCLUDED: tuple[str, ...] = (
        "anchor_companion_fusion",
        "geometry_embedding",
        "geomc",
        "predictor",
        "relation_embedding",
    )

    def __init__(self, online: FormalEncoderBundle) -> None:
        super().__init__()
        self.online = online
        self.target = copy.deepcopy(online)
        self._freeze_target()

    def _freeze_target(self) -> None:
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "OnlineEMABundle":
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def initialize_target_from_online(self) -> None:
        self.target.load_state_dict(self.online.state_dict(), strict=True)
        self._freeze_target()

    @staticmethod
    @torch.no_grad()
    def _ema_module(target: nn.Module, online: nn.Module, momentum: float) -> None:
        target_parameters = dict(target.named_parameters())
        online_parameters = dict(online.named_parameters())
        if target_parameters.keys() != online_parameters.keys():
            raise RuntimeError("Online and target parameter contracts differ")
        for name, target_parameter in target_parameters.items():
            source = online_parameters[name].detach()
            target_parameter.mul_(momentum).add_(source, alpha=1.0 - momentum)
        target_buffers = dict(target.named_buffers())
        online_buffers = dict(online.named_buffers())
        if target_buffers.keys() != online_buffers.keys():
            raise RuntimeError("Online and target buffer contracts differ")
        for name, target_buffer in target_buffers.items():
            source = online_buffers[name].detach()
            if target_buffer.dtype.is_floating_point:
                target_buffer.mul_(momentum).add_(source, alpha=1.0 - momentum)
            else:
                target_buffer.copy_(source)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        value = float(momentum)
        if not 0.0 <= value < 1.0:
            raise ValueError("EMA momentum must lie in [0,1)")
        self._ema_module(self.target, self.online, value)

    def encode_online(
        self,
        volume: Tensor,
        condition_batch: ConditionBatch,
        modality_kind: Tensor,
    ) -> EncodedObservation:
        return self.online(volume, condition_batch, modality_kind)

    @torch.no_grad()
    def encode_target(
        self,
        volume: Tensor,
        condition_batch: ConditionBatch,
        modality_kind: Tensor,
    ) -> EncodedObservation:
        return self.target(volume, condition_batch, modality_kind)

    def ema_contract(self) -> dict[str, object]:
        return {
            "included": list(self.EMA_SCOPE),
            "excluded": list(self.EMA_EXCLUDED),
            "target_trainable_parameters": sum(
                parameter.numel()
                for parameter in self.target.parameters()
                if parameter.requires_grad
            ),
        }


__all__ = ["EncodedObservation", "FormalEncoderBundle", "OnlineEMABundle"]
