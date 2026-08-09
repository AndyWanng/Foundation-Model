"""Shared-SAT3D GeoMC foundation model.

Visible MRI/PET observations share one encoder and one metadata-conditioned
token projector. Their tokens are interpolated onto common FEM nodes and
combined with a reliability-weighted mean. GeoMC updates the resulting node
features, and one MLP query decoder predicts target latent tokens. An
exponential-moving-average encoder supplies pretraining targets.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from ..field.node_features import (
    NodeFeatureAggregator,
    NodeFeatureField,
    ObservationTokenBatch,
)
from ..field.geometry_embedding import GeometryEmbedding
from ..geometry.interpolation import FixedSpatialInterpolator
from .geomc import GeoMCCore, GeoMCDiagnostics


def _metadata_batch(
    metadata: Tensor,
    *,
    batch: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    value = torch.as_tensor(metadata, device=device, dtype=dtype)
    if value.ndim == 1:
        value = value.unsqueeze(0).expand(batch, -1)
    if value.shape != (batch, dimension):
        raise ValueError(
            f"metadata must be [{dimension}] or [B,{dimension}], got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError("metadata contains non-finite values")
    return value


class TokenProjector(nn.Module):
    """Project encoder features and modulate them with observation metadata."""

    def __init__(self, input_dim: int, latent_dim: int, metadata_dim: int) -> None:
        super().__init__()
        if min(int(input_dim), int(latent_dim), int(metadata_dim)) < 1:
            raise ValueError("input_dim, latent_dim and metadata_dim must be positive")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.metadata_dim = int(metadata_dim)
        self.normalization = nn.LayerNorm(self.input_dim)
        self.projection = nn.Linear(self.input_dim, self.latent_dim)
        self.metadata_modulation = nn.Linear(self.metadata_dim, 2 * self.latent_dim)
        # MRI and PET initially share the exact same feature transform.
        nn.init.zeros_(self.metadata_modulation.weight)
        nn.init.zeros_(self.metadata_modulation.bias)

    def forward(self, feature_map: Tensor, metadata: Tensor) -> Tensor:
        if feature_map.ndim != 5 or feature_map.shape[1] != self.input_dim:
            raise ValueError(
                f"feature_map must be [B,{self.input_dim},Gx,Gy,Gz], "
                f"got {tuple(feature_map.shape)}"
            )
        tokens = feature_map.flatten(2).transpose(1, 2)
        base = self.projection(self.normalization(tokens))
        metadata_batch = _metadata_batch(
            metadata,
            batch=base.shape[0],
            dimension=self.metadata_dim,
            device=base.device,
            dtype=base.dtype,
        )
        scale, shift = self.metadata_modulation(metadata_batch).chunk(2, dim=-1)
        result = base * (1.0 + torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Token projection produced non-finite values")
        return result


@dataclass(frozen=True)
class ObservationBatch:
    """One visible MRI or PET observation prepared for the shared encoder."""

    volume: Tensor
    token_coverage: Tensor
    token_reliability: Tensor
    metadata: Tensor
    observation_uid: str | None = None


class QueryDecoder(nn.Module):
    """Decode latent node features into target latent tokens."""

    def __init__(
        self,
        node_to_token: FixedSpatialInterpolator,
        latent_dim: int,
        metadata_dim: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.node_to_token = node_to_token
        self.latent_dim = int(latent_dim)
        self.metadata_dim = int(metadata_dim)
        hidden = int(hidden_dim or max(self.latent_dim, 64))
        self.normalization = nn.LayerNorm(self.latent_dim)
        self.metadata_modulation = nn.Linear(self.metadata_dim, 2 * self.latent_dim)
        nn.init.zeros_(self.metadata_modulation.weight)
        nn.init.zeros_(self.metadata_modulation.bias)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.latent_dim),
        )

    def forward(
        self,
        latent_field: Tensor,
        target_metadata: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, "DecoderDiagnostics"]:
        input_tokens, support = self.node_to_token(latent_field)
        metadata = _metadata_batch(
            target_metadata,
            batch=input_tokens.shape[0],
            dimension=self.metadata_dim,
            device=input_tokens.device,
            dtype=input_tokens.dtype,
        )
        scale, shift = self.metadata_modulation(metadata).chunk(2, dim=-1)
        hidden = self.normalization(input_tokens)
        hidden = hidden * (1.0 + torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
        output_tokens = self.decoder(hidden)
        if torch.any(support <= 0):
            raise RuntimeError("Node-to-token interpolation has unsupported SAT3D tokens")
        if not torch.isfinite(output_tokens).all():
            raise FloatingPointError("Query decoder produced non-finite values")
        if not return_diagnostics:
            return output_tokens
        diagnostics = DecoderDiagnostics(
            input_token_rms=float(
                input_tokens.detach().float().square().mean().sqrt()
            ),
            output_token_rms=float(
                output_tokens.detach().float().square().mean().sqrt()
            ),
        )
        return output_tokens, diagnostics


@dataclass(frozen=True)
class DecoderDiagnostics:
    input_token_rms: float
    output_token_rms: float

    def as_dict(self) -> dict[str, float]:
        return {
            "input_token_rms": self.input_token_rms,
            "output_token_rms": self.output_token_rms,
        }

@dataclass(frozen=True)
class FoundationOutput:
    predicted_latents: Tensor
    target_latents: Tensor
    query_mask: Tensor
    node_feature_field: NodeFeatureField
    latent_field: Tensor
    core_diagnostics: GeoMCDiagnostics | None
    decoder_diagnostics: DecoderDiagnostics | None
    observation_grid: tuple[int, int, int]
    target_grid: tuple[int, int, int]

def _feature_grid(feature_map: Tensor) -> tuple[int, int, int]:
    return tuple(int(value) for value in feature_map.shape[-3:])


class GeoMCFoundationModel(nn.Module):
    """One shared encoder, aggregated node-feature field, GeoMC core and MLP readout."""

    def __init__(
        self,
        *,
        online_encoder: nn.Module,
        encoder_output_dim: int,
        latent_dim: int,
        metadata_dim: int,
        token_to_node: FixedSpatialInterpolator,
        node_to_token: FixedSpatialInterpolator,
        core: GeoMCCore,
        geometry_embedding: GeometryEmbedding,
    ) -> None:
        super().__init__()
        if core.d_model != int(latent_dim) or geometry_embedding.d_model != int(
            latent_dim
        ):
            raise ValueError("Core, geometry embedding and latent dimensions must agree")
        if core.input_dim != int(latent_dim):
            raise ValueError("Node-feature and GeoMC input dimensions must agree")
        if token_to_node.target_count != core.frame.node_count:
            raise ValueError("Token-to-node interpolation and FEM node counts differ")
        if node_to_token.source_count != core.frame.node_count:
            raise ValueError("Node-to-token interpolation must consume the FEM nodes")
        if token_to_node.source_count != node_to_token.target_count:
            raise ValueError("Token counts differ between forward and reverse interpolation")

        self.online_encoder = online_encoder
        self.online_projector = TokenProjector(
            int(encoder_output_dim), int(latent_dim), int(metadata_dim)
        )
        self.target_encoder = copy.deepcopy(online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        self.token_to_node = token_to_node
        self.node_feature_aggregator = NodeFeatureAggregator(int(latent_dim), int(latent_dim))
        self.core = core
        self.geometry_embedding = geometry_embedding
        self.query_decoder = QueryDecoder(
            node_to_token,
            int(latent_dim),
            int(metadata_dim),
        )
        self.latent_dim = int(latent_dim)
        self.metadata_dim = int(metadata_dim)
        self._freeze_target()

    def _freeze_target(self) -> None:
        self.target_encoder.eval()
        self.target_projector.eval()
        for module in (self.target_encoder, self.target_projector):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "GeoMCFoundationModel":
        super().train(mode)
        self.target_encoder.eval()
        self.target_projector.eval()
        return self

    @torch.no_grad()
    def initialize_target_from_online(self) -> None:
        self.target_encoder.load_state_dict(self.online_encoder.state_dict(), strict=True)
        self.target_projector.load_state_dict(self.online_projector.state_dict(), strict=True)
        self._freeze_target()

    @staticmethod
    @torch.no_grad()
    def _ema_module(target: nn.Module, online: nn.Module, momentum: float) -> None:
        target_parameters = dict(target.named_parameters())
        online_parameters = dict(online.named_parameters())
        if target_parameters.keys() != online_parameters.keys():
            raise RuntimeError("EMA target and online parameter contracts differ")
        for name, target_parameter in target_parameters.items():
            target_parameter.mul_(momentum).add_(
                online_parameters[name].detach(), alpha=1.0 - momentum
            )
        target_buffers = dict(target.named_buffers())
        online_buffers = dict(online.named_buffers())
        if target_buffers.keys() != online_buffers.keys():
            raise RuntimeError("EMA target and online buffer contracts differ")
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
        self._ema_module(self.target_encoder, self.online_encoder, value)
        self._ema_module(self.target_projector, self.online_projector, value)

    def encode_online(
        self, volume: Tensor, metadata: Tensor
    ) -> tuple[Tensor, tuple[int, int, int]]:
        feature_map = self.online_encoder(volume)
        if not isinstance(feature_map, Tensor):
            raise TypeError("The shared encoder must return one tensor")
        return self.online_projector(feature_map, metadata), _feature_grid(feature_map)

    @torch.no_grad()
    def encode_target(
        self, volume: Tensor, metadata: Tensor
    ) -> tuple[Tensor, tuple[int, int, int]]:
        feature_map = self.target_encoder(volume)
        if not isinstance(feature_map, Tensor):
            raise TypeError("The EMA target encoder must return one tensor")
        return self.target_projector(feature_map, metadata), _feature_grid(feature_map)

    def forward(
        self,
        observations: Sequence[ObservationBatch],
        target_volume: Tensor,
        *,
        target_metadata: Tensor,
        query_mask: Tensor,
        target_support: Tensor,
        return_diagnostics: bool = True,
    ) -> FoundationOutput:
        if not observations:
            raise ValueError("At least one observation is required")
        input_observations = tuple(observations)
        token_batches: list[ObservationTokenBatch] = []
        observation_grid: tuple[int, int, int] | None = None
        for observation in input_observations:
            tokens, grid = self.encode_online(
                observation.volume, observation.metadata
            )
            if observation_grid is None:
                observation_grid = grid
            elif grid != observation_grid:
                raise RuntimeError("Observations produced different encoder grids")
            token_batches.append(
                ObservationTokenBatch(
                    features=tokens,
                    interpolation=self.token_to_node,
                    coverage=observation.token_coverage,
                    reliability=observation.token_reliability,
                )
            )

        node_feature_field = self.node_feature_aggregator(token_batches)
        geometry_embedding = self.geometry_embedding(
            node_feature_field.features.shape[0],
            reference=node_feature_field.features,
        )
        core_result = self.core(
            node_feature_field.features,
            geometry_embedding,
            return_diagnostics=bool(return_diagnostics),
        )
        if isinstance(core_result, tuple):
            latent_field, core_diagnostics = core_result
        else:
            latent_field, core_diagnostics = core_result, None

        decoder_result = self.query_decoder(
            latent_field,
            target_metadata,
            return_diagnostics=bool(return_diagnostics),
        )
        if isinstance(decoder_result, tuple):
            predicted_latents, decoder_diagnostics = decoder_result
        else:
            predicted_latents, decoder_diagnostics = decoder_result, None

        with torch.no_grad():
            target_latents, target_grid = self.encode_target(
                target_volume, target_metadata
            )
        assert observation_grid is not None
        if (
            observation_grid != target_grid
            or predicted_latents.shape != target_latents.shape
        ):
            raise RuntimeError(
                "Predicted and target latent grids differ: "
                f"{observation_grid} versus {target_grid}"
            )
        mask = torch.as_tensor(
            query_mask, device=predicted_latents.device, dtype=torch.bool
        )
        support = torch.as_tensor(target_support, device=predicted_latents.device)
        if mask.shape != predicted_latents.shape[:2] or support.shape != mask.shape:
            raise ValueError("query_mask and target_support must be [B,T]")
        valid_query = mask & (support > 0)
        if torch.any(valid_query.sum(dim=1) < 2):
            raise ValueError("Each item needs at least two supported query tokens")
        return FoundationOutput(
            predicted_latents=predicted_latents,
            target_latents=target_latents.detach(),
            query_mask=valid_query,
            node_feature_field=node_feature_field,
            latent_field=latent_field,
            core_diagnostics=core_diagnostics,
            decoder_diagnostics=decoder_diagnostics,
            observation_grid=observation_grid,
            target_grid=target_grid,
        )

    def optimizer_groups(
        self,
        *,
        encoder_learning_rate: float,
        model_learning_rate: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        if min(float(encoder_learning_rate), float(model_learning_rate)) <= 0:
            raise ValueError("Learning rates must be positive")
        if float(weight_decay) < 0:
            raise ValueError("weight_decay must be non-negative")
        encoder = [
            parameter
            for parameter in self.online_encoder.parameters()
            if parameter.requires_grad
        ]
        other_modules = (
            self.online_projector,
            self.node_feature_aggregator,
            self.core,
            self.geometry_embedding,
            self.query_decoder,
        )
        model_parameters = [
            parameter
            for module in other_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if not model_parameters:
            raise RuntimeError("Non-encoder model parameter group is empty")
        groups: list[dict[str, Any]] = []
        if encoder:
            groups.append(
                {
                    "name": "shared_encoder",
                    "params": encoder,
                    "lr": float(encoder_learning_rate),
                    "weight_decay": float(weight_decay),
                }
            )
        groups.append(
            {
                "name": "non_encoder_model",
                "params": model_parameters,
                "lr": float(model_learning_rate),
                "weight_decay": float(weight_decay),
            }
        )
        return groups

    def model_specification(self) -> dict[str, Any]:
        online_modules = (
            self.online_encoder,
            self.online_projector,
            self.node_feature_aggregator,
            self.core,
            self.geometry_embedding,
            self.query_decoder,
        )
        return {
            "name": "shared_sat3d_geomc",
            "trainable_shared_encoder_instances": 1,
            "ema_target_encoder_instances": 1,
            "modality_specific_encoders": False,
            "latent_field_uses_observation_metadata": True,
            "latent_field_uses_target_metadata": False,
            "core_receives_metadata_directly": False,
            "decoder_uses_target_metadata": True,
            "decoder_has_encoder_skip": False,
            "latent_domain": "fem_nodes",
            "routing_mode": self.core.routing_mode,
            "frame_factorization": self.core.frame.factorization,
            "observation_aggregation": "reliability_weighted_mean",
            "geomc_blocks": len(self.core.blocks),
            "scale_count": self.core.frame.scale_count,
            "geometry_embedding_type": self.geometry_embedding.feature_type,
            "decoder_type": "mlp",
            "online_parameters": sum(
                parameter.numel()
                for module in online_modules
                for parameter in module.parameters()
            ),
            "online_trainable_parameters": sum(
                parameter.numel()
                for module in online_modules
                for parameter in module.parameters()
                if parameter.requires_grad
            ),
            "target_trainable_parameters": sum(
                parameter.numel()
                for module in (self.target_encoder, self.target_projector)
                for parameter in module.parameters()
                if parameter.requires_grad
            ),
        }

def cosine_ema_momentum(
    step: int,
    total_steps: int,
    *,
    start: float = 0.996,
    end: float = 1.0 - 1.0e-6,
) -> float:
    if total_steps < 1 or step < 0 or step > total_steps:
        raise ValueError("Require 0<=step<=total_steps and total_steps>=1")
    if not 0.0 <= start <= end < 1.0:
        raise ValueError("EMA endpoints must satisfy 0<=start<=end<1")
    progress = float(step) / float(total_steps)
    weight = 0.5 - 0.5 * torch.cos(torch.tensor(progress * torch.pi)).item()
    return float(start + weight * (end - start))


__all__ = [
    "ObservationBatch",
    "FoundationOutput",
    "GeoMCFoundationModel",
    "DecoderDiagnostics",
    "QueryDecoder",
    "TokenProjector",
    "cosine_ema_momentum",
]
