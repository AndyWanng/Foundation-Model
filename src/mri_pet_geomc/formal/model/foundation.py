"""Formal packed-batch MRI/PET GeoMC-JEPA model wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ...field.geometry_embedding import GeometryEmbedding
from ...model.geomc import GeoMCCore, GeoMCDiagnostics
from .checkpointing import activation_checkpointing_status
from .condition import ConditionBatch
from .ema import EncodedObservation, OnlineEMABundle
from .fusion import AnchorCentricField, AnchorCentricFusion
from .predictor import QueryAwareJEPA3DPredictor


@dataclass(frozen=True)
class FormalObservationBatch:
    """One packed anchor/companion slot for a physical batch."""

    volume: Tensor
    token_coverage: Tensor
    condition: ConditionBatch
    modality_kind: Tensor
    observation_uid: tuple[str, ...] | None = None

    @property
    def batch_size(self) -> int:
        return int(self.volume.shape[0])

    def validate(self, *, token_count: int) -> None:
        self.condition.validate()
        if self.volume.ndim != 5 or self.volume.shape[1] != 1:
            raise ValueError("observation volume must be [B,1,D,H,W]")
        if not torch.is_floating_point(self.volume) or not torch.isfinite(self.volume).all():
            raise ValueError("observation volume must be finite floating point")
        if self.token_coverage.shape != (self.batch_size, int(token_count)):
            raise ValueError("token_coverage must be [B,T]")
        if not torch.isfinite(self.token_coverage).all():
            raise ValueError("token_coverage contains non-finite values")
        if torch.any(self.token_coverage < 0) or torch.any(self.token_coverage > 1):
            raise ValueError("token_coverage must lie in [0,1]")
        if self.condition.batch_size != self.batch_size:
            raise ValueError("condition and observation physical batch sizes differ")
        kind = torch.as_tensor(self.modality_kind)
        if kind.shape != (self.batch_size,):
            raise ValueError("modality_kind must be [B]")
        if self.observation_uid is not None and len(self.observation_uid) != self.batch_size:
            raise ValueError("observation_uid must contain one id per physical batch item")


@dataclass(frozen=True)
class PackedCompanionBatch:
    """Only materialised companions, packed as K rows for a B-anchor batch."""

    observation: FormalObservationBatch
    anchor_indices: Tensor

    def validate(self, *, anchor_batch: int, token_count: int) -> None:
        self.observation.validate(token_count=token_count)
        packed = self.observation.batch_size
        if packed < 1 or packed > int(anchor_batch):
            raise ValueError("Packed companion count K must lie in [1,B]")
        indices = torch.as_tensor(self.anchor_indices)
        if indices.dtype != torch.long or indices.shape != (packed,):
            raise ValueError("anchor_indices must be torch.long [K]")
        if torch.any(indices < 0) or torch.any(indices >= int(anchor_batch)):
            raise ValueError("Packed companion anchor_indices are outside [0,B)")
        if torch.unique(indices).numel() != packed:
            raise ValueError("Each anchor may have at most one packed companion")


@dataclass(frozen=True)
class FormalFoundationOutput:
    predicted_latents: Tensor
    target_latents: Tensor
    query_valid: Tensor
    relation_codes: Tensor
    node_field: AnchorCentricField
    latent_field: Tensor
    core_diagnostics: GeoMCDiagnostics | None
    anchor_grid: tuple[int, int, int]
    target_grid: tuple[int, int, int]


class FormalGeoMCJEPAModel(nn.Module):
    """Anchor-first formal JEPA model with optional same-subject companion context.

    Input masking and relation eligibility are data-layer responsibilities.  The
    model consumes the already-masked anchor and companion, processes exactly
    one optional companion slot, applies the unchanged GeoMC core, and predicts
    only packed physical query locations with a post-GeoMC ViT.
    """

    def __init__(
        self,
        *,
        encoder_bundle: OnlineEMABundle,
        fusion: AnchorCentricFusion,
        core: GeoMCCore,
        geometry_embedding: GeometryEmbedding,
        predictor: QueryAwareJEPA3DPredictor,
        token_xyz_mm: Tensor,
        node_xyz_mm: Tensor,
        relation_count: int = 8,
    ) -> None:
        super().__init__()
        if int(relation_count) < 1:
            raise ValueError("relation_count must be positive")
        if fusion.feature_dim != encoder_bundle.online.latent_dim:
            raise ValueError("Encoder latent and fusion input dimensions differ")
        if fusion.output_dim != core.input_dim:
            raise ValueError("Fusion output and GeoMC input dimensions differ")
        if predictor.latent_dim != core.d_model:
            raise ValueError("GeoMC and predictor latent dimensions differ")
        if predictor.condition_dim != encoder_bundle.online.condition_dim:
            raise ValueError("Encoder and predictor condition dimensions differ")
        if predictor.relation_dim != fusion.relation_dim:
            raise ValueError("Fusion and predictor relation dimensions differ")
        token_positions = torch.as_tensor(token_xyz_mm, dtype=torch.float32, device="cpu")
        node_positions = torch.as_tensor(node_xyz_mm, dtype=torch.float32, device="cpu")
        if token_positions.shape != (fusion.token_to_node.source_count, 3):
            raise ValueError("token_xyz_mm must match the SAT3D token count")
        if node_positions.shape != (fusion.token_to_node.target_count, 3):
            raise ValueError("node_xyz_mm must match the FEM node count")
        if core.frame.node_count != fusion.token_to_node.target_count:
            raise ValueError("Fusion interpolator and GeoMC FEM node counts differ")
        if geometry_embedding.node_count != core.frame.node_count:
            raise ValueError("Geometry embedding and GeoMC FEM node counts differ")
        if not torch.isfinite(token_positions).all() or not torch.isfinite(node_positions).all():
            raise ValueError("Physical token/node coordinates must be finite")
        self.encoder_bundle = encoder_bundle
        self.fusion = fusion
        self.core = core
        self.geometry_embedding = geometry_embedding
        self.predictor = predictor
        self.relation_embedding = nn.Embedding(int(relation_count), predictor.relation_dim)
        self.relation_count = int(relation_count)
        self.register_buffer("token_xyz_mm", token_positions, persistent=True)
        self.register_buffer("node_xyz_mm", node_positions, persistent=True)

    @staticmethod
    def _gather_tokens(tokens: Tensor, indices: Tensor) -> Tensor:
        return torch.gather(
            tokens,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )

    def _packed_query_contract(
        self,
        query_indices: Tensor,
        query_valid: Tensor,
        *,
        batch: int,
        token_count: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        indices = torch.as_tensor(query_indices, device=device, dtype=torch.long)
        valid = torch.as_tensor(query_valid, device=device, dtype=torch.bool)
        if indices.ndim != 2 or indices.shape[0] != batch or valid.shape != indices.shape:
            raise ValueError("query_indices and query_valid must be matching [B,Q]")
        if indices.shape[1] > self.predictor.max_queries:
            raise ValueError("Packed query count exceeds the predictor limit")
        if torch.any(valid.sum(dim=1) < 1):
            raise ValueError("Every anchor needs at least one valid query")
        invalid_legal = (~valid) & (indices == -1)
        valid_legal = valid & (indices >= 0) & (indices < token_count)
        if not torch.all(invalid_legal | valid_legal):
            raise ValueError("Valid query indices must be in range; padding must use -1")
        return indices.clamp_min(0), valid

    def forward(
        self,
        anchor: FormalObservationBatch,
        target_volume: Tensor,
        *,
        target_support: Tensor,
        query_indices: Tensor,
        query_valid: Tensor,
        relation_codes: Tensor,
        companion: PackedCompanionBatch | None = None,
        return_diagnostics: bool = True,
    ) -> FormalFoundationOutput:
        token_count = self.fusion.token_to_node.source_count
        anchor.validate(token_count=token_count)
        batch = anchor.batch_size
        if target_volume.shape != anchor.volume.shape:
            raise ValueError("Full target volume must match the masked anchor tensor shape")
        if target_support.shape != (batch, token_count):
            raise ValueError("target_support must be [B,T]")
        if companion is not None:
            companion.validate(anchor_batch=batch, token_count=token_count)
        relation = torch.as_tensor(
            relation_codes, device=anchor.volume.device, dtype=torch.long
        )
        if relation.shape != (batch,) or torch.any(relation < 0) or torch.any(
            relation >= self.relation_count
        ):
            raise ValueError("relation_codes must be valid [B] vocabulary ids")
        relation_condition = self.relation_embedding(relation)

        online_anchor = self.encoder_bundle.encode_online(
            anchor.volume, anchor.condition, anchor.modality_kind
        )
        companion_encoded: EncodedObservation | None = None
        aligned_companion_tokens: Tensor | None = None
        aligned_companion_coverage: Tensor | None = None
        aligned_companion_condition: Tensor | None = None
        companion_present: Tensor | None = None
        if companion is not None:
            companion_encoded = self.encoder_bundle.encode_online(
                companion.observation.volume,
                companion.observation.condition,
                companion.observation.modality_kind,
            )
            if companion_encoded.feature_grid != online_anchor.feature_grid:
                raise RuntimeError("Anchor and companion SAT3D grids differ")
            anchor_indices = companion.anchor_indices.to(
                device=online_anchor.tokens.device, dtype=torch.long
            )
            aligned_companion_tokens = torch.zeros_like(online_anchor.tokens).index_copy(
                0, anchor_indices, companion_encoded.tokens
            )
            aligned_companion_coverage = torch.zeros_like(anchor.token_coverage).index_copy(
                0,
                anchor_indices.to(anchor.token_coverage.device),
                companion.observation.token_coverage,
            )
            aligned_companion_condition = torch.zeros_like(
                online_anchor.condition
            ).index_copy(0, anchor_indices, companion_encoded.condition)
            companion_present = torch.zeros(
                batch, device=online_anchor.tokens.device, dtype=torch.bool
            )
            companion_present[anchor_indices] = True
        node_field = self.fusion(
            online_anchor.tokens,
            anchor.token_coverage,
            online_anchor.condition,
            relation_condition=relation_condition,
            companion_tokens=aligned_companion_tokens,
            companion_coverage=aligned_companion_coverage,
            companion_condition=aligned_companion_condition,
            companion_present=companion_present,
        )
        geometry = self.geometry_embedding(batch, reference=node_field.features)
        core_result = self.core(
            node_field.features,
            geometry,
            return_diagnostics=bool(return_diagnostics),
        )
        if isinstance(core_result, tuple):
            latent_field, core_diagnostics = core_result
        else:
            latent_field, core_diagnostics = core_result, None

        target = self.encoder_bundle.encode_target(
            target_volume, anchor.condition, anchor.modality_kind
        )
        if target.feature_grid != online_anchor.feature_grid:
            raise RuntimeError("Online anchor and EMA target SAT3D grids differ")
        indices, packed_valid = self._packed_query_contract(
            query_indices,
            query_valid,
            batch=batch,
            token_count=target.tokens.shape[1],
            device=target.tokens.device,
        )
        support = torch.as_tensor(target_support, device=target.tokens.device)
        gathered_support = torch.gather(support, 1, indices) > 0
        effective_valid = packed_valid & gathered_support
        if torch.any(effective_valid.sum(dim=1) < 1):
            raise ValueError("Every item needs at least one supported query token")
        token_positions = self.token_xyz_mm.to(target.tokens)
        query_positions = token_positions[indices]
        predicted = self.predictor(
            latent_field,
            self.node_xyz_mm.to(latent_field),
            node_field.coverage,
            query_positions,
            effective_valid,
            online_anchor.condition,
            relation_condition,
        )
        target_latents = self._gather_tokens(target.tokens, indices)
        target_latents = torch.where(
            effective_valid.unsqueeze(-1), target_latents, torch.zeros_like(target_latents)
        )
        return FormalFoundationOutput(
            predicted_latents=predicted,
            target_latents=target_latents.detach(),
            query_valid=effective_valid,
            relation_codes=relation,
            node_field=node_field,
            latent_field=latent_field,
            core_diagnostics=core_diagnostics,
            anchor_grid=online_anchor.feature_grid,
            target_grid=target.feature_grid,
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        self.encoder_bundle.update_target(momentum)

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
        encoder_parameters = [
            parameter
            for parameter in self.encoder_bundle.online.parameters()
            if parameter.requires_grad
        ]
        model_modules = (
            self.fusion,
            self.core,
            self.geometry_embedding,
            self.predictor,
            self.relation_embedding,
        )
        model_parameters = [
            parameter
            for module in model_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if not encoder_parameters or not model_parameters:
            raise RuntimeError("Formal optimizer parameter groups cannot be empty")
        return [
            {
                "name": "online_acquisition_encoder",
                "params": encoder_parameters,
                "lr": float(encoder_learning_rate),
                "weight_decay": float(weight_decay),
            },
            {
                "name": "geomc_predictor",
                "params": model_parameters,
                "lr": float(model_learning_rate),
                "weight_decay": float(weight_decay),
            },
        ]

    def model_specification(self) -> dict[str, Any]:
        return {
            "name": "formal_shared_sat3d_geomc_jepa",
            "physical_batch_gt_one": True,
            "companions_per_anchor_max": 1,
            "companion_fusion": "anchor_centric_content_gate",
            "quality_or_reliability_input": False,
            "predictor": {
                "type": "query_aware_3d_vit_post_geomc",
                "d_model": self.predictor.d_model,
                "depth": self.predictor.depth,
                "max_queries": self.predictor.max_queries,
                "raw_encoder_skip": False,
                "physical_position_encoding": "fixed_sinusoidal_world_mm",
            },
            "ema": self.encoder_bundle.ema_contract(),
            "activation_checkpointing": activation_checkpointing_status(
                self.encoder_bundle.online.encoder
            ).as_dict(),
        }


__all__ = [
    "FormalFoundationOutput",
    "FormalGeoMCJEPAModel",
    "FormalObservationBatch",
    "PackedCompanionBatch",
]
