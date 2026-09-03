"""Concrete PyTorch integration for the first formal 24k-observation run.

The data, model, and runtime packages expose deliberately separate contracts.
This module is the only bridge between them: it owns deterministic physical
masking, acquisition-only condition projection, real DataLoader consumption,
optimizer/EMA execution, and the configured fixed-coverage loop.

The tiny builder at the end of this file is exclusively for CPU contract tests.
It is explicitly marked as invalid for scientific results.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from torch.utils.data import DataLoader

from ..field.geometry_embedding import GeometryEmbedding
from ..geometry.fem import FEMGeometry
from ..geometry.frame import ParsevalResolventFrame
from ..geometry.interpolation import (
    build_interpolator,
    token_centers_from_reference,
)
from ..model.geomc import GeoMCCore
from ..model.trainable_sat3d import TrainableSAT3DEncoder
from ..training.jepa import entropy_effective_rank, masked_huber_objective
from .data import (
    CoverageSampler,
    FormalVolumeDataset,
    SampleKey,
    packed_collate,
    robust_normalize_visible,
)
from .model import (
    AcquisitionConditionEncoder,
    ActivationCheckpointingReport,
    AnchorCentricFusion,
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    ConditionBatch,
    ConditionedFeatureAdapter3D,
    ContentGatedTokenProjector,
    FormalEncoderBundle,
    FormalGeoMCJEPAModel,
    FormalObservationBatch,
    MRIPETResidualStems,
    MetadataVocabulary,
    OnlineEMABundle,
    PackedCompanionBatch,
    QueryAwareJEPA3DPredictor,
    configure_sat3d_activation_checkpointing,
)
from .runtime import (
    AtomicStateStore,
    CheckpointCursor,
    FormalTrainingSchedule,
    Microbatch,
    ModelContractError,
    RelationMixture,
    ResumeMismatchError,
    ResumeState,
    TrainingBackend,
    TrainingRuntimeController,
    TrainingStepResult,
    ValidationRequest,
    ensure_finite_metrics,
)


BatchPayload = dict[str, Any]


RELATION_CODES: Mapping[str, int] = {
    "same_observation": 0,
    "same_session_cross_sequence": 1,
    "same_session_repeat": 2,
    "longitudinal_same_sequence": 3,
    "longitudinal_same_tracer": 4,
}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _triplet(value: Any) -> tuple[float | None, float | None, float | None]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return (None, None, None)
    result: list[float | None] = []
    for item in value:
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            parsed = math.nan
        result.append(parsed if math.isfinite(parsed) and parsed > 0 else None)
    return result[0], result[1], result[2]


def acquisition_condition_record(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Project catalogue metadata onto the model's strict acquisition whitelist.

    The returned mapping contains only fields accepted by
    :class:`MetadataVocabulary`.  Dataset, site, subject, diagnosis, QC,
    quality, and reliability keys cannot survive this construction.
    """

    acquisition = _as_mapping(metadata.get("acquisition_metadata"))
    preprocessing = _as_mapping(metadata.get("preprocessing"))
    geometry = _as_mapping(preprocessing.get("geometry"))
    modality = str(metadata.get("modality", "")).strip().lower()
    sequence = str(metadata.get("sequence", "")).strip()
    product = str(metadata.get("product", "")).strip()
    tracer = str(metadata.get("tracer", "")).strip()
    family = str(metadata.get("tracer_family", "")).strip()
    native_spacing = _triplet(
        _first_present(
            geometry.get("native_spacing"),
            acquisition.get("native_spacing"),
            acquisition.get("NativeSpacing"),
        )
    )
    native_resolution = _triplet(
        _first_present(
            acquisition.get("native_resolution"),
            acquisition.get("NativeResolution"),
        )
    )
    dwi_representation = _first_present(
        acquisition.get("dwi_representation"),
        acquisition.get("DWIRepresentation"),
        "trace" if "trace" in product.lower() else None,
    )
    mp2rage_component = _first_present(
        acquisition.get("mp2rage_component"),
        acquisition.get("MP2RAGEComponent"),
        product if sequence.upper() == "MP2RAGE" and product.upper() in {"INV1", "UNI", "INV2"} else None,
    )
    record: dict[str, Any] = {
        "modality": modality,
        "mri_sequence": sequence if modality == "mri" else None,
        "mri_product": product if modality == "mri" else None,
        "pet_family": family if modality == "pet" else None,
        "exact_tracer": tracer if modality == "pet" else None,
        "dwi_representation": dwi_representation,
        "mp2rage_component": mp2rage_component,
        "derived_flag": "true" if bool(metadata.get("derived", False)) else "false",
        "native_spacing_x": native_spacing[0],
        "native_spacing_y": native_spacing[1],
        "native_spacing_z": native_spacing[2],
        "native_resolution_x": native_resolution[0],
        "native_resolution_y": native_resolution[1],
        "native_resolution_z": native_resolution[2],
    }
    allowed = set(CATEGORICAL_FIELDS) | set(CONTINUOUS_FIELDS)
    if set(record) != allowed:
        raise ModelContractError("Acquisition condition projection changed its whitelist")
    return record


def build_acquisition_vocabulary(
    metadata_records: Sequence[Mapping[str, Any]],
) -> MetadataVocabulary:
    if not metadata_records:
        raise ValueError("Training metadata is required to freeze the condition vocabulary")
    projected = [acquisition_condition_record(record) for record in metadata_records]
    return MetadataVocabulary.from_records(projected)


def encode_condition_batch(
    vocabulary: MetadataVocabulary,
    metadata_records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device | str,
    observation_uids: Sequence[str] | None = None,
    coverage_ids: Sequence[int] | None = None,
    view_ids: Sequence[int] | None = None,
    dropout_probability: float = 0.0,
    dropout_seed: int = 20260822,
) -> ConditionBatch:
    projected = [acquisition_condition_record(record) for record in metadata_records]
    probability = float(dropout_probability)
    if not 0.0 <= probability < 1.0:
        raise ValueError("metadata dropout probability must lie in [0,1)")
    if probability:
        if observation_uids is None or coverage_ids is None or view_ids is None:
            raise ValueError("deterministic metadata dropout requires uid/coverage/view")
        if not (
            len(projected)
            == len(observation_uids)
            == len(coverage_ids)
            == len(view_ids)
        ):
            raise ValueError("metadata dropout identity vectors differ from batch size")
        dropped: list[dict[str, Any]] = []
        for row, record in enumerate(projected):
            current = dict(record)
            for field_name in (*CATEGORICAL_FIELDS, *CONTINUOUS_FIELDS):
                if field_name == "modality":
                    continue
                digest = hashlib.sha256(
                    "|".join(
                        (
                            str(dropout_seed),
                            str(observation_uids[row]),
                            str(coverage_ids[row]),
                            str(view_ids[row]),
                            field_name,
                            "metadata-dropout",
                        )
                    ).encode()
                ).digest()
                draw = int.from_bytes(digest[:8], "big") / float(2**64)
                if draw < probability:
                    current[field_name] = None
            dropped.append(current)
        projected = dropped
    return vocabulary.encode(projected).to(device)


def _modality_kinds(metadata_records: Sequence[Mapping[str, Any]]) -> Tensor:
    values: list[int] = []
    for record in metadata_records:
        modality = str(record.get("modality", "")).strip().lower()
        if modality == "mri":
            values.append(MRIPETResidualStems.MRI)
        elif modality == "pet":
            values.append(MRIPETResidualStems.PET)
        else:
            raise ModelContractError(f"Unsupported modality in packed batch: {modality!r}")
    return torch.tensor(values, dtype=torch.long)


@dataclass(frozen=True)
class TokenMaskPlan:
    query_indices: Tensor
    query_valid: Tensor
    hidden_token_mask: Tensor
    hidden_voxel_mask: Tensor
    full_token_support: Tensor


class DeterministicPhysicalBlockMasker:
    """Deterministic multi-block masking on the SAT3D physical token grid."""

    def __init__(
        self,
        token_xyz_mm: Tensor,
        *,
        grid_shape: Sequence[int],
        volume_shape: Sequence[int],
        hidden_fraction: float = 0.30,
        max_queries: int = 128,
        blocks: int = 2,
        seed: int = 20260822,
    ) -> None:
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.volume_shape = tuple(int(value) for value in volume_shape)
        if len(self.grid_shape) != 3 or len(self.volume_shape) != 3:
            raise ValueError("grid_shape and volume_shape must contain three dimensions")
        if any(value < 1 for value in (*self.grid_shape, *self.volume_shape)):
            raise ValueError("mask grid and volume dimensions must be positive")
        if any(size % cells for size, cells in zip(self.volume_shape, self.grid_shape)):
            raise ValueError("Every volume dimension must be divisible by its token grid")
        if not 0.0 < float(hidden_fraction) < 1.0:
            raise ValueError("hidden_fraction must lie in (0,1)")
        if min(int(max_queries), int(blocks)) < 1:
            raise ValueError("max_queries and blocks must be positive")
        coordinates = torch.as_tensor(token_xyz_mm, dtype=torch.float64, device="cpu")
        token_count = math.prod(self.grid_shape)
        if coordinates.shape != (token_count, 3) or not torch.isfinite(coordinates).all():
            raise ValueError("token_xyz_mm must be finite [prod(grid_shape),3]")
        self.token_xyz_mm = coordinates
        self.hidden_fraction = float(hidden_fraction)
        self.max_queries = int(max_queries)
        self.blocks = int(blocks)
        self.seed = int(seed)

    @staticmethod
    def _seed(*parts: object) -> int:
        digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
        return int.from_bytes(digest[:8], "big", signed=False) % (2**63 - 1)

    def _select_queries(
        self,
        supported: Tensor,
        *,
        anchor_uid: str,
        coverage_id: int,
        view_id: int,
    ) -> Tensor:
        supported_indices = torch.nonzero(supported, as_tuple=False).flatten().cpu()
        if supported_indices.numel() < 1:
            raise ModelContractError(f"Anchor {anchor_uid} has no supported SAT3D token")
        target_count = min(
            self.max_queries,
            int(supported_indices.numel()),
            max(1, int(round(self.hidden_fraction * int(supported_indices.numel())))),
        )
        generator = torch.Generator(device="cpu").manual_seed(
            self._seed(self.seed, coverage_id, anchor_uid, view_id, "physical-mask")
        )
        coordinates = self.token_xyz_mm.index_select(0, supported_indices)
        first = int(torch.randint(len(supported_indices), (1,), generator=generator))
        center_rows = [first]
        center_count = min(self.blocks, target_count, len(supported_indices))
        while len(center_rows) < center_count:
            centers = coordinates[torch.tensor(center_rows)]
            distance = torch.cdist(coordinates, centers).amin(dim=1)
            distance[torch.tensor(center_rows)] = -1.0
            center_rows.append(int(torch.argmax(distance)))
        centers = coordinates[torch.tensor(center_rows)]
        extent = coordinates.amax(dim=0) - coordinates.amin(dim=0)
        base_scale = extent.clamp_min(1.0) / max(float(target_count) ** (1.0 / 3.0), 1.0)
        scale_jitter = 0.75 + 0.50 * torch.rand(
            center_count, 3, generator=generator, dtype=torch.float64
        )
        delta = (coordinates[:, None, :] - centers[None, :, :]).abs()
        block_distance = (delta / (base_scale[None, None, :] * scale_jitter[None])).amax(
            dim=-1
        )
        score = block_distance.amin(dim=1)
        score = score + torch.rand(score.shape, generator=generator, dtype=score.dtype) * 1.0e-9
        selected_rows = torch.argsort(score)[:target_count]
        return supported_indices.index_select(0, selected_rows)

    def _tokens_to_voxels(self, hidden_tokens: Tensor) -> Tensor:
        batch = hidden_tokens.shape[0]
        grid = hidden_tokens.reshape(batch, *self.grid_shape)
        result = grid
        for axis, repeat in enumerate(
            (size // cells for size, cells in zip(self.volume_shape, self.grid_shape)),
            start=1,
        ):
            result = result.repeat_interleave(repeat, dim=axis)
        return result

    def token_coverage(self, voxel_support: Tensor) -> Tensor:
        support = torch.as_tensor(voxel_support, dtype=torch.float32, device="cpu")
        if support.ndim != 4 or tuple(support.shape[1:]) != self.volume_shape:
            raise ValueError("voxel support must be [B,*volume_shape]")
        kernel = tuple(
            size // cells for size, cells in zip(self.volume_shape, self.grid_shape)
        )
        pooled = F.avg_pool3d(support.unsqueeze(1), kernel_size=kernel, stride=kernel)
        return pooled.flatten(1).clamp(0.0, 1.0)

    def plan(
        self,
        support: Tensor,
        *,
        anchor_uids: Sequence[str],
        coverage_ids: Sequence[int],
        view_ids: Sequence[int],
    ) -> TokenMaskPlan:
        support_bool = torch.as_tensor(support, dtype=torch.bool, device="cpu")
        batch = support_bool.shape[0]
        if tuple(support_bool.shape[1:]) != self.volume_shape:
            raise ValueError("support shape differs from the formal volume shape")
        if not (len(anchor_uids) == len(coverage_ids) == len(view_ids) == batch):
            raise ValueError("mask identity vectors must match the physical batch")
        full_token_support = self.token_coverage(support_bool)
        hidden = torch.zeros_like(full_token_support, dtype=torch.bool)
        queries = torch.full((batch, self.max_queries), -1, dtype=torch.long)
        valid = torch.zeros((batch, self.max_queries), dtype=torch.bool)
        for row in range(batch):
            selected = self._select_queries(
                full_token_support[row] > 0,
                anchor_uid=str(anchor_uids[row]),
                coverage_id=int(coverage_ids[row]),
                view_id=int(view_ids[row]),
            )
            hidden[row, selected] = True
            queries[row, : selected.numel()] = selected
            valid[row, : selected.numel()] = True
        return TokenMaskPlan(
            query_indices=queries,
            query_valid=valid,
            hidden_token_mask=hidden,
            hidden_voxel_mask=self._tokens_to_voxels(hidden),
            full_token_support=full_token_support,
        )


@dataclass(frozen=True)
class PreparedFormalBatch:
    anchor: FormalObservationBatch
    target_volume: Tensor
    target_support: Tensor
    query_indices: Tensor
    query_valid: Tensor
    relation_codes: Tensor
    relation_types: tuple[str, ...]
    modalities: tuple[str, ...]
    subject_ids: tuple[str, ...]
    companion: PackedCompanionBatch | None
    normalization_median: tuple[float, ...]
    normalization_scale: tuple[float, ...]
    companion_requested_count: int = 0
    companion_fallbacks: tuple[Mapping[str, Any], ...] = ()

    @property
    def anchor_count(self) -> int:
        return self.anchor.batch_size


class FormalBatchPreparer:
    """Turn packed cache tensors into coupled, visible-normalized model inputs."""

    def __init__(
        self,
        *,
        vocabulary: MetadataVocabulary,
        masker: DeterministicPhysicalBlockMasker,
        device: torch.device | str,
        non_blocking: bool = True,
        metadata_dropout_probability: float = 0.30,
        metadata_dropout_seed: int = 20260822,
        companion_fallback_callback: (
            Callable[[Sequence[Mapping[str, Any]]], None] | None
        ) = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.masker = masker
        self.device = torch.device(device)
        self.non_blocking = bool(non_blocking)
        self.metadata_dropout_probability = float(metadata_dropout_probability)
        self.metadata_dropout_seed = int(metadata_dropout_seed)
        self.companion_fallback_callback = companion_fallback_callback
        if not 0.0 <= self.metadata_dropout_probability < 1.0:
            raise ValueError("metadata_dropout_probability must lie in [0,1)")

    @staticmethod
    def _normalize_rows(
        image: Tensor,
        support: Tensor,
        hidden_voxel_mask: Tensor,
    ) -> tuple[Tensor, Tensor, tuple[float, ...], tuple[float, ...]]:
        full_rows: list[Tensor] = []
        online_rows: list[Tensor] = []
        medians: list[float] = []
        scales: list[float] = []
        for row in range(image.shape[0]):
            visible = ~hidden_voxel_mask[row]
            normalized, statistics = robust_normalize_visible(
                image[row].float(), support[row].bool(), visible
            )
            online = normalized.masked_fill(hidden_voxel_mask[row].unsqueeze(0), 0.0)
            full_rows.append(normalized)
            online_rows.append(online)
            medians.append(float(statistics["median"]))
            scales.append(float(statistics["robust_scale"]))
        return (
            torch.stack(online_rows),
            torch.stack(full_rows),
            tuple(medians),
            tuple(scales),
        )

    def prepare(
        self,
        batch: Mapping[str, Any],
        *,
        mask_view_override: int | None = None,
        training: bool = True,
    ) -> PreparedFormalBatch:
        anchor_image = torch.as_tensor(batch["anchor_image"], device="cpu")
        anchor_support = torch.as_tensor(batch["anchor_support"], device="cpu").bool()
        anchor_uids = tuple(str(value) for value in batch["anchor_uid"])
        anchor_metadata = tuple(_as_mapping(value) for value in batch["anchor_metadata"])
        coverage_ids = [int(value) for value in torch.as_tensor(batch["coverage_id"]).tolist()]
        source_views = [int(value) for value in torch.as_tensor(batch["view_id"]).tolist()]
        view_ids = (
            [int(mask_view_override)] * len(anchor_uids)
            if mask_view_override is not None
            else source_views
        )
        if anchor_image.ndim != 5 or anchor_image.shape[1] != 1:
            raise ModelContractError("packed anchor_image must be [B,1,D,H,W]")
        plan = self.masker.plan(
            anchor_support,
            anchor_uids=anchor_uids,
            coverage_ids=coverage_ids,
            view_ids=view_ids,
        )
        online, target, medians, scales = self._normalize_rows(
            anchor_image, anchor_support, plan.hidden_voxel_mask
        )
        visible_support = anchor_support & ~plan.hidden_voxel_mask
        anchor_coverage = self.masker.token_coverage(visible_support)
        if torch.any(anchor_coverage[plan.hidden_token_mask] != 0):
            raise ModelContractError("Hidden anchor token retained online coverage")
        anchor_condition = encode_condition_batch(
            self.vocabulary,
            anchor_metadata,
            device=self.device,
            observation_uids=anchor_uids,
            coverage_ids=coverage_ids,
            view_ids=view_ids,
            dropout_probability=(self.metadata_dropout_probability if training else 0.0),
            dropout_seed=self.metadata_dropout_seed,
        )
        anchor_observation = FormalObservationBatch(
            volume=online.to(self.device, non_blocking=self.non_blocking),
            token_coverage=anchor_coverage.to(
                self.device, non_blocking=self.non_blocking
            ),
            condition=anchor_condition,
            modality_kind=_modality_kinds(anchor_metadata).to(
                self.device, non_blocking=self.non_blocking
            ),
            observation_uid=anchor_uids,
        )

        relation_types_list = [str(value) for value in batch["relation_type"]]
        if len(relation_types_list) != len(anchor_uids):
            raise ModelContractError("relation_type must contain one value per anchor")
        companion_batch: PackedCompanionBatch | None = None
        companion_fallbacks: list[Mapping[str, Any]] = []
        companion_indices = torch.as_tensor(
            batch["companion_batch_index"], dtype=torch.long, device="cpu"
        )
        companion_requested_count = int(companion_indices.numel())
        if companion_indices.numel():
            companion_image = torch.as_tensor(batch["companion_image"], device="cpu")
            companion_support = torch.as_tensor(
                batch["companion_support"], device="cpu"
            ).bool()
            companion_uids = tuple(str(value) for value in batch["companion_uid"])
            companion_metadata = tuple(
                _as_mapping(value) for value in batch["companion_metadata"]
            )
            if not (
                companion_image.shape[0]
                == companion_support.shape[0]
                == companion_indices.numel()
                == len(companion_uids)
                == len(companion_metadata)
            ):
                raise ModelContractError("Packed companion rows are inconsistent")
            if torch.any(companion_indices < 0) or torch.any(
                companion_indices >= len(anchor_uids)
            ):
                raise ModelContractError("Companion anchor index is outside the batch")
            coupled_hidden = plan.hidden_voxel_mask.index_select(0, companion_indices)
            companion_visible = companion_support & ~coupled_hidden
            support_counts = companion_support.flatten(1).sum(dim=1)
            if torch.any(support_counts == 0):
                row = int(torch.nonzero(support_counts == 0, as_tuple=False)[0])
                raise ModelContractError(
                    f"Cached companion {companion_uids[row]} has empty structural support"
                )
            visible_counts = companion_visible.flatten(1).sum(dim=1)
            keep_rows = visible_counts > 0
            relation_uids = tuple(str(value) for value in batch["relation_uid"])
            for row in torch.nonzero(~keep_rows, as_tuple=False).flatten().tolist():
                anchor_index = int(companion_indices[row])
                original_relation = relation_types_list[anchor_index]
                companion_fallbacks.append(
                    {
                        "anchor_uid": anchor_uids[anchor_index],
                        "companion_uid": companion_uids[row],
                        "relation_type_requested": original_relation,
                        "relation_uid": relation_uids[anchor_index],
                        "coverage_id": coverage_ids[anchor_index],
                        "view_id": view_ids[anchor_index],
                        "companion_support_voxels": int(support_counts[row]),
                        "companion_visible_voxels": 0,
                        "fallback_relation_type": "same_observation",
                        "fallback": "drop_optional_companion_keep_anchor_self_jepa",
                        "training": bool(training),
                    }
                )
                # A companion is optional context.  If the anchor-coupled
                # physical mask removes all of its nonempty registered field
                # of view, retain the anchor JEPA objective and make both the
                # actual relation condition and reported relation self-only.
                relation_types_list[anchor_index] = "same_observation"
            if companion_fallbacks and self.companion_fallback_callback is not None:
                self.companion_fallback_callback(tuple(companion_fallbacks))

            selected_rows = torch.nonzero(keep_rows, as_tuple=False).flatten()
            if selected_rows.numel():
                companion_indices = companion_indices.index_select(0, selected_rows)
                companion_image = companion_image.index_select(0, selected_rows)
                companion_support = companion_support.index_select(0, selected_rows)
                coupled_hidden = coupled_hidden.index_select(0, selected_rows)
                companion_visible = companion_visible.index_select(0, selected_rows)
                selected = selected_rows.tolist()
                companion_uids = tuple(companion_uids[row] for row in selected)
                companion_metadata = tuple(companion_metadata[row] for row in selected)
                companion_online, _, _, _ = self._normalize_rows(
                    companion_image, companion_support, coupled_hidden
                )
                companion_coverage = self.masker.token_coverage(companion_visible)
                coupled_hidden_tokens = plan.hidden_token_mask.index_select(
                    0, companion_indices
                )
                if torch.any(companion_coverage[coupled_hidden_tokens] != 0):
                    raise ModelContractError(
                        "Companion mask is not coupled in physical space"
                    )
                companion_observation = FormalObservationBatch(
                    volume=companion_online.to(
                        self.device, non_blocking=self.non_blocking
                    ),
                    token_coverage=companion_coverage.to(
                        self.device, non_blocking=self.non_blocking
                    ),
                    condition=encode_condition_batch(
                        self.vocabulary,
                        companion_metadata,
                        device=self.device,
                        observation_uids=companion_uids,
                        coverage_ids=[
                            coverage_ids[index] for index in companion_indices.tolist()
                        ],
                        view_ids=[view_ids[index] for index in companion_indices.tolist()],
                        dropout_probability=(
                            self.metadata_dropout_probability if training else 0.0
                        ),
                        dropout_seed=self.metadata_dropout_seed,
                    ),
                    modality_kind=_modality_kinds(companion_metadata).to(
                        self.device, non_blocking=self.non_blocking
                    ),
                    observation_uid=companion_uids,
                )
                companion_batch = PackedCompanionBatch(
                    observation=companion_observation,
                    anchor_indices=companion_indices.to(
                        self.device, non_blocking=self.non_blocking
                    ),
                )

        relation_types = tuple(relation_types_list)
        try:
            relation_codes = torch.tensor(
                [RELATION_CODES[value] for value in relation_types], dtype=torch.long
            )
        except KeyError as error:
            raise ModelContractError(f"Unknown formal relation type: {error.args[0]}") from error
        modalities = tuple(
            str(record.get("modality", "")).strip().lower() for record in anchor_metadata
        )
        subject_ids = tuple(
            str(record.get("canonical_subject_id", "")).strip()
            for record in anchor_metadata
        )
        if any(not value for value in subject_ids):
            raise ModelContractError("Every formal anchor requires a canonical subject ID")
        return PreparedFormalBatch(
            anchor=anchor_observation,
            target_volume=target.to(self.device, non_blocking=self.non_blocking),
            target_support=plan.full_token_support.to(
                self.device, non_blocking=self.non_blocking
            ),
            query_indices=plan.query_indices.to(
                self.device, non_blocking=self.non_blocking
            ),
            query_valid=plan.query_valid.to(
                self.device, non_blocking=self.non_blocking
            ),
            relation_codes=relation_codes.to(
                self.device, non_blocking=self.non_blocking
            ),
            relation_types=relation_types,
            modalities=modalities,
            subject_ids=subject_ids,
            companion=companion_batch,
            normalization_median=medians,
            normalization_scale=scales,
            companion_requested_count=companion_requested_count,
            companion_fallbacks=tuple(companion_fallbacks),
        )


def _seed_loader_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


def _slice_packed_batch(batch: Mapping[str, Any], anchors: int) -> BatchPayload:
    total = len(batch["anchor_uid"])
    count = int(anchors)
    if not 0 < count <= total:
        raise ValueError("packed batch slice is outside its anchor batch")
    if count == total:
        return dict(batch)
    result = dict(batch)
    for key in ("anchor_image", "anchor_support", "coverage_id", "view_id"):
        result[key] = torch.as_tensor(batch[key])[:count]
    for key in (
        "anchor_uid",
        "anchor_metadata",
        "relation_type",
        "relation_uid",
        "temporal_delta_days",
    ):
        result[key] = list(batch[key])[:count]
    companion_indices = torch.as_tensor(batch["companion_batch_index"], dtype=torch.long)
    keep = companion_indices < count
    selected_rows = torch.nonzero(keep, as_tuple=False).flatten()
    result["companion_batch_index"] = companion_indices.index_select(0, selected_rows)
    for key in ("companion_image", "companion_support"):
        result[key] = torch.as_tensor(batch[key]).index_select(0, selected_rows)
    selected_list_rows = selected_rows.tolist()
    for key in ("companion_uid", "companion_metadata"):
        result[key] = [batch[key][index] for index in selected_list_rows]
    return result


class FormalCoverageDataSource:
    """Real DataLoader source with an exact consumed-anchor resume cursor."""

    def __init__(
        self,
        dataset: FormalVolumeDataset,
        *,
        seed: int,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ) -> None:
        if int(num_workers) < 0 or int(prefetch_factor) < 1:
            raise ValueError("DataLoader worker/prefetch settings are invalid")
        if not hasattr(dataset, "observation_uids"):
            raise TypeError("FormalCoverageDataSource requires FormalVolumeDataset semantics")
        self.dataset = dataset
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.pin_memory = bool(pin_memory)
        self._uids = tuple(str(value) for value in dataset.observation_uids)
        if len(self._uids) != len(dataset) or len(set(self._uids)) != len(self._uids):
            raise ModelContractError("Formal dataset UIDs are not a unique stable coverage")
        observations = getattr(dataset, "observations", None)
        self._modalities: tuple[str, ...] | None = None
        if observations is not None and len(observations) == len(self._uids):
            modalities = tuple(
                str(getattr(row, "modality", "")).strip().lower()
                for row in observations
            )
            if all(value in {"mri", "pet"} for value in modalities):
                self._modalities = modalities

    def total_anchors(self, coverage: int) -> int:
        if int(coverage) < 1:
            raise ValueError("coverage is 1-based")
        return len(self._uids)

    @staticmethod
    def _runtime_weights(mixture: RelationMixture) -> dict[str, float]:
        return {
            "same_observation": float(mixture.self_only),
            "same_session_cross_sequence": float(
                mixture.same_session_cross_sequence
            ),
            "longitudinal": float(mixture.longitudinal_same_acquisition),
            "same_session_repeat": float(mixture.same_session_repeat),
        }

    def assert_relation_mixture(
        self, coverage: int, mixture: RelationMixture
    ) -> Mapping[str, float]:
        expected = self._runtime_weights(mixture)
        selector = getattr(self.dataset, "selector", None)
        schedule = getattr(selector, "schedule", None)
        if schedule is None or not callable(getattr(schedule, "weights", None)):
            if isinstance(self.dataset, FormalVolumeDataset):
                raise ModelContractError("Formal dataset exposes no relation schedule")
            return expected
        actual = {
            str(key): float(value)
            for key, value in schedule.weights(int(coverage)).items()
        }
        if set(actual) != set(expected) or any(
            not math.isclose(actual[key], expected[key], rel_tol=0.0, abs_tol=1.0e-12)
            for key in expected
        ):
            raise ModelContractError(
                f"Runtime/data relation schedule mismatch at coverage {coverage}: "
                f"runtime={expected} data={actual}"
            )
        return actual

    def iter_microbatches(
        self,
        *,
        coverage: int,
        start_anchor_index: int,
        micro_batch_anchors: int,
        relation_mixture: RelationMixture,
    ) -> Iterable[Microbatch[BatchPayload]]:
        return self.iter_microbatches_for_view(
            coverage=coverage,
            start_anchor_index=start_anchor_index,
            micro_batch_anchors=micro_batch_anchors,
            relation_mixture=relation_mixture,
            view_id=0,
            max_anchors=None,
        )

    def iter_microbatches_for_view(
        self,
        *,
        coverage: int,
        start_anchor_index: int,
        micro_batch_anchors: int,
        relation_mixture: RelationMixture,
        view_id: int,
        max_anchors: int | None = None,
        anchor_indices: Sequence[int] | None = None,
    ) -> Iterable[Microbatch[BatchPayload]]:
        self.assert_relation_mixture(coverage, relation_mixture)
        total = self.total_anchors(coverage)
        if not 0 <= int(start_anchor_index) <= total:
            raise ValueError("Consumed-anchor cursor is outside this coverage")
        if int(micro_batch_anchors) < 1:
            raise ValueError("micro_batch_anchors must be positive")
        if anchor_indices is None:
            remaining_limit = total - int(start_anchor_index)
            if max_anchors is not None:
                if int(max_anchors) < 1:
                    raise ValueError("max_anchors must be positive")
                remaining_limit = min(remaining_limit, int(max_anchors))
            sampler: Any = CoverageSampler(
                self._uids,
                coverage_id=int(coverage),
                seed=self.seed,
                start_offset=int(start_anchor_index),
                view_id=int(view_id),
            )
        else:
            if int(start_anchor_index) != 0 or max_anchors is not None:
                raise ValueError(
                    "Explicit evaluation indices cannot be combined with cursor/max_anchors"
                )
            selected = tuple(int(value) for value in anchor_indices)
            if not selected or len(set(selected)) != len(selected):
                raise ValueError("Evaluation anchor indices must be non-empty and unique")
            if any(value < 0 or value >= total for value in selected):
                raise ValueError("Evaluation anchor index is outside the dataset")
            sampler = tuple(
                SampleKey(value, int(coverage), int(view_id)) for value in selected
            )
            remaining_limit = len(selected)
        generator = torch.Generator(device="cpu").manual_seed(
            DeterministicPhysicalBlockMasker._seed(
                self.seed, coverage, view_id, "dataloader"
            )
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "batch_size": int(micro_batch_anchors),
            "sampler": sampler,
            "num_workers": self.num_workers,
            "collate_fn": packed_collate,
            "drop_last": False,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "worker_init_fn": _seed_loader_worker,
            "generator": generator,
        }
        if self.num_workers:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        loader = DataLoader(**loader_kwargs)

        def generate() -> Iterable[Microbatch[BatchPayload]]:
            consumed = int(start_anchor_index)
            emitted = 0
            iterator = iter(loader)
            while emitted < remaining_limit:
                wait_started = time.perf_counter()
                try:
                    packed = next(iterator)
                except StopIteration:
                    break
                loader_wait = time.perf_counter() - wait_started
                available = len(packed["anchor_uid"])
                count = min(available, remaining_limit - emitted)
                if count < available:
                    packed = _slice_packed_batch(packed, count)
                consumed += count
                emitted += count
                yield Microbatch(
                    payload=dict(packed),
                    anchor_count=count,
                    next_anchor_index=consumed,
                    loader_wait_seconds=loader_wait,
                )
            if emitted != remaining_limit:
                raise ResumeMismatchError(
                    f"DataLoader ended after {emitted} anchors; expected {remaining_limit}"
                )

        return generate()

    def evaluation_anchor_indices(
        self,
        *,
        coverage: int,
        max_anchors: int | None,
    ) -> tuple[int, ...]:
        """Choose a deterministic MRI/PET-balanced observational subset."""

        total = self.total_anchors(coverage)
        limit = total if max_anchors is None else min(total, int(max_anchors))
        if limit < 1:
            raise ValueError("Evaluation sample limit must be positive")

        def order_key(index: int, salt: str) -> bytes:
            return hashlib.sha256(
                f"{self.seed}|{coverage}|{salt}|{self._uids[index]}".encode("utf-8")
            ).digest()

        global_order = sorted(range(total), key=lambda index: order_key(index, "all"))
        if limit == total or self._modalities is None:
            return tuple(global_order[:limit])
        groups = {
            modality: sorted(
                (
                    index
                    for index, value in enumerate(self._modalities)
                    if value == modality
                ),
                key=lambda index: order_key(index, modality),
            )
            for modality in ("mri", "pet")
        }
        if limit < 2 or any(not values for values in groups.values()):
            return tuple(global_order[:limit])
        desired = {"mri": limit // 2, "pet": limit - (limit // 2)}
        counts = {
            modality: min(desired[modality], len(groups[modality]))
            for modality in groups
        }
        remaining = limit - sum(counts.values())
        for modality in ("mri", "pet"):
            if remaining <= 0:
                break
            available = len(groups[modality]) - counts[modality]
            extra = min(remaining, available)
            counts[modality] += extra
            remaining -= extra
        selected = [
            index
            for modality in ("mri", "pet")
            for index in groups[modality][: counts[modality]]
        ]
        if len(selected) != limit:
            raise ModelContractError("Unable to build the fixed evaluation subset")
        return tuple(sorted(selected, key=lambda index: order_key(index, "selected")))


class EmbeddedEMAStateProxy:
    """Checkpoint the EMA update cursor without duplicating target tensors.

    ``FormalGeoMCJEPAModel.state_dict()`` already includes the full EMA target
    bundle.  Runtime's mandatory ``ema`` checkpoint slot therefore stores only
    this explicit ownership marker and update counter.
    """

    def __init__(self) -> None:
        self.updates = 0

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "storage": "embedded_in_model.encoder_bundle.target",
            "updates": int(self.updates),
        }

    def load_state_dict(
        self, state_dict: Mapping[str, Any], *, strict: bool = True
    ) -> None:
        del strict
        if int(state_dict.get("schema_version", -1)) != 1:
            raise ModelContractError("Unsupported embedded EMA proxy state")
        if state_dict.get("storage") != "embedded_in_model.encoder_bundle.target":
            raise ModelContractError("EMA checkpoint ownership marker changed")
        updates = int(state_dict.get("updates", -1))
        if updates < 0:
            raise ModelContractError("EMA update cursor is invalid")
        self.updates = updates


@dataclass(frozen=True)
class CosineEMAMomentum:
    initial: float = 0.996
    final: float = 0.9999
    total_updates: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.initial <= self.final < 1.0:
            raise ValueError("EMA momenta must satisfy 0 <= initial <= final < 1")
        if self.total_updates < 1:
            raise ValueError("total_updates must be positive")

    def __call__(self, completed_updates: int) -> float:
        progress = min(max(int(completed_updates), 0), self.total_updates) / float(
            self.total_updates
        )
        blend = 0.5 * (1.0 - math.cos(math.pi * progress))
        return float(self.initial + (self.final - self.initial) * blend)


def _per_example_huber(prediction: Tensor, target: Tensor, valid: Tensor, beta: float) -> Tensor:
    element = F.smooth_l1_loss(
        prediction.float(), target.detach().float(), beta=float(beta), reduction="none"
    ).mean(dim=-1)
    weight = valid.to(element.dtype)
    return (element * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _mean_by_key(values: Mapping[str, list[float]]) -> tuple[dict[str, float], dict[str, int]]:
    means = {
        key: float(sum(current) / len(current))
        for key, current in values.items()
        if current
    }
    counts = {key: len(current) for key, current in values.items() if current}
    return means, counts


class TorchFormalTrainingBackend(TrainingBackend[BatchPayload]):
    """Real BF16 PyTorch optimizer backend for the formal GeoMC-JEPA model."""

    def __init__(
        self,
        *,
        model: FormalGeoMCJEPAModel,
        optimizer: Optimizer,
        preparer: FormalBatchPreparer,
        ema_momentum: Callable[[int], float],
        scheduler: LRScheduler | None = None,
        validation_source: FormalCoverageDataSource | None = None,
        validation_callback: (
            Callable[[ValidationRequest, "TorchFormalTrainingBackend"], Mapping[str, Any]]
            | None
        ) = None,
        evaluation_progress: (
            Callable[[str, int, int, str], None] | None
        ) = None,
        validation_modality_weights: Mapping[str, float] | None = None,
        precision: str = "bf16",
        huber_beta: float = 1.0,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        if precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 or fp32")
        if min(float(huber_beta), float(gradient_clip_norm)) <= 0:
            raise ValueError("Huber beta and gradient clipping norm must be positive")
        self.model = model.to(preparer.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = None
        self.ema = EmbeddedEMAStateProxy()
        self.preparer = preparer
        self.ema_momentum = ema_momentum
        self.validation_source = validation_source
        self.validation_callback = validation_callback
        self.evaluation_progress = evaluation_progress
        raw_validation_weights = dict(
            validation_modality_weights or {"mri": 0.5, "pet": 0.5}
        )
        if set(raw_validation_weights) != {"mri", "pet"} or any(
            float(value) < 0.0 for value in raw_validation_weights.values()
        ):
            raise ValueError("Validation modality weights must be non-negative MRI/PET")
        weight_total = sum(float(value) for value in raw_validation_weights.values())
        if weight_total <= 0.0:
            raise ValueError("Validation modality weights must have positive mass")
        self.validation_modality_weights = {
            key: float(value) / weight_total
            for key, value in raw_validation_weights.items()
        }
        self.precision = precision
        self.huber_beta = float(huber_beta)
        self.gradient_clip_norm = float(gradient_clip_norm)

    @property
    def device(self) -> torch.device:
        return self.preparer.device

    def extra_checkpoint_stateful(self) -> Mapping[str, Any]:
        return {}

    def _autocast(self) -> Any:
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=True,
        )

    def train_accumulated_step(
        self,
        microbatches: Sequence[Microbatch[BatchPayload]],
        *,
        coverage: int,
        relation_mixture: RelationMixture,
        expected_global_batch_anchors: int,
    ) -> TrainingStepResult:
        del relation_mixture
        if not 1 <= len(microbatches) <= 2:
            raise ModelContractError("Backend accepts one or two microbatches per step")
        total_anchors = sum(item.anchor_count for item in microbatches)
        if not 1 <= total_anchors <= int(expected_global_batch_anchors):
            raise ModelContractError("Accumulated physical anchor count is invalid")
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        used_learning_rates = {
            str(group.get("name", f"group_{index}")): float(group["lr"])
            for index, group in enumerate(self.optimizer.param_groups)
        }
        modality_values: dict[str, list[float]] = defaultdict(list)
        relation_values: dict[str, list[float]] = defaultdict(list)
        self_loss_value = 0.0
        prediction_std = 0.0
        target_std = 0.0
        prediction_mean = 0.0
        target_mean = 0.0
        prediction_effective_rank = 0.0
        target_effective_rank = 0.0
        query_count = 0.0
        encoded_volumes = 0
        companion_requested_count = 0
        companion_fallback_count = 0
        normalization_scale = 0.0
        normalization_median = 0.0
        node_diagnostics: dict[str, float] = defaultdict(float)
        compute_started = time.perf_counter()
        for microbatch in microbatches:
            prepared = self.preparer.prepare(microbatch.payload, training=True)
            if prepared.anchor_count != microbatch.anchor_count:
                raise ModelContractError("Prepared physical batch count changed")
            encoded_volumes += prepared.anchor_count
            companion_requested_count += prepared.companion_requested_count
            companion_fallback_count += len(prepared.companion_fallbacks)
            if prepared.companion is not None:
                encoded_volumes += prepared.companion.observation.batch_size
            batch_weight = prepared.anchor_count / float(total_anchors)
            with self._autocast():
                output = self.model(
                    prepared.anchor,
                    prepared.target_volume,
                    target_support=prepared.target_support,
                    query_indices=prepared.query_indices,
                    query_valid=prepared.query_valid,
                    relation_codes=prepared.relation_codes,
                    companion=prepared.companion,
                    return_diagnostics=True,
                )
                objective = masked_huber_objective(
                    output.predicted_latents,
                    output.target_latents,
                    output.query_valid,
                    beta=self.huber_beta,
                )
                contribution = batch_weight * objective.loss
            if not torch.isfinite(contribution):
                raise FloatingPointError("Formal masked-Huber contribution is non-finite")
            contribution.backward()

            per_example = _per_example_huber(
                output.predicted_latents,
                output.target_latents,
                output.query_valid,
                self.huber_beta,
            ).detach()
            for row, value in enumerate(per_example.tolist()):
                modality_values[prepared.modalities[row]].append(float(value))
                relation_values[prepared.relation_types[row]].append(float(value))
            selected_prediction = output.predicted_latents[output.query_valid].detach().float()
            selected_target = output.target_latents[output.query_valid].detach().float()
            self_loss_value += batch_weight * float(objective.loss.detach())
            prediction_std += batch_weight * float(
                selected_prediction.std(unbiased=False)
            )
            target_std += batch_weight * float(selected_target.std(unbiased=False))
            prediction_mean += batch_weight * float(selected_prediction.mean())
            target_mean += batch_weight * float(selected_target.mean())
            prediction_effective_rank += batch_weight * float(
                entropy_effective_rank(selected_prediction)
            )
            target_effective_rank += batch_weight * float(
                entropy_effective_rank(selected_target)
            )
            query_count += batch_weight * float(output.query_valid.sum(dim=1).float().mean())
            normalization_scale += batch_weight * float(
                sum(prepared.normalization_scale) / len(prepared.normalization_scale)
            )
            normalization_median += batch_weight * float(
                sum(prepared.normalization_median) / len(prepared.normalization_median)
            )
            for key, value in output.node_field.diagnostics().items():
                node_diagnostics[key] += batch_weight * float(value)

        parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        grad_norm = float(grad_norm_tensor.detach())
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        momentum = float(self.ema_momentum(self.ema.updates))
        self.model.update_target(momentum)
        self.ema.updates += 1
        compute_elapsed = time.perf_counter() - compute_started
        loader_wait = sum(item.loader_wait_seconds for item in microbatches)
        modality_losses, modality_counts = _mean_by_key(modality_values)
        relation_losses, relation_counts = _mean_by_key(relation_values)
        loss_total = self_loss_value
        components = {
            "self": self_loss_value,
            "prediction_std": prediction_std,
            "target_std": target_std,
            "prediction_mean": prediction_mean,
            "target_mean": target_mean,
            "prediction_effective_rank": prediction_effective_rank,
            "target_effective_rank": target_effective_rank,
            "mean_query_count": query_count,
            "visible_normalization_median": normalization_median,
            "visible_normalization_scale": normalization_scale,
            "companion_requested_count": float(companion_requested_count),
            "companion_used_count": float(
                companion_requested_count - companion_fallback_count
            ),
            "companion_fallback_count": float(companion_fallback_count),
            **{f"node_{key}": value for key, value in node_diagnostics.items()},
        }
        ensure_finite_metrics(
            {
                "loss_total": loss_total,
                "loss_components": components,
                "modality_losses": modality_losses,
                "relation_losses": relation_losses,
                "learning_rates": used_learning_rates,
                "ema": momentum,
                "grad_norm": grad_norm,
            },
            context="formal optimizer step",
        )
        return TrainingStepResult(
            anchors_processed=total_anchors,
            next_anchor_index=microbatches[-1].next_anchor_index,
            elapsed_seconds=max(compute_elapsed + loader_wait, 1.0e-12),
            loader_wait_seconds=loader_wait,
            loss_total=loss_total,
            loss_components=components,
            modality_losses=modality_losses,
            modality_counts=modality_counts,
            relation_losses=relation_losses,
            relation_counts=relation_counts,
            learning_rates=used_learning_rates,
            ema_decay=momentum,
            grad_norm=grad_norm,
            encoded_volumes_processed=encoded_volumes,
            is_coverage_tail=total_anchors < int(expected_global_batch_anchors),
        )

    @torch.no_grad()
    def validate(self, request: ValidationRequest) -> Mapping[str, Any]:
        was_training = self.model.training
        self.model.eval()
        try:
            if self.validation_callback is not None:
                result = dict(self.validation_callback(request, self))
                ensure_finite_metrics(result, context="custom formal validation")
                return result
            if self.validation_source is None:
                raise ModelContractError("No observational validation source/callback configured")
            return self._validate_source(request)
        finally:
            self.model.train(was_training)

    def _validate_source(self, request: ValidationRequest) -> Mapping[str, Any]:
        assert self.validation_source is not None
        schedule = FormalTrainingSchedule()
        mixture = schedule.relation_mixture(request.coverage)
        total_loss = 0.0
        total_anchors = 0
        prediction_std = 0.0
        target_std = 0.0
        companion_requested_count = 0
        companion_fallback_count = 0
        subject_modality_values: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        subject_relation_values: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        source_total = self.validation_source.total_anchors(request.coverage)
        limit = min(source_total, request.sample_limit or source_total)
        evaluation_indices = self.validation_source.evaluation_anchor_indices(
            coverage=request.coverage,
            max_anchors=request.sample_limit,
        )
        if len(evaluation_indices) != limit:
            raise ModelContractError("Evaluation subset size differs from its fixed limit")
        expected_anchor_views = limit * request.masks_per_anchor
        label = f"{request.tier.value}: {request.reason}"
        if self.evaluation_progress is not None:
            self.evaluation_progress("start", 0, expected_anchor_views, label)
        try:
            for mask_view in range(request.masks_per_anchor):
                iterator = self.validation_source.iter_microbatches_for_view(
                    coverage=request.coverage,
                    start_anchor_index=0,
                    micro_batch_anchors=8,
                    relation_mixture=mixture,
                    view_id=0,
                    anchor_indices=evaluation_indices,
                )
                for microbatch in iterator:
                    prepared = self.preparer.prepare(
                        microbatch.payload,
                        mask_view_override=mask_view,
                        training=False,
                    )
                    companion_requested_count += prepared.companion_requested_count
                    companion_fallback_count += len(prepared.companion_fallbacks)
                    with self._autocast():
                        output = self.model(
                            prepared.anchor,
                            prepared.target_volume,
                            target_support=prepared.target_support,
                            query_indices=prepared.query_indices,
                            query_valid=prepared.query_valid,
                            relation_codes=prepared.relation_codes,
                            companion=prepared.companion,
                            return_diagnostics=False,
                        )
                        objective = masked_huber_objective(
                            output.predicted_latents,
                            output.target_latents,
                            output.query_valid,
                            beta=self.huber_beta,
                        )
                    per_example = _per_example_huber(
                        output.predicted_latents,
                        output.target_latents,
                        output.query_valid,
                        self.huber_beta,
                    )
                    count = prepared.anchor_count
                    total_loss += float(objective.loss) * count
                    total_anchors += count
                    prediction_std += float(
                        output.predicted_latents[output.query_valid]
                        .float()
                        .std(unbiased=False)
                    ) * count
                    target_std += float(
                        output.target_latents[output.query_valid]
                        .float()
                        .std(unbiased=False)
                    ) * count
                    for row, value in enumerate(per_example.tolist()):
                        subject_id = prepared.subject_ids[row]
                        subject_modality_values[prepared.modalities[row]][
                            subject_id
                        ].append(float(value))
                        subject_relation_values[prepared.relation_types[row]][
                            subject_id
                        ].append(float(value))
                    if self.evaluation_progress is not None:
                        self.evaluation_progress(
                            "update", total_anchors, expected_anchor_views, label
                        )
        except Exception:
            if self.evaluation_progress is not None:
                self.evaluation_progress(
                    "error", total_anchors, expected_anchor_views, label
                )
            raise
        if total_anchors != limit * request.masks_per_anchor:
            raise ResumeMismatchError("Observational validation did not consume its fixed set")

        def subject_macro(
            values: Mapping[str, Mapping[str, Sequence[float]]],
        ) -> tuple[dict[str, float], dict[str, int]]:
            losses: dict[str, float] = {}
            counts: dict[str, int] = {}
            for key, subjects in values.items():
                subject_means = [
                    sum(items) / len(items) for items in subjects.values() if items
                ]
                if subject_means:
                    losses[key] = float(sum(subject_means) / len(subject_means))
                    counts[key] = len(subject_means)
            return losses, counts

        modality_losses, modality_counts = subject_macro(subject_modality_values)
        relation_losses, relation_counts = subject_macro(subject_relation_values)
        required_modalities = {
            key
            for key, weight in self.validation_modality_weights.items()
            if weight > 0.0
        }
        missing_modalities = required_modalities - set(modality_losses)
        if missing_modalities:
            raise ModelContractError(
                "Fixed MRI/PET modality macro is missing: "
                + ", ".join(sorted(missing_modalities))
            )
        subject_modality_macro_loss = sum(
            modality_losses[key] * self.validation_modality_weights[key]
            for key in required_modalities
        )
        result: dict[str, Any] = {
            "loss_total": subject_modality_macro_loss,
            "loss_observation_mean": total_loss / total_anchors,
            "prediction_std": prediction_std / total_anchors,
            "target_std": target_std / total_anchors,
            "anchor_views": total_anchors,
            "companion_requested_count": companion_requested_count,
            "companion_used_count": (
                companion_requested_count - companion_fallback_count
            ),
            "companion_fallback_count": companion_fallback_count,
            "subject_count": len(
                {
                    subject
                    for subjects in subject_modality_values.values()
                    for subject in subjects
                }
            ),
            "aggregation": "subject_then_fixed_modality_macro",
            **{f"loss_modality_{key}": value for key, value in modality_losses.items()},
            **{f"count_modality_{key}": value for key, value in modality_counts.items()},
            **{f"loss_relation_{key}": value for key, value in relation_losses.items()},
            **{f"count_relation_{key}": value for key, value in relation_counts.items()},
        }
        ensure_finite_metrics(result, context="observational formal validation")
        if self.evaluation_progress is not None:
            self.evaluation_progress(
                "complete", total_anchors, expected_anchor_views, label
            )
        return result


@dataclass(frozen=True)
class FormalModelBuildSpec:
    sat3d_code_root: Path
    sat3d_checkpoint: Path
    fem_geometry_path: Path
    reference_shape: tuple[int, int, int]
    reference_affine: Sequence[Sequence[float]]
    expected_checkpoint_sha256: str
    expected_source_tree_sha256: str
    trainability: str = "all"
    activation_checkpointing: bool = True
    latent_dim: int = 128
    condition_dim: int = 128
    relation_dim: int = 32
    condition_categorical_embedding_dim: int = 16
    stem_hidden_channels: int = 8
    adapter_bottleneck_ratio: float = 0.25
    interpolation_neighbors: int = 8
    interpolation_sigma_mm: float = 12.0
    resolvent_radii: tuple[float, ...] = (3.0, 6.0, 12.0, 24.0, 48.0)
    geomc_blocks: int = 4
    geomc_hidden_dim: int = 256
    geomc_scale_embedding_dim: int = 16
    geomc_residual_scale: float = 0.10
    fusion_hidden_dim: int = 128
    geometry_embedding_hidden_dim: int = 64
    geometry_embedding_initial_scale: float = 0.05
    geometry_embedding_trainable_scale: bool = True
    predictor_dim: int = 256
    predictor_depth: int = 6
    predictor_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    predictor_drop_path: float = 0.10
    max_queries: int = 128
    geometry_feature_type: str = "spectral"
    mask_hidden_fraction: float = 0.30
    mask_blocks: int = 2
    sat3d_last_stage_lr: float = 1.0e-5
    sat3d_stage_multipliers: tuple[float, ...] = (0.35, 0.50, 0.70, 1.0)
    sat3d_patch_embed_multiplier: float = 0.25
    acquisition_lr: float = 1.0e-4
    model_lr: float = 2.0e-4
    weight_decay: float = 0.05
    warmup_coverages: int = 2
    final_lr_ratio: float = 0.01
    ema_initial_momentum: float = 0.996
    ema_final_momentum: float = 0.9999
    huber_beta: float = 1.0
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sat3d_code_root", Path(self.sat3d_code_root))
        object.__setattr__(self, "sat3d_checkpoint", Path(self.sat3d_checkpoint))
        object.__setattr__(self, "fem_geometry_path", Path(self.fem_geometry_path))
        if tuple(self.reference_shape) != (128, 128, 128):
            raise ValueError("The real SAT3D formal builder requires a 128^3 reference")
        affine = np.asarray(self.reference_affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("reference_affine must be a finite 4x4 matrix")
        if self.trainability != "all":
            raise ValueError("The real SAT3D formal builder requires trainability='all'")
        if not isinstance(self.activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be boolean")
        if not isinstance(self.geometry_embedding_trainable_scale, bool):
            raise ValueError("geometry_embedding_trainable_scale must be boolean")
        if not self.expected_checkpoint_sha256 or not self.expected_source_tree_sha256:
            raise ValueError("Real SAT3D builder requires checkpoint and source-tree SHA256")
        # These widths are already propagated through the common model builder;
        # only the pretrained SAT3D image/grid interface is fixed, not latent width.
        positive_integer_fields = (
            "latent_dim",
            "condition_dim",
            "relation_dim",
            "condition_categorical_embedding_dim",
            "stem_hidden_channels",
            "interpolation_neighbors",
            "geomc_blocks",
            "geomc_hidden_dim",
            "geomc_scale_embedding_dim",
            "fusion_hidden_dim",
            "geometry_embedding_hidden_dim",
            "predictor_dim",
            "predictor_depth",
            "predictor_heads",
            "max_queries",
            "mask_blocks",
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.geomc_blocks < 3:
            raise ValueError("geomc_blocks must be at least 3 for GeoMCCore")
        if self.predictor_dim < 6 or self.predictor_dim % self.predictor_heads:
            raise ValueError("predictor_dim must be at least 6 and divisible by predictor_heads")
        if self.max_queries > 8 * 8 * 8:
            raise ValueError("max_queries must lie in [1,512] for the fixed SAT3D token grid")
        if (
            isinstance(self.warmup_coverages, bool)
            or not isinstance(self.warmup_coverages, Integral)
            or self.warmup_coverages < 0
        ):
            raise ValueError("warmup_coverages must be a non-negative integer")

        positive_float_fields = (
            "interpolation_sigma_mm",
            "predictor_mlp_ratio",
            "sat3d_last_stage_lr",
            "sat3d_patch_embed_multiplier",
            "acquisition_lr",
            "model_lr",
            "huber_beta",
            "gradient_clip_norm",
        )
        for name in positive_float_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(round(self.predictor_dim * self.predictor_mlp_ratio)) < 1:
            raise ValueError("predictor_mlp_ratio must produce a positive hidden dimension")
        for name in ("geomc_residual_scale", "geometry_embedding_initial_scale"):
            # Both modules use a signed residual gain; zero is a valid warm start.
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.adapter_bottleneck_ratio) or not (
            0.0 < self.adapter_bottleneck_ratio <= 1.0
        ):
            raise ValueError("adapter_bottleneck_ratio must lie in (0,1]")
        if not math.isfinite(self.predictor_drop_path) or not (
            0.0 <= self.predictor_drop_path < 1.0
        ):
            raise ValueError("predictor_drop_path must lie in [0,1)")
        if not math.isfinite(self.mask_hidden_fraction) or not (
            0.0 < self.mask_hidden_fraction < 1.0
        ):
            raise ValueError("mask_hidden_fraction must lie in (0,1)")
        radii = tuple(float(value) for value in self.resolvent_radii)
        if (
            not radii
            or any(not math.isfinite(value) or value <= 0.0 for value in radii)
            or any(right <= left for left, right in zip(radii, radii[1:]))
        ):
            raise ValueError("resolvent_radii must be finite, positive and strictly increasing")
        multipliers = tuple(float(value) for value in self.sat3d_stage_multipliers)
        if len(multipliers) != 4 or any(
            not math.isfinite(value) or value <= 0.0 for value in multipliers
        ):
            raise ValueError("sat3d_stage_multipliers must contain four finite positive values")
        if self.geometry_feature_type not in {"spectral", "spectral_xyz"}:
            raise ValueError("Real formal geometry embedding must be spectral")
        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
            or not math.isfinite(self.final_lr_ratio)
            or not 0.0 < self.final_lr_ratio <= 1.0
        ):
            raise ValueError("Formal optimizer decay/scheduler ratio is invalid")
        if (
            not math.isfinite(self.ema_initial_momentum)
            or not math.isfinite(self.ema_final_momentum)
            or not 0.0 <= self.ema_initial_momentum <= self.ema_final_momentum < 1.0
        ):
            raise ValueError("Formal EMA momentum interval is invalid")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        sat3d_code_root: str | Path,
        sat3d_checkpoint: str | Path,
        fem_geometry_path: str | Path,
        reference_affine: Sequence[Sequence[float]],
    ) -> "FormalModelBuildSpec":
        """Construct the scientific build contract directly from formal YAML."""

        def section(parent: Mapping[str, Any], name: str) -> Mapping[str, Any]:
            value = parent.get(name)
            if not isinstance(value, Mapping):
                raise ModelContractError(f"Formal config section {name!r} is missing")
            return value

        model = section(config, "model")
        metadata = section(config, "metadata_conditioning")
        masking = section(config, "masking")
        training = section(config, "training")
        sat3d = section(model, "sat3d")
        adapters = section(model, "adapters")
        geometry = section(model, "geometry")
        fusion = section(model, "fusion")
        predictor = section(model, "predictor")
        embedding = section(geometry, "embedding")
        learning_rates = section(training, "learning_rates")
        scheduler = section(training, "scheduler")
        ema = section(training, "ema")
        loss = section(training, "loss")
        return cls(
            sat3d_code_root=Path(sat3d_code_root),
            sat3d_checkpoint=Path(sat3d_checkpoint),
            fem_geometry_path=Path(fem_geometry_path),
            reference_shape=tuple(int(value) for value in model["input_shape"]),
            reference_affine=reference_affine,
            expected_checkpoint_sha256=str(sat3d["expected_checkpoint_sha256"]),
            expected_source_tree_sha256=str(sat3d["expected_source_tree_sha256"]),
            trainability=str(sat3d["trainability"]),
            activation_checkpointing=bool(sat3d["activation_checkpointing"]),
            latent_dim=int(model["latent_dim"]),
            condition_dim=int(metadata["dimension"]),
            relation_dim=int(fusion["relation_embedding_dim"]),
            stem_hidden_channels=int(adapters["stem_hidden_channels"]),
            adapter_bottleneck_ratio=float(adapters["bottleneck_ratio"]),
            interpolation_neighbors=int(geometry["interpolation_neighbors"]),
            interpolation_sigma_mm=float(geometry["interpolation_sigma_mm"]),
            resolvent_radii=tuple(float(value) for value in geometry["resolvent_radii"]),
            geomc_blocks=int(geometry["geomc_blocks"]),
            geomc_hidden_dim=int(geometry["hidden_dim"]),
            geomc_scale_embedding_dim=int(geometry["scale_embedding_dim"]),
            geomc_residual_scale=float(geometry["residual_scale"]),
            fusion_hidden_dim=int(fusion["hidden_dim"]),
            geometry_embedding_hidden_dim=int(embedding["hidden_dim"]),
            geometry_embedding_initial_scale=float(embedding["initial_scale"]),
            geometry_embedding_trainable_scale=bool(embedding["trainable_scale"]),
            predictor_dim=int(predictor["embed_dim"]),
            predictor_depth=int(predictor["depth"]),
            predictor_heads=int(predictor["heads"]),
            predictor_mlp_ratio=float(predictor["mlp_ratio"]),
            predictor_drop_path=float(predictor["drop_path"]),
            max_queries=int(predictor["max_queries"]),
            geometry_feature_type=str(embedding["type"]),
            mask_hidden_fraction=float(masking["hidden_fraction"]),
            mask_blocks=int(masking["blocks"]),
            sat3d_last_stage_lr=float(learning_rates["sat3d_last_stage"]),
            sat3d_stage_multipliers=tuple(
                float(value) for value in learning_rates["sat3d_stage_multipliers"]
            ),
            sat3d_patch_embed_multiplier=float(
                learning_rates["sat3d_patch_embed_multiplier"]
            ),
            acquisition_lr=float(learning_rates["stems_adapters_projector"]),
            model_lr=float(learning_rates["geomc_predictor"]),
            weight_decay=float(training["weight_decay"]),
            warmup_coverages=int(scheduler["warmup_coverages"]),
            final_lr_ratio=float(scheduler["final_lr_ratio"]),
            ema_initial_momentum=float(ema["initial_momentum"]),
            ema_final_momentum=float(ema["final_momentum"]),
            huber_beta=float(loss["huber_beta"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )


@dataclass(frozen=True)
class FormalModelBuildResult:
    model: FormalGeoMCJEPAModel
    token_xyz_mm: Tensor
    geometry: FEMGeometry
    activation_checkpointing: ActivationCheckpointingReport
    specification: Mapping[str, Any]
    valid_for_scientific_results: bool


def _formal_model_from_parts(
    *,
    encoder: nn.Module,
    encoder_channels: int,
    token_xyz_mm: Tensor,
    geometry: FEMGeometry,
    vocabulary: MetadataVocabulary,
    latent_dim: int,
    condition_dim: int,
    relation_dim: int,
    condition_categorical_embedding_dim: int,
    stem_hidden_channels: int,
    adapter_bottleneck_ratio: float,
    resolvent_radii: Sequence[float],
    geomc_blocks: int,
    geomc_hidden_dim: int,
    geomc_scale_embedding_dim: int,
    geomc_residual_scale: float,
    fusion_hidden_dim: int,
    geometry_embedding_hidden_dim: int,
    geometry_embedding_initial_scale: float,
    geometry_embedding_trainable_scale: bool,
    predictor_dim: int,
    predictor_depth: int,
    predictor_heads: int,
    predictor_mlp_ratio: float,
    predictor_drop_path: float,
    max_queries: int,
    interpolation_neighbors: int,
    interpolation_sigma_mm: float,
    geometry_feature_type: str,
) -> FormalGeoMCJEPAModel:
    condition_encoder = AcquisitionConditionEncoder(
        vocabulary,
        output_dim=condition_dim,
        categorical_embedding_dim=condition_categorical_embedding_dim,
    )
    online = FormalEncoderBundle(
        stems=MRIPETResidualStems(hidden_channels=stem_hidden_channels),
        condition_encoder=condition_encoder,
        encoder=encoder,
        feature_adapter=ConditionedFeatureAdapter3D(
            encoder_channels,
            condition_dim,
            bottleneck_dim=max(
                1, int(round(encoder_channels * adapter_bottleneck_ratio))
            ),
        ),
        projector=ContentGatedTokenProjector(
            encoder_channels, latent_dim, condition_dim
        ),
    )
    encoder_pair = OnlineEMABundle(online)
    frame = ParsevalResolventFrame(
        geometry.eigenvalues,
        geometry.eigenvectors,
        geometry.mass,
        radii=tuple(float(value) for value in resolvent_radii),
        normalize_spectrum=False,
        factorization="parseval_sqrt",
    )
    token_to_node = build_interpolator(
        token_xyz_mm,
        geometry.node_xyz_mm,
        neighbors=int(interpolation_neighbors),
        sigma_mm=float(interpolation_sigma_mm),
    )
    fusion = AnchorCentricFusion(
        token_to_node,
        feature_dim=latent_dim,
        condition_dim=condition_dim,
        relation_dim=relation_dim,
        output_dim=latent_dim,
        hidden_dim=fusion_hidden_dim,
    )
    core = GeoMCCore(
        frame,
        latent_dim,
        input_dim=latent_dim,
        num_blocks=geomc_blocks,
        hidden_dim=geomc_hidden_dim,
        scale_embedding_dim=geomc_scale_embedding_dim,
        residual_scale=geomc_residual_scale,
    )
    geometry_embedding = GeometryEmbedding(
        frame,
        latent_dim,
        node_xyz_mm=geometry.node_xyz_mm,
        feature_type=geometry_feature_type,  # type: ignore[arg-type]
        hidden_dim=geometry_embedding_hidden_dim,
        initial_scale=geometry_embedding_initial_scale,
        trainable_scale=geometry_embedding_trainable_scale,
    )
    predictor = QueryAwareJEPA3DPredictor(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        relation_dim=relation_dim,
        d_model=predictor_dim,
        depth=predictor_depth,
        num_heads=predictor_heads,
        mlp_ratio=predictor_mlp_ratio,
        drop_path_rate=predictor_drop_path,
        max_queries=max_queries,
    )
    return FormalGeoMCJEPAModel(
        encoder_bundle=encoder_pair,
        fusion=fusion,
        core=core,
        geometry_embedding=geometry_embedding,
        predictor=predictor,
        token_xyz_mm=token_xyz_mm,
        node_xyz_mm=geometry.node_xyz_mm,
        relation_count=len(RELATION_CODES),
    )


def build_formal_model(
    spec: FormalModelBuildSpec,
    vocabulary: MetadataVocabulary,
) -> FormalModelBuildResult:
    """Build the real pretrained-SAT3D formal model with immutable geometry."""

    geometry = FEMGeometry.load(spec.fem_geometry_path)
    encoder = TrainableSAT3DEncoder.from_sat3d(
        spec.sat3d_code_root,
        spec.sat3d_checkpoint,
        trainability=spec.trainability,  # type: ignore[arg-type]
        input_shape=(1, *spec.reference_shape),
        output_shape=(384, 8, 8, 8),
        expected_checkpoint_sha256=spec.expected_checkpoint_sha256,
        expected_source_tree_sha256=spec.expected_source_tree_sha256,
    )
    image_encoder = getattr(encoder, "image_encoder", encoder)
    stages = getattr(image_encoder, "layers", None)
    expected_stage_count = (
        len(stages)
        if isinstance(stages, (nn.ModuleList, nn.Sequential))
        else 0
    )
    if spec.activation_checkpointing and expected_stage_count < 1:
        raise ModelContractError("Loaded SAT3D exposes no checkpointable stages")
    checkpoint_report = configure_sat3d_activation_checkpointing(
        encoder, enabled=spec.activation_checkpointing
    )
    if spec.activation_checkpointing and (
        not checkpoint_report.enabled
        or checkpoint_report.safe_wrapper_count != expected_stage_count
        or len(checkpoint_report.stage_names) != expected_stage_count
        or checkpoint_report.native_flags_enabled
    ):
        raise ModelContractError(
            "Real SAT3D did not satisfy safe whole-stage activation checkpointing"
        )
    affine = np.asarray(spec.reference_affine, dtype=np.float64)
    token_xyz = token_centers_from_reference(
        spec.reference_shape, affine, grid_shape=(8, 8, 8)
    )
    model = _formal_model_from_parts(
        encoder=encoder,
        encoder_channels=384,
        token_xyz_mm=token_xyz,
        geometry=geometry,
        vocabulary=vocabulary,
        latent_dim=spec.latent_dim,
        condition_dim=spec.condition_dim,
        relation_dim=spec.relation_dim,
        condition_categorical_embedding_dim=spec.condition_categorical_embedding_dim,
        stem_hidden_channels=spec.stem_hidden_channels,
        adapter_bottleneck_ratio=spec.adapter_bottleneck_ratio,
        resolvent_radii=spec.resolvent_radii,
        geomc_blocks=spec.geomc_blocks,
        geomc_hidden_dim=spec.geomc_hidden_dim,
        geomc_scale_embedding_dim=spec.geomc_scale_embedding_dim,
        geomc_residual_scale=spec.geomc_residual_scale,
        fusion_hidden_dim=spec.fusion_hidden_dim,
        geometry_embedding_hidden_dim=spec.geometry_embedding_hidden_dim,
        geometry_embedding_initial_scale=spec.geometry_embedding_initial_scale,
        geometry_embedding_trainable_scale=spec.geometry_embedding_trainable_scale,
        predictor_dim=spec.predictor_dim,
        predictor_depth=spec.predictor_depth,
        predictor_heads=spec.predictor_heads,
        predictor_mlp_ratio=spec.predictor_mlp_ratio,
        predictor_drop_path=spec.predictor_drop_path,
        max_queries=spec.max_queries,
        interpolation_neighbors=spec.interpolation_neighbors,
        interpolation_sigma_mm=spec.interpolation_sigma_mm,
        geometry_feature_type=spec.geometry_feature_type,
    )
    specification = {
        **model.model_specification(),
        "sat3d_provenance": encoder.provenance,
        "fem_geometry_hash": geometry.metadata.get("geometry_hash"),
        "reference_shape": list(spec.reference_shape),
        "formal_yaml_parameters": {
            "interpolation_sigma_mm": spec.interpolation_sigma_mm,
            "geometry_feature_type": spec.geometry_feature_type,
            "stem_hidden_channels": spec.stem_hidden_channels,
            "adapter_bottleneck_ratio": spec.adapter_bottleneck_ratio,
            "geomc_scale_embedding_dim": spec.geomc_scale_embedding_dim,
            "geomc_residual_scale": spec.geomc_residual_scale,
            "fusion_hidden_dim": spec.fusion_hidden_dim,
            "geometry_embedding_hidden_dim": spec.geometry_embedding_hidden_dim,
            "geometry_embedding_initial_scale": spec.geometry_embedding_initial_scale,
            "expected_activation_checkpoint_stages": expected_stage_count,
        },
        "valid_for_scientific_results": True,
    }
    return FormalModelBuildResult(
        model=model,
        token_xyz_mm=token_xyz,
        geometry=geometry,
        activation_checkpointing=checkpoint_report,
        specification=specification,
        valid_for_scientific_results=True,
    )


def _tiny_geometry() -> FEMGeometry:
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=torch.float64,
    )
    # Tiny geometry is never serialized as a scientific FEM artifact; only the
    # spectral fields are consumed by the tiny contract model.
    return FEMGeometry(
        node_xyz_mm=coordinates,
        tetrahedra=torch.tensor([[0, 1, 2, 4]], dtype=torch.long),
        mass=torch.ones(8, dtype=torch.float64),
        stiffness_indices=torch.tensor([[0], [0]], dtype=torch.long),
        stiffness_values=torch.zeros(1, dtype=torch.float64),
        eigenvalues=torch.tensor([0.0, 0.1, 0.4, 1.0], dtype=torch.float64),
        eigenvectors=torch.eye(8, dtype=torch.float64)[:, :4],
        metadata={"geometry_hash": "tiny-synthetic-not-scientific"},
    )


def build_tiny_synthetic_formal_model(
    vocabulary: MetadataVocabulary,
    *,
    seed: int = 7,
) -> FormalModelBuildResult:
    """Build a tiny CPU model that is explicitly invalid for scientific use."""

    encoder = TrainableSAT3DEncoder.tiny(
        trainability="all",
        input_size=8,
        hidden_channels=4,
        output_channels=8,
        output_grid=2,
        seed=seed,
    )
    checkpoint_report = configure_sat3d_activation_checkpointing(encoder, enabled=True)
    geometry = _tiny_geometry()
    token_xyz = geometry.node_xyz_mm.clone()
    model = _formal_model_from_parts(
        encoder=encoder,
        encoder_channels=8,
        token_xyz_mm=token_xyz,
        geometry=geometry,
        vocabulary=vocabulary,
        latent_dim=8,
        condition_dim=8,
        relation_dim=4,
        condition_categorical_embedding_dim=4,
        stem_hidden_channels=2,
        adapter_bottleneck_ratio=0.50,
        resolvent_radii=(1.0, 2.0),
        geomc_blocks=3,
        geomc_hidden_dim=16,
        geomc_scale_embedding_dim=4,
        geomc_residual_scale=0.10,
        fusion_hidden_dim=8,
        geometry_embedding_hidden_dim=8,
        geometry_embedding_initial_scale=0.05,
        geometry_embedding_trainable_scale=True,
        predictor_dim=32,
        predictor_depth=1,
        predictor_heads=4,
        predictor_mlp_ratio=2.0,
        predictor_drop_path=0.0,
        max_queries=4,
        interpolation_neighbors=1,
        interpolation_sigma_mm=1.0,
        geometry_feature_type="none",
    )
    return FormalModelBuildResult(
        model=model,
        token_xyz_mm=token_xyz,
        geometry=geometry,
        activation_checkpointing=checkpoint_report,
        specification={
            **model.model_specification(),
            "backend": "tiny_synthetic_contract_only",
            "valid_for_scientific_results": False,
        },
        valid_for_scientific_results=False,
    )


def formal_optimizer_parameter_groups(
    model: FormalGeoMCJEPAModel,
    *,
    sat3d_last_stage_lr: float = 1.0e-5,
    stage_multipliers: Sequence[float] = (0.35, 0.50, 0.70, 1.0),
    patch_embed_multiplier: float = 0.25,
    acquisition_lr: float = 1.0e-4,
    model_lr: float = 2.0e-4,
    weight_decay: float = 0.05,
) -> list[dict[str, Any]]:
    if min(sat3d_last_stage_lr, acquisition_lr, model_lr) <= 0 or weight_decay < 0:
        raise ValueError("Formal optimizer learning rates/weight decay are invalid")
    online = model.encoder_bundle.online
    encoder = online.encoder
    image_encoder = getattr(encoder, "image_encoder", encoder)
    groups: list[dict[str, Any]] = []
    assigned: set[int] = set()

    def add(name: str, parameters: Iterable[nn.Parameter], learning_rate: float) -> None:
        selected = [
            parameter
            for parameter in parameters
            if parameter.requires_grad and id(parameter) not in assigned
        ]
        if not selected:
            return
        assigned.update(id(parameter) for parameter in selected)
        groups.append(
            {
                "name": name,
                "params": selected,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            }
        )

    patch_embed = getattr(image_encoder, "patch_embed", None)
    if isinstance(patch_embed, nn.Module):
        add(
            "sat3d_patch_embed",
            patch_embed.parameters(),
            sat3d_last_stage_lr * patch_embed_multiplier,
        )
    layers = getattr(image_encoder, "layers", None)
    if isinstance(layers, (nn.ModuleList, nn.Sequential)):
        multipliers = list(float(value) for value in stage_multipliers)
        if len(multipliers) < len(layers):
            raise ValueError("Not enough SAT3D stage learning-rate multipliers")
        multipliers = multipliers[-len(layers) :]
        for index, (layer, multiplier) in enumerate(zip(layers, multipliers)):
            add(
                f"sat3d_stage_{index}",
                layer.parameters(),
                sat3d_last_stage_lr * multiplier,
            )
    add("sat3d_remaining", encoder.parameters(), sat3d_last_stage_lr)
    acquisition_modules = (
        online.stems,
        online.condition_encoder,
        online.feature_adapter,
        online.projector,
    )
    add(
        "stems_adapters_condition_projector",
        (parameter for module in acquisition_modules for parameter in module.parameters()),
        acquisition_lr,
    )
    scientific_modules = (
        model.fusion,
        model.core,
        model.geometry_embedding,
        model.predictor,
        model.relation_embedding,
    )
    add(
        "geomc_fusion_predictor",
        (parameter for module in scientific_modules for parameter in module.parameters()),
        model_lr,
    )
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if assigned != expected:
        raise ModelContractError(
            f"Optimizer grouping missed {len(expected - assigned)} trainable parameters"
        )
    return groups


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    warmup_updates: int,
    total_updates: int,
    final_lr_ratio: float = 0.01,
) -> LambdaLR:
    if not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates must lie in [0,total_updates)")
    if not 0.0 < final_lr_ratio <= 1.0:
        raise ValueError("final_lr_ratio must lie in (0,1]")

    def multiplier(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return float(step + 1) / float(warmup_updates)
        progress = (step - warmup_updates) / float(
            max(total_updates - warmup_updates, 1)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(final_lr_ratio + (1.0 - final_lr_ratio) * cosine)

    return LambdaLR(optimizer, lr_lambda=multiplier)


def _coverage_aligned_warmup_updates(
    *,
    total_optimizer_steps: int,
    total_coverages: int,
    warmup_coverages: int,
) -> int:
    """Convert a configured coverage count into an exact update count."""

    total_steps = int(total_optimizer_steps)
    coverages = int(total_coverages)
    warmup = int(warmup_coverages)
    if total_steps < 1 or coverages < 1:
        raise ValueError("Training steps and coverages must be positive")
    if total_steps % coverages:
        raise ValueError("Optimizer steps must be coverage-aligned")
    if not 0 <= warmup < coverages:
        raise ValueError("warmup_coverages must lie in [0,total_coverages)")
    return (total_steps // coverages) * warmup


@dataclass(frozen=True)
class FormalTrainingComponents:
    model_result: FormalModelBuildResult
    masker: DeterministicPhysicalBlockMasker
    preparer: FormalBatchPreparer
    optimizer: Optimizer
    scheduler: LRScheduler
    backend: TorchFormalTrainingBackend


def build_real_training_components(
    *,
    spec: FormalModelBuildSpec,
    vocabulary: MetadataVocabulary,
    device: torch.device | str,
    total_optimizer_steps: int,
    total_coverages: int,
    validation_source: FormalCoverageDataSource | None = None,
    evaluation_progress: Callable[[str, int, int, str], None] | None = None,
    validation_modality_weights: Mapping[str, float] | None = None,
    precision: str = "bf16",
    metadata_dropout_probability: float = 0.30,
    mask_seed: int = 20260822,
    companion_fallback_callback: (
        Callable[[Sequence[Mapping[str, Any]]], None] | None
    ) = None,
) -> FormalTrainingComponents:
    built = build_formal_model(spec, vocabulary)
    masker = DeterministicPhysicalBlockMasker(
        built.token_xyz_mm,
        grid_shape=(8, 8, 8),
        volume_shape=spec.reference_shape,
        hidden_fraction=spec.mask_hidden_fraction,
        max_queries=spec.max_queries,
        blocks=spec.mask_blocks,
        seed=mask_seed,
    )
    preparer = FormalBatchPreparer(
        vocabulary=vocabulary,
        masker=masker,
        device=device,
        metadata_dropout_probability=metadata_dropout_probability,
        metadata_dropout_seed=mask_seed,
        companion_fallback_callback=companion_fallback_callback,
    )
    optimizer = AdamW(
        formal_optimizer_parameter_groups(
            built.model,
            sat3d_last_stage_lr=spec.sat3d_last_stage_lr,
            stage_multipliers=spec.sat3d_stage_multipliers,
            patch_embed_multiplier=spec.sat3d_patch_embed_multiplier,
            acquisition_lr=spec.acquisition_lr,
            model_lr=spec.model_lr,
            weight_decay=spec.weight_decay,
        )
    )
    warmup_updates = _coverage_aligned_warmup_updates(
        total_optimizer_steps=total_optimizer_steps,
        total_coverages=total_coverages,
        warmup_coverages=spec.warmup_coverages,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_updates=warmup_updates,
        total_updates=int(total_optimizer_steps),
        final_lr_ratio=spec.final_lr_ratio,
    )
    backend = TorchFormalTrainingBackend(
        model=built.model,
        optimizer=optimizer,
        preparer=preparer,
        ema_momentum=CosineEMAMomentum(
            initial=spec.ema_initial_momentum,
            final=spec.ema_final_momentum,
            total_updates=int(total_optimizer_steps),
        ),
        scheduler=scheduler,
        validation_source=validation_source,
        evaluation_progress=evaluation_progress,
        validation_modality_weights=validation_modality_weights,
        precision=precision,
        huber_beta=spec.huber_beta,
        gradient_clip_norm=spec.gradient_clip_norm,
    )
    return FormalTrainingComponents(
        model_result=built,
        masker=masker,
        preparer=preparer,
        optimizer=optimizer,
        scheduler=scheduler,
        backend=backend,
    )


@dataclass(frozen=True)
class TrainingLoopResult:
    """Outcome of one invocation of the fixed-horizon formal training loop."""

    completed: bool
    resumed: bool
    already_complete: bool
    cursor: CheckpointCursor
    trained_anchors: int
    trained_optimizer_steps: int
    repeated_validation_coverages: tuple[int, ...]
    held_out_test_complete: bool
    final_checkpoint: Path | None
    completion_marker: Path | None


def _completion_state(
    *,
    controller: TrainingRuntimeController,
    cursor: CheckpointCursor,
    checkpoint_name: str,
    checkpoint_sha256: str,
    held_out_test_complete: bool = False,
    held_out_test_state_path: str | None = None,
) -> ResumeState:
    return ResumeState(
        kind="training",
        run_id=controller.run_id,
        hashes=controller.hashes,
        status="completed",
        cursor=cursor.as_dict(),
        payload={
            "coverage_complete": True,
            "final_complete": True,
            "no_performance_early_stop": True,
            "checkpoint_name": checkpoint_name,
            "checkpoint_sha256": checkpoint_sha256,
            "held_out_test_complete": bool(held_out_test_complete),
            "held_out_test_state_path": held_out_test_state_path,
        },
    )


def _validate_completed_marker(
    *,
    state_store: AtomicStateStore,
    controller: TrainingRuntimeController,
    backend: TrainingBackend[Any],
) -> tuple[ResumeState, Any]:
    state = state_store.load(
        expected_kind="training",
        expected_run_id=controller.run_id,
        expected_hashes=controller.hashes,
    )
    if state.status != "completed" or not bool(state.payload.get("final_complete")):
        raise ResumeMismatchError("Training completion marker is not final")
    loaded = controller.resume(backend=backend)
    expected_coverage = controller.schedule.total_coverages
    if (
        loaded.cursor.coverage != expected_coverage
        or not bool(loaded.extra.get("coverage_complete"))
        or not bool(loaded.extra.get("final_complete"))
    ):
        raise ResumeMismatchError(
            "Completion marker does not point to a final coverage checkpoint"
        )
    if (
        str(state.payload.get("checkpoint_name", "")) != loaded.path.name
        or str(state.payload.get("checkpoint_sha256", "")) != loaded.sha256
        or dict(state.cursor) != loaded.cursor.as_dict()
    ):
        raise ResumeMismatchError(
            "Completion marker differs from the verified latest checkpoint"
        )
    return state, loaded


def _checkpoint_extra(
    *,
    total_anchors: int,
    training_anchors_complete: bool,
    coverage_complete: bool,
    final_complete: bool,
    validation_tiers: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "total_anchors": int(total_anchors),
        "training_anchors_complete": bool(training_anchors_complete),
        "coverage_complete": bool(coverage_complete),
        "final_complete": bool(final_complete),
        "validation_tiers": [str(value) for value in validation_tiers],
        "observational_validation_only": True,
        "no_performance_early_stop": True,
    }


def run_full_training(
    *,
    controller: TrainingRuntimeController,
    data_source: FormalCoverageDataSource,
    backend: TrainingBackend[BatchPayload],
    resume: bool = True,
    completion_state_store: AtomicStateStore | None = None,
    keep_last: int = 3,
    keep_every_coverages: int | None = 5,
    explicit_keep_names: Sequence[str] = (),
) -> TrainingLoopResult:
    """Run every deterministic anchor for the configured fixed horizon.

    Validation is observational only and is never consulted to decide whether
    training continues.  A checkpoint written after all anchors but before
    validation makes a validation crash resumable without replaying training.
    The final completed marker is written only after the final checkpoint is
    atomically committed and verified through the latest-pointer contract.
    """

    if controller.schedule.total_coverages < 1:
        raise ModelContractError(
            "Formal integration requires a positive coverage horizon"
        )
    if controller.schedule.batch.global_batch_anchors != 16:
        raise ModelContractError("Formal integration requires global batch 16")
    if int(keep_last) < 1:
        raise ValueError("keep_last must be positive")
    state_store = completion_state_store or AtomicStateStore(
        controller.checkpoints.directory / "training_complete.json"
    )
    controller.initialize(resume_requested=bool(resume))
    latest_exists = controller.checkpoints.latest_path.is_file()
    if (state_store.exists or latest_exists) and not resume:
        raise ResumeMismatchError(
            "A formal training state already exists; explicit resume is required"
        )

    if state_store.exists:
        completed_state, loaded = _validate_completed_marker(
            state_store=state_store,
            controller=controller,
            backend=backend,
        )
        controller.journal.event(
            "training_already_complete",
            "Verified final checkpoint and completion marker; no training was replayed",
            cursor=loaded.cursor.as_dict(),
            checkpoint_path=str(loaded.path),
        )
        controller.journal.sync()
        return TrainingLoopResult(
            completed=True,
            resumed=True,
            already_complete=True,
            cursor=loaded.cursor,
            trained_anchors=0,
            trained_optimizer_steps=0,
            repeated_validation_coverages=(),
            held_out_test_complete=bool(
                completed_state.payload.get("held_out_test_complete", False)
            ),
            final_checkpoint=loaded.path,
            completion_marker=state_store.path,
        )

    loaded = controller.resume(backend=backend) if latest_exists else None
    resumed = loaded is not None
    if loaded is None:
        cursor = CheckpointCursor(
            epoch=1,
            coverage=1,
            next_anchor_index=0,
            global_step=0,
            optimizer_step=0,
            accumulation_index=0,
        )
        start_coverage = 1
        start_anchor = 0
    else:
        cursor = loaded.cursor
        if not 1 <= cursor.coverage <= controller.schedule.total_coverages:
            raise ResumeMismatchError("Checkpoint coverage lies outside the fixed horizon")
        total_at_resume = data_source.total_anchors(cursor.coverage)
        recorded_total = int(loaded.extra.get("total_anchors", total_at_resume))
        if recorded_total != total_at_resume:
            raise ResumeMismatchError(
                "Checkpoint coverage size differs from the current deterministic dataset"
            )
        if not 0 <= cursor.next_anchor_index <= total_at_resume:
            raise ResumeMismatchError("Checkpoint anchor cursor lies outside its coverage")
        coverage_complete = bool(loaded.extra.get("coverage_complete", False))
        final_complete = bool(loaded.extra.get("final_complete", False))
        if coverage_complete and cursor.next_anchor_index != total_at_resume:
            raise ResumeMismatchError(
                "Coverage-complete checkpoint did not consume every anchor"
            )
        if final_complete:
            if not coverage_complete or cursor.coverage != controller.schedule.total_coverages:
                raise ResumeMismatchError("Invalid final-complete checkpoint flags")
            completion = _completion_state(
                controller=controller,
                cursor=cursor,
                checkpoint_name=loaded.path.name,
                checkpoint_sha256=loaded.sha256,
            )
            state_store.save(completion)
            controller.journal.event(
                "training_completion_marker_recovered",
                "Recovered the missing completion marker from the verified final checkpoint",
                checkpoint_path=str(loaded.path),
                cursor=cursor.as_dict(),
            )
            controller.journal.sync()
            return TrainingLoopResult(
                completed=True,
                resumed=True,
                already_complete=True,
                cursor=cursor,
                trained_anchors=0,
                trained_optimizer_steps=0,
                repeated_validation_coverages=(),
                held_out_test_complete=False,
                final_checkpoint=loaded.path,
                completion_marker=state_store.path,
            )
        if coverage_complete:
            start_coverage = cursor.coverage + 1
            start_anchor = 0
        else:
            start_coverage = cursor.coverage
            start_anchor = cursor.next_anchor_index

    if int(getattr(backend.ema, "updates", cursor.optimizer_step)) != cursor.optimizer_step:
        raise ResumeMismatchError("EMA update count differs from optimizer-step cursor")
    if start_coverage > controller.schedule.total_coverages:
        raise ResumeMismatchError(
            "Non-final checkpoint points beyond the configured coverage horizon"
        )

    trained_anchors = 0
    trained_optimizer_steps = 0
    repeated_validation: list[int] = []
    last_checkpoint: Path | None = None if loaded is None else loaded.path
    last_metrics: Mapping[str, Any] = {} if loaded is None else loaded.metrics

    for coverage in range(start_coverage, controller.schedule.total_coverages + 1):
        total_anchors = data_source.total_anchors(coverage)
        coverage_start = start_anchor if coverage == start_coverage else 0
        mixture = controller.relation_mixture(coverage)
        actual_mixture = data_source.assert_relation_mixture(coverage, mixture)
        controller.journal.event(
            "relation_mixture_contract_verified",
            "Runtime and dataset relation schedules agree exactly",
            coverage=coverage,
            runtime=FormalCoverageDataSource._runtime_weights(mixture),
            data=dict(actual_mixture),
        )
        controller.journal.sync()
        controller.begin_coverage(
            coverage,
            total_anchors=total_anchors,
            start_anchor_index=coverage_start,
        )
        if coverage_start == total_anchors:
            repeated_validation.append(coverage)
            controller.journal.warning(
                "observational_validation_repeat_after_resume",
                "All training anchors were already checkpointed; observational validation may repeat",
                coverage=coverage,
                optimizer_step=cursor.optimizer_step,
                affects_training_continuation=False,
            )

        iterator = iter(
            data_source.iter_microbatches(
                coverage=coverage,
                start_anchor_index=coverage_start,
                micro_batch_anchors=controller.schedule.batch.micro_batch_anchors,
                relation_mixture=mixture,
            )
        )
        while True:
            accumulation: list[Microbatch[BatchPayload]] = []
            for _ in range(
                controller.schedule.batch.gradient_accumulation_steps
            ):
                try:
                    accumulation.append(next(iterator))
                except StopIteration:
                    break
            if not accumulation:
                if cursor.coverage != coverage or cursor.next_anchor_index != total_anchors:
                    raise ResumeMismatchError(
                        "DataLoader exhausted before the exact coverage cursor"
                    )
                break
            anchors_in_step = sum(item.anchor_count for item in accumulation)
            is_short_tail = (
                anchors_in_step < controller.schedule.batch.global_batch_anchors
            )
            result = controller.train_accumulated_step(
                backend=backend,
                microbatches=tuple(accumulation),
                coverage=coverage,
                is_coverage_tail=is_short_tail,
            )
            cursor = CheckpointCursor(
                epoch=coverage,
                coverage=coverage,
                next_anchor_index=result.next_anchor_index,
                global_step=cursor.global_step + len(accumulation),
                optimizer_step=cursor.optimizer_step + 1,
                accumulation_index=0,
            )
            if int(getattr(backend.ema, "updates", cursor.optimizer_step)) != cursor.optimizer_step:
                raise ModelContractError(
                    "Backend EMA update count did not advance with optimizer step"
                )
            telemetry = controller.record_optimizer_step(cursor=cursor, result=result)
            last_metrics = telemetry.as_dict()
            trained_anchors += result.anchors_processed
            trained_optimizer_steps += 1
            if controller.checkpoint_due(optimizer_step=cursor.optimizer_step):
                last_checkpoint = controller.save_checkpoint(
                    cursor=cursor,
                    backend=backend,
                    metrics=last_metrics,
                    extra=_checkpoint_extra(
                        total_anchors=total_anchors,
                        training_anchors_complete=(
                            cursor.next_anchor_index == total_anchors
                        ),
                        coverage_complete=False,
                        final_complete=False,
                    ),
                )
            if cursor.next_anchor_index == total_anchors:
                break

        if cursor.next_anchor_index != total_anchors:
            raise ResumeMismatchError("Formal coverage ended at an inexact anchor cursor")

        # Commit consumed anchors before validation.  If validation crashes,
        # resume repeats only observational validation and never optimizer work.
        last_checkpoint = controller.save_checkpoint(
            cursor=cursor,
            backend=backend,
            metrics=last_metrics,
            extra=_checkpoint_extra(
                total_anchors=total_anchors,
                training_anchors_complete=True,
                coverage_complete=False,
                final_complete=False,
            ),
        )
        # Anchor training is complete, so close the training progress view
        # before observational validation opens its own progress stage.  The
        # persisted coverage_complete flag remains false until validation
        # succeeds, preserving the exact resume semantics below.
        controller.finish_coverage(
            coverage=coverage,
            optimizer_step=cursor.optimizer_step,
        )
        validation_tiers: list[str] = []
        for request in controller.validation_requests(
            coverage=coverage,
            optimizer_step=cursor.optimizer_step,
        ):
            controller.run_validation(backend=backend, request=request)
            validation_tiers.append(request.tier.value)
        final_complete = coverage == controller.schedule.total_coverages
        last_checkpoint = controller.save_checkpoint(
            cursor=cursor,
            backend=backend,
            metrics=last_metrics,
            extra=_checkpoint_extra(
                total_anchors=total_anchors,
                training_anchors_complete=True,
                coverage_complete=True,
                final_complete=final_complete,
                validation_tiers=validation_tiers,
            ),
        )
        controller.checkpoints.prune(
            keep_last=int(keep_last),
            keep_every_coverages=keep_every_coverages,
            keep_names=tuple(str(value) for value in explicit_keep_names),
        )
        start_anchor = 0

    assert last_checkpoint is not None
    loaded_final = controller.resume(backend=backend)
    if (
        loaded_final.path != last_checkpoint.resolve()
        or not bool(loaded_final.extra.get("coverage_complete"))
        or not bool(loaded_final.extra.get("final_complete"))
        or loaded_final.cursor.coverage != controller.schedule.total_coverages
    ):
        raise ResumeMismatchError("Final checkpoint verification failed")
    state_store.save(
        _completion_state(
            controller=controller,
            cursor=loaded_final.cursor,
            checkpoint_name=loaded_final.path.name,
            checkpoint_sha256=loaded_final.sha256,
        )
    )
    controller.journal.event(
        "formal_training_completed",
        f"Completed all {controller.schedule.total_coverages} coverages without a performance gate",
        cursor=loaded_final.cursor.as_dict(),
        final_checkpoint=str(loaded_final.path),
        completion_marker=str(state_store.path),
        trained_anchors_this_invocation=trained_anchors,
        trained_optimizer_steps_this_invocation=trained_optimizer_steps,
        repeated_validation_coverages=repeated_validation,
        no_performance_early_stop=True,
    )
    controller.journal.sync()
    return TrainingLoopResult(
        completed=True,
        resumed=resumed,
        already_complete=False,
        cursor=loaded_final.cursor,
        trained_anchors=trained_anchors,
        trained_optimizer_steps=trained_optimizer_steps,
        repeated_validation_coverages=tuple(repeated_validation),
        held_out_test_complete=False,
        final_checkpoint=loaded_final.path,
        completion_marker=state_store.path,
    )


def run_final_test_once(
    *,
    controller: TrainingRuntimeController,
    backend: TrainingBackend[Any],
    callback: Callable[[TrainingBackend[Any]], Mapping[str, Any]],
    completion_state_store: AtomicStateStore | None = None,
    test_state_store: AtomicStateStore | None = None,
) -> Mapping[str, Any]:
    """Run a held-out TEST only after the configured final coverage, once per run.

    Health monitoring and five-coverage full validation remain observational
    training-time checks.  This separate callback is the held-out TEST split.
    Its own atomic state is committed before the training completion marker is
    amended, so a crash between those two writes recovers without evaluating
    the TEST split again.
    """

    completion_store = completion_state_store or AtomicStateStore(
        controller.checkpoints.directory / "training_complete.json"
    )
    test_store = test_state_store or AtomicStateStore(
        controller.checkpoints.directory / "held_out_test_complete.json"
    )
    completion, loaded = _validate_completed_marker(
        state_store=completion_store,
        controller=controller,
        backend=backend,
    )
    if loaded.cursor.coverage != controller.schedule.total_coverages:
        raise ResumeMismatchError(
            "Held-out TEST is forbidden before the configured final coverage"
        )

    if test_store.exists:
        test_state = test_store.load(
            expected_kind="training",
            expected_run_id=controller.run_id,
            expected_hashes=controller.hashes,
        )
        if (
            test_state.status != "completed"
            or not bool(test_state.payload.get("held_out_test_complete"))
            or dict(test_state.cursor) != loaded.cursor.as_dict()
            or str(test_state.payload.get("checkpoint_sha256", "")) != loaded.sha256
        ):
            raise ResumeMismatchError("Held-out TEST state is not bound to the final model")
        metrics = dict(_as_mapping(test_state.payload.get("metrics")))
        ensure_finite_metrics(metrics, context="restored held-out TEST metrics")
        if not bool(completion.payload.get("held_out_test_complete")):
            completion_store.save(
                _completion_state(
                    controller=controller,
                    cursor=loaded.cursor,
                    checkpoint_name=loaded.path.name,
                    checkpoint_sha256=loaded.sha256,
                    held_out_test_complete=True,
                    held_out_test_state_path=str(test_store.path),
                )
            )
            controller.journal.event(
                "held_out_test_completion_marker_recovered",
                "Recovered TEST completion without repeating evaluation",
                checkpoint_path=str(loaded.path),
                test_state_path=str(test_store.path),
            )
            controller.journal.sync()
        else:
            controller.journal.event(
                "held_out_test_already_complete",
                "Verified TEST state; evaluation was not repeated",
                checkpoint_path=str(loaded.path),
                test_state_path=str(test_store.path),
            )
        return metrics

    if bool(completion.payload.get("held_out_test_complete")):
        raise ResumeMismatchError(
            "Training marker claims held-out TEST completion but its state is missing"
        )
    metrics = dict(callback(backend))
    ensure_finite_metrics(metrics, context="held-out TEST metrics")
    test_state = ResumeState(
        kind="training",
        run_id=controller.run_id,
        hashes=controller.hashes,
        status="completed",
        cursor=loaded.cursor.as_dict(),
        payload={
            "held_out_test_complete": True,
            "checkpoint_name": loaded.path.name,
            "checkpoint_sha256": loaded.sha256,
            "metrics": metrics,
            "observational_only": True,
            "affects_training_continuation": False,
        },
    )
    test_store.save(test_state)
    completion_store.save(
        _completion_state(
            controller=controller,
            cursor=loaded.cursor,
            checkpoint_name=loaded.path.name,
            checkpoint_sha256=loaded.sha256,
            held_out_test_complete=True,
            held_out_test_state_path=str(test_store.path),
        )
    )
    controller.journal.metric(
        "held_out_test_completed",
        metrics,
        coverage=loaded.cursor.coverage,
        optimizer_step=loaded.cursor.optimizer_step,
        checkpoint_path=str(loaded.path),
        test_state_path=str(test_store.path),
        ran_after_fixed_horizon=True,
        affects_training_continuation=False,
    )
    controller.journal.sync()
    return metrics


def run_synthetic_smoke(
    *,
    work_dir: str | Path,
    device: torch.device | str = "cpu",
) -> Mapping[str, Any]:
    """Execute one tiny optimizer step; never treat it as scientific evidence."""

    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "canonical_subject_id": "synthetic-mri-subject",
            "modality": "mri",
            "sequence": "T1w",
            "product": "native",
            "derived": False,
            "preprocessing": {"geometry": {"native_spacing": [1.0, 1.0, 1.0]}},
        },
        {
            "canonical_subject_id": "synthetic-pet-subject",
            "modality": "pet",
            "tracer_family": "amyloid",
            "tracer": "florbetapir",
            "derived": False,
            "preprocessing": {"geometry": {"native_spacing": [2.0, 2.0, 2.0]}},
        },
    ]
    vocabulary = build_acquisition_vocabulary(records)
    built = build_tiny_synthetic_formal_model(vocabulary)
    masker = DeterministicPhysicalBlockMasker(
        built.token_xyz_mm,
        grid_shape=(2, 2, 2),
        volume_shape=(8, 8, 8),
        hidden_fraction=0.50,
        max_queries=4,
        blocks=2,
        seed=17,
    )
    preparer = FormalBatchPreparer(
        vocabulary=vocabulary,
        masker=masker,
        device=device,
        non_blocking=False,
        metadata_dropout_probability=0.30,
        metadata_dropout_seed=23,
    )
    optimizer = AdamW(
        formal_optimizer_parameter_groups(
            built.model,
            sat3d_last_stage_lr=1.0e-4,
            acquisition_lr=2.0e-4,
            model_lr=3.0e-4,
            weight_decay=0.0,
        )
    )
    backend = TorchFormalTrainingBackend(
        model=built.model,
        optimizer=optimizer,
        preparer=preparer,
        ema_momentum=CosineEMAMomentum(
            initial=0.90,
            final=0.99,
            total_updates=1,
        ),
        precision="fp32",
        gradient_clip_norm=1.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(101)
    packed: BatchPayload = {
        "anchor_image": torch.randn(2, 1, 8, 8, 8, generator=generator),
        "anchor_support": torch.ones(2, 8, 8, 8, dtype=torch.bool),
        "anchor_uid": ["tiny-mri", "tiny-pet"],
        "anchor_metadata": records,
        "coverage_id": torch.tensor([1, 1], dtype=torch.long),
        "view_id": torch.tensor([0, 0], dtype=torch.long),
        "relation_type": ["same_observation", "same_observation"],
        "relation_uid": ["tiny-mri", "tiny-pet"],
        "temporal_delta_days": [None, None],
        "companion_image": torch.empty(0, 1, 8, 8, 8),
        "companion_support": torch.empty(0, 8, 8, 8, dtype=torch.bool),
        "companion_batch_index": torch.empty(0, dtype=torch.long),
        "companion_uid": [],
        "companion_metadata": [],
    }
    result = backend.train_accumulated_step(
        (
            Microbatch(
                payload=packed,
                anchor_count=2,
                next_anchor_index=2,
                loader_wait_seconds=0.0,
            ),
        ),
        coverage=1,
        relation_mixture=FormalTrainingSchedule().relation_mixture(1),
        expected_global_batch_anchors=16,
    )
    summary: dict[str, Any] = {
        "valid_for_scientific_results": False,
        "purpose": "low_resource_contract_smoke_only",
        "device": str(torch.device(device)),
        "anchors_processed": result.anchors_processed,
        "optimizer_steps": backend.ema.updates,
        "loss_total": result.loss_total,
        "loss_components": dict(result.loss_components),
        "modality_losses": dict(result.modality_losses),
        "relation_losses": dict(result.relation_losses),
        "grad_norm": result.grad_norm,
    }
    ensure_finite_metrics(summary, context="tiny synthetic smoke")
    output_path = work_path / "synthetic_smoke.json"
    encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=work_path
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**summary, "artifact": str(output_path)}


__all__ = [
    "CosineEMAMomentum",
    "DeterministicPhysicalBlockMasker",
    "EmbeddedEMAStateProxy",
    "FormalBatchPreparer",
    "FormalCoverageDataSource",
    "FormalModelBuildResult",
    "FormalModelBuildSpec",
    "FormalTrainingComponents",
    "PreparedFormalBatch",
    "RELATION_CODES",
    "TokenMaskPlan",
    "TorchFormalTrainingBackend",
    "TrainingLoopResult",
    "acquisition_condition_record",
    "build_acquisition_vocabulary",
    "build_formal_model",
    "build_real_training_components",
    "build_tiny_synthetic_formal_model",
    "build_warmup_cosine_scheduler",
    "encode_condition_batch",
    "formal_optimizer_parameter_groups",
    "run_full_training",
    "run_final_test_once",
    "run_synthetic_smoke",
]
