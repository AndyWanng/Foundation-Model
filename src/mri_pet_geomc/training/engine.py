"""Main multi-observation GeoMC training, validation and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import get
from ..data.observations import (
    ObservationVolume,
    ObservationRelationSampler,
    ObservationTables,
    ObservationVolumeCache,
    RelationExample,
)
from ..evaluation.fixed_reference import (
    WeightedSimilarityTransform,
    subject_balanced_weights,
)
from ..field.geometry_embedding import GeometryEmbedding
from ..geometry.fem import FEMGeometry
from ..geometry.frame import ParsevalResolventFrame
from ..geometry.interpolation import (
    FixedSpatialInterpolator,
    InterpolationPlan,
    token_centers_from_reference,
)
from ..model.foundation import (
    ObservationBatch,
    FoundationOutput,
    GeoMCFoundationModel,
    cosine_ema_momentum,
)
from ..model.geomc import GeoMCCore
from ..model.trainable_sat3d import TrainableSAT3DEncoder
from ..utils import EventLogger, atomic_write_json, digest_object, sha256_file
from .checkpoint import (
    bind_best_checkpoint,
    checkpoint_identity,
    load_checkpoint,
    materialize_checkpoint,
    save_checkpoint,
    validate_best_checkpoint_binding,
)
from .jepa import (
    entropy_effective_rank,
    field_representation_diagnostics,
    masked_huber_objective,
    representation_diagnostics,
)


MODALITY_TO_ID = {"mri": 0, "pet": 1}


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _hash_seed(*values: object) -> int:
    return int(digest_object([str(value) for value in values])[:16], 16) % (2**63 - 1)


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _token_field(field: Tensor, grid: tuple[int, int, int]) -> Tensor:
    if field.ndim != 4 or field.shape[0] != 1:
        raise ValueError("Voxel field must be [1,D,H,W]")
    return F.adaptive_avg_pool3d(field.unsqueeze(0).float(), grid).flatten(1).squeeze(0)


def _token_visibility(
    visible_voxels: Tensor,
    support_voxels: Tensor,
    grid: tuple[int, int, int],
) -> tuple[Tensor, Tensor]:
    support = _token_field(support_voxels, grid).clamp(0.0, 1.0)
    visible_mass = _token_field(visible_voxels, grid).clamp(0.0, 1.0)
    ratio = torch.where(support > 1.0e-6, visible_mass / support.clamp_min(1.0e-6), 0.0)
    return support, ratio.clamp(0.0, 1.0)


def _upsample_token_mask(mask: Tensor, shape: tuple[int, int, int]) -> Tensor:
    count = int(mask.numel())
    grid = round(count ** (1.0 / 3.0))
    if grid**3 != count:
        raise ValueError("Token mask must describe a cubic grid")
    value = mask.reshape(1, 1, grid, grid, grid).float()
    return F.interpolate(value, size=shape, mode="nearest").squeeze(0)


def _normalize_visible_observation(
    observation_volume: ObservationVolume,
    visibility: Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> Tensor:
    """Mask before encoding and recompute normalization from visible voxels."""

    visible = (visibility > 0.5) & (observation_volume.brain_mask > 0.5)
    if int(visible.sum()) < 2:
        raise RuntimeError("Observation mask leaves fewer than two visible brain voxels")
    raw = observation_volume.volume * float(observation_volume.normalization_std) + float(
        observation_volume.normalization_mean
    )
    values = raw[visible].double()
    mean = values.mean()
    standard_deviation = values.std(unbiased=False)
    if not torch.isfinite(standard_deviation) or float(standard_deviation) <= epsilon:
        raise RuntimeError("Visible-voxel observation normalization is degenerate")
    result = torch.zeros_like(observation_volume.volume)
    result[visible] = ((raw[visible].double() - mean) / standard_deviation).float()
    return result


def _select_query_mask(candidates: Tensor, *, count: int, seed: int) -> Tensor:
    available = torch.nonzero(candidates, as_tuple=False).flatten()
    if available.numel() < 2:
        raise RuntimeError("Fewer than two valid target-query tokens")
    chosen_count = min(max(2, int(count)), int(available.numel()))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(available.numel(), generator=generator)[:chosen_count]
    result = torch.zeros_like(candidates, dtype=torch.bool)
    result[available[order]] = True
    return result


def _observation_metadata(observation_volume: ObservationVolume, dimension: int) -> Tensor:
    """Build observation metadata without target values or fabricated fields."""

    if int(dimension) < 4:
        raise ValueError("model.metadata_dim must be at least four")
    result = torch.zeros(int(dimension), dtype=torch.float32)
    result[MODALITY_TO_ID[observation_volume.observation.modality]] = 1.0
    result[2] = float(observation_volume.observation.reliability)
    acquisition_available = observation_volume.observation.metadata.get(
        "acquisition_metadata_available", False
    )
    if not isinstance(acquisition_available, bool):
        raise ValueError("acquisition_metadata_available must be an explicit boolean")
    result[3] = float(acquisition_available)
    return result


def _prepare_example_observation(
    observation_volume: ObservationVolume,
    visible_tokens: Tensor,
    *,
    grid: tuple[int, int, int],
    metadata_dim: int,
) -> "PreparedObservation":
    voxel_visibility = _upsample_token_mask(
        visible_tokens, tuple(int(value) for value in observation_volume.volume.shape[1:])
    )
    voxel_visibility = voxel_visibility * observation_volume.brain_mask
    volume = _normalize_visible_observation(observation_volume, voxel_visibility)
    token_support, token_visible = _token_visibility(
        voxel_visibility, observation_volume.brain_mask, grid
    )
    reliability = _token_field(observation_volume.reliability, grid)
    reliability = torch.where(
        token_support > 1.0e-6,
        reliability / token_support.clamp_min(1.0e-6),
        0.0,
    ).clamp(0.0, 1.0)
    return PreparedObservation(
        volume=volume.unsqueeze(0),
        token_coverage=token_visible.unsqueeze(0),
        token_reliability=reliability.unsqueeze(0),
        metadata=_observation_metadata(observation_volume, metadata_dim),
        observation_uid=observation_volume.observation.observation_uid,
    )


@dataclass(frozen=True)
class PreparedObservation:
    volume: Tensor
    token_coverage: Tensor
    token_reliability: Tensor
    metadata: Tensor
    observation_uid: str


@dataclass(frozen=True)
class PreparedExample:
    observations: tuple[PreparedObservation, ...]
    target_volume: Tensor
    target_metadata: Tensor
    target_support: Tensor
    query_mask: Tensor
    query_weight: Tensor
    example: RelationExample
    target_observation_included: bool


def prepare_example(
    cache: ObservationVolumeCache,
    example: RelationExample,
    *,
    grid: tuple[int, int, int],
    hidden_fraction: float,
    query_tokens: int,
    include_target_observation_probability: float,
    metadata_dim: int,
    seed: int,
) -> PreparedExample:
    source = cache.get(example.source_observation_uid)
    target = cache.get(example.target_observation_uid)
    if source.volume.shape != target.volume.shape:
        raise RuntimeError("Source and target model volumes have different shapes")
    source_support, _ = _token_visibility(
        source.brain_mask, source.brain_mask, grid
    )
    target_support, _ = _token_visibility(
        target.brain_mask, target.brain_mask, grid
    )
    source_candidates = source_support > 0.05
    target_candidates = target_support > 0.05
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    observations: list[PreparedObservation] = []
    target_observation_included = False

    if example.is_cross_modal:
        query_mask = _select_query_mask(
            target_candidates,
            count=query_tokens,
            seed=_hash_seed(seed, "cross-query"),
        )
        observations.append(
            _prepare_example_observation(
                source,
                source_candidates,
                grid=grid,
                metadata_dim=metadata_dim,
            )
        )
        include_target = bool(
            torch.rand((), generator=generator).item()
            < float(include_target_observation_probability)
        )
        target_visible = target_candidates & ~query_mask
        if include_target and int(target_visible.sum()) >= 2:
            observations.append(
                _prepare_example_observation(
                    target,
                    target_visible,
                    grid=grid,
                    metadata_dim=metadata_dim,
                )
            )
            target_observation_included = True
    else:
        supported = torch.nonzero(source_candidates, as_tuple=False).flatten()
        if supported.numel() < 3:
            raise RuntimeError("Intra-modal masking needs at least three supported tokens")
        hidden_count = min(
            max(2, int(round(float(hidden_fraction) * supported.numel()))),
            int(supported.numel()) - 1,
        )
        hidden = supported[
            torch.randperm(supported.numel(), generator=generator)[:hidden_count]
        ]
        visible_tokens = source_candidates.clone()
        visible_tokens[hidden] = False
        query_candidates = torch.zeros_like(source_candidates, dtype=torch.bool)
        query_candidates[hidden] = target_candidates[hidden]
        query_mask = _select_query_mask(
            query_candidates,
            count=query_tokens,
            seed=_hash_seed(seed, "within-query"),
        )
        observations.append(
            _prepare_example_observation(
                source,
                visible_tokens,
                grid=grid,
                metadata_dim=metadata_dim,
            )
        )

    target_reliability = _token_field(target.reliability, grid)
    target_reliability = torch.where(
        target_support > 1.0e-6,
        target_reliability / target_support.clamp_min(1.0e-6),
        0.0,
    ).clamp(0.0, 1.0)
    if example.is_cross_modal:
        target_reliability = target_reliability * float(example.link_reliability)
    return PreparedExample(
        observations=tuple(observations),
        target_volume=target.volume.unsqueeze(0),
        target_metadata=_observation_metadata(target, metadata_dim),
        target_support=target_support.unsqueeze(0),
        query_mask=query_mask.unsqueeze(0),
        query_weight=target_reliability.unsqueeze(0),
        example=example,
        target_observation_included=target_observation_included,
    )


@dataclass(frozen=True)
class ModelGeometry:
    geometry: FEMGeometry
    token_xyz_mm: Tensor
    token_to_node: FixedSpatialInterpolator
    node_to_token: FixedSpatialInterpolator
    grid_shape: tuple[int, int, int]
    reference_shape: tuple[int, int, int]
    contract_digest: str


def build_model_geometry(
    geometry: FEMGeometry,
    *,
    reference_shape: Sequence[int],
    reference_affine: np.ndarray | Tensor,
    grid_shape: Sequence[int],
    neighbors: int,
    sigma_mm: float,
) -> ModelGeometry:
    geometry.validate()
    shape = tuple(int(value) for value in reference_shape)
    grid = tuple(int(value) for value in grid_shape)
    if len(shape) != 3 or len(grid) != 3:
        raise ValueError("reference_shape and grid_shape must be three-dimensional")
    token_xyz = token_centers_from_reference(shape, reference_affine, grid_shape=grid)
    affine = np.asarray(reference_affine, dtype=np.float64)
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    cell_voxels = math.prod(size / cells for size, cells in zip(shape, grid, strict=True))
    token_volume = torch.full(
        (token_xyz.shape[0],), voxel_volume * cell_voxels, dtype=torch.float64
    )
    forward_plan = InterpolationPlan.from_coordinates(
        token_xyz,
        geometry.node_xyz_mm,
        neighbors=int(neighbors),
        sigma_mm=float(sigma_mm),
        source_cell_volume=token_volume,
    )
    reverse_plan = InterpolationPlan.from_coordinates(
        geometry.node_xyz_mm,
        token_xyz,
        neighbors=int(neighbors),
        sigma_mm=float(sigma_mm),
        source_cell_volume=geometry.mass,
    )
    contract = {
        "geometry_hash": geometry.metadata.get("geometry_hash"),
        "reference_shape": list(shape),
        "reference_affine": affine.tolist(),
        "grid_shape": list(grid),
        "neighbors": int(neighbors),
        "sigma_mm": float(sigma_mm),
        "token_count": int(token_xyz.shape[0]),
        "node_count": geometry.node_count,
    }
    return ModelGeometry(
        geometry=geometry,
        token_xyz_mm=token_xyz,
        token_to_node=FixedSpatialInterpolator(forward_plan),
        node_to_token=FixedSpatialInterpolator(reverse_plan),
        grid_shape=grid,
        reference_shape=shape,
        contract_digest=digest_object(contract),
    )


def build_encoder(
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    *,
    seed: int,
) -> TrainableSAT3DEncoder:
    backend = str(get(config, "resources.encoder_backend", "tiny"))
    trainability = str(
        experiment.get("trainability", get(config, "sat3d.trainability", "all"))
    )
    initialization = str(
        experiment.get("initialization", get(config, "sat3d.initialization", "pretrained"))
    )
    if backend == "tiny":
        grid = tuple(int(value) for value in get(config, "geometry.token_grid", [2, 2, 2]))
        if len(set(grid)) != 1:
            raise ValueError("Tiny backend requires a cubic token grid")
        return TrainableSAT3DEncoder.tiny(
            trainability=trainability,  # type: ignore[arg-type]
            input_size=int(get(config, "data.synthetic_input_size", 16)),
            hidden_channels=int(get(config, "sat3d.tiny_hidden_channels", 8)),
            output_channels=int(get(config, "sat3d.tiny_output_channels", 16)),
            output_grid=grid[0],
            seed=seed,
        )
    if backend != "sat3d":
        raise ValueError(f"Unknown encoder backend: {backend}")
    source_sha = str(get(config, "sat3d.expected_source_tree_sha256", "")).strip() or None
    if initialization == "scratch":
        return TrainableSAT3DEncoder.from_sat3d_scratch(
            get(config, "paths.sat3d_code_root"),
            trainability=trainability,  # type: ignore[arg-type]
            seed=seed,
            expected_source_tree_sha256=source_sha,
        )
    if initialization != "pretrained":
        raise ValueError(f"Unknown SAT3D initialization: {initialization}")
    return TrainableSAT3DEncoder.from_sat3d(
        get(config, "paths.sat3d_code_root"),
        get(config, "paths.sat3d_checkpoint"),
        trainability=trainability,  # type: ignore[arg-type]
        expected_checkpoint_sha256=str(get(config, "sat3d.expected_checkpoint_sha256")),
        expected_source_tree_sha256=source_sha,
    )


def build_foundation_model(
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    field_geometry: ModelGeometry,
    *,
    seed: int,
) -> GeoMCFoundationModel:
    _seed_everything(seed)
    encoder = build_encoder(config, experiment, seed=seed)
    frame = ParsevalResolventFrame(
        field_geometry.geometry.eigenvalues,
        field_geometry.geometry.eigenvectors,
        field_geometry.geometry.mass,
        radii=tuple(float(value) for value in get(config, "geometry.resolvent_radii_mm")),
        normalize_spectrum=bool(get(config, "geometry.normalize_spectrum", False)),
        factorization="parseval_sqrt",
    )
    core = GeoMCCore(
        frame,
        int(get(config, "model.latent_dim", 96)),
        num_blocks=int(get(config, "model.geomc_blocks", 3)),
        hidden_dim=int(get(config, "model.geomc_hidden_dim", 192)),
        scale_embedding_dim=int(get(config, "model.geomc_scale_embedding_dim", 16)),
        residual_scale=float(get(config, "model.geomc_residual_scale", 0.1)),
    )
    _seed_everything(_hash_seed(seed, "shared_geometry_embedding"))
    geometry_embedding = GeometryEmbedding(
        core.frame,
        int(get(config, "model.latent_dim", 96)),
        node_xyz_mm=field_geometry.geometry.node_xyz_mm,
        feature_type=str(get(config, "geometry_embedding.type", "spectral")),  # type: ignore[arg-type]
        hidden_dim=int(get(config, "geometry_embedding.hidden_dim", 64)),
        initial_scale=float(get(config, "geometry_embedding.initial_scale", 0.05)),
        trainable_scale=bool(get(config, "geometry_embedding.trainable_scale", True)),
    )
    _seed_everything(_hash_seed(seed, "shared_foundation_modules"))
    return GeoMCFoundationModel(
        online_encoder=encoder,
        encoder_output_dim=int(encoder.output_shape[0]),
        latent_dim=int(get(config, "model.latent_dim", 96)),
        metadata_dim=int(get(config, "model.metadata_dim", 4)),
        token_to_node=field_geometry.token_to_node,
        node_to_token=field_geometry.node_to_token,
        core=core,
        geometry_embedding=geometry_embedding,
    )


def _make_sampler(
    tables: ObservationTables,
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    *,
    split: str,
    seed: int,
    use_configured_pairing: bool | None = None,
) -> ObservationRelationSampler:
    configured = dict(get(config, "training.relation_sampling_weights", {}))
    relation_weights = {
        str(name).replace("_to_", "->"): float(value)
        for name, value in configured.items()
    }
    apply_configured_pairing = (
        split == "train"
        if use_configured_pairing is None
        else bool(use_configured_pairing)
    )
    configured_pairing_mode = str(experiment.get("pairing_mode", "verified"))
    return ObservationRelationSampler(
        tables,
        split,
        verified_pair_retention_fraction=(
            float(experiment.get("verified_pair_retention_fraction", 1.0))
            if apply_configured_pairing
            else 1.0
        ),
        pairing_mode=(
            configured_pairing_mode if apply_configured_pairing else "verified"
        ),
        relation_weights=relation_weights or None,
        allowed_relations=experiment.get("allowed_relations"),
        retained_subject_order=(
            get(config, "data.frozen_verified_pair_retention_fraction_subject_order", None)
            if apply_configured_pairing and split == "train"
            else None
        ),
        relation_cycle_length=int(get(config, "training.relation_cycle_length", 20)),
        seed=int(seed),
    )


def _move_observations(
    prepared: PreparedExample,
    device: torch.device,
) -> tuple[ObservationBatch, ...]:
    return tuple(
        ObservationBatch(
            volume=item.volume.to(device, non_blocking=True),
            token_coverage=item.token_coverage.to(device, non_blocking=True),
            token_reliability=item.token_reliability.to(device, non_blocking=True),
            metadata=item.metadata.to(device, non_blocking=True),
            observation_uid=item.observation_uid,
        )
        for item in prepared.observations
    )


def _forward_prepared(
    model: GeoMCFoundationModel,
    prepared: PreparedExample,
    *,
    device: torch.device,
    precision: str,
    collect_diagnostics: bool,
):
    with _autocast(device, precision):
        output = model(
            _move_observations(prepared, device),
            prepared.target_volume.to(device, non_blocking=True),
            target_metadata=prepared.target_metadata.to(device, non_blocking=True),
            query_mask=prepared.query_mask.to(device, non_blocking=True),
            target_support=prepared.target_support.to(device, non_blocking=True),
            return_diagnostics=collect_diagnostics,
        )
        objective = masked_huber_objective(
            output.predicted_latents,
            output.target_latents,
            output.query_mask,
            query_weight=prepared.query_weight.to(device, non_blocking=True),
            beta=1.0,
        )
    return output, objective


def _validation_subset(
    examples: Sequence[RelationExample], maximum_examples: int
) -> tuple[RelationExample, ...]:
    ordered = tuple(
        sorted(
            examples,
            key=lambda item: (
                item.relation,
                item.subject_id,
                item.target_subject_id,
                item.source_observation_uid,
            ),
        )
    )
    limit = int(maximum_examples)
    if limit < 1:
        raise ValueError("maximum_examples must be positive")
    if len(ordered) <= limit:
        return ordered
    groups: dict[str, list[RelationExample]] = {}
    for item in ordered:
        groups.setdefault(item.relation, []).append(item)
    selected: list[RelationExample] = []
    index = 0
    names = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for name in names:
            if index < len(groups[name]) and len(selected) < limit:
                selected.append(groups[name][index])
                progressed = True
        if not progressed:
            break
        index += 1
    return tuple(selected)


@torch.no_grad()
def validate_model(
    model: GeoMCFoundationModel,
    tables: ObservationTables,
    cache: ObservationVolumeCache,
    field_geometry: ModelGeometry,
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
    precision: str,
    maximum_examples: int,
    views_per_example: int = 1,
    split: str = "validation",
    pairing_policy: str = "verified",
) -> dict[str, Any]:
    """Evaluate the main multi-observation model subject first."""

    was_training = model.training
    model.eval()
    if pairing_policy not in {"verified", "configured"}:
        raise ValueError("pairing_policy must be verified or configured")
    sampler = _make_sampler(
        tables,
        config,
        experiment,
        split=split,
        seed=seed + 9817,
        use_configured_pairing=pairing_policy == "configured",
    )
    view_count = max(1, int(views_per_example))
    examples = _validation_subset(
        sampler.supported_examples,
        max(1, int(maximum_examples) // view_count),
    )
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        for view in range(view_count):
            paired_views = example.is_cross_modal and view_count >= 2
            target_probability = (
                float(view % 2)
                if paired_views
                else float(
                    get(config, "inputs.include_target_observation_probability", 0.5)
                )
            )
            prepared = prepare_example(
                cache,
                example,
                grid=field_geometry.grid_shape,
                hidden_fraction=float(get(config, "masking.hidden_fraction", 0.30)),
                query_tokens=int(get(config, "masking.query_tokens", 128)),
                include_target_observation_probability=target_probability,
                metadata_dim=int(get(config, "model.metadata_dim", 4)),
                seed=_hash_seed(
                    seed,
                    split,
                    index,
                    0 if paired_views else view,
                    example.source_observation_uid,
                ),
            )
            output, objective = _forward_prepared(
                model,
                prepared,
                device=device,
                precision=precision,
                collect_diagnostics=True,
            )
            core = output.core_diagnostics
            if core is None:
                raise RuntimeError("Validation requested missing GeoMC diagnostics")
            blocks = [item.as_dict() for item in core.blocks]

            def block_mean(name: str) -> float:
                return float(np.mean([float(item[name]) for item in blocks] or [0.0]))

            def block_max(name: str) -> float:
                return float(max([float(item[name]) for item in blocks] or [0.0]))

            scale_rows = [
                [float(value) for value in item["scale_energy_fractions"]]
                for item in blocks
                if list(item.get("scale_energy_fractions") or [])
            ]
            scale_mean = (
                np.mean(np.asarray(scale_rows, dtype=np.float64), axis=0).tolist()
                if scale_rows
                else []
            )
            centered_rows = [
                [float(value) for value in item["centered_scale_energy_fractions"]]
                for item in blocks
                if list(item.get("centered_scale_energy_fractions") or [])
            ]
            centered_mean = (
                np.mean(np.asarray(centered_rows, dtype=np.float64), axis=0).tolist()
                if centered_rows
                else []
            )
            decoder_input_tokens, _ = model.query_decoder.node_to_token(output.latent_field)
            fem_mass = model.core.frame.mass.to(device=device, dtype=torch.float32)
            node_support_mass = fem_mass[None, :] * (
                output.node_feature_field.weight_sum > model.core.frame.eps
            ).to(torch.float32)
            representation = representation_diagnostics(
                output.predicted_latents,
                output.target_latents,
                output.query_mask,
                query_weight=prepared.query_weight.to(device),
            )
            node_feature_representation = field_representation_diagnostics(
                output.node_feature_field.features, point_weight=node_support_mass
            )
            latent_representation = field_representation_diagnostics(
                output.latent_field, point_weight=fem_mass
            )
            query_representation = field_representation_diagnostics(
                decoder_input_tokens,
                point_weight=(
                    prepared.query_weight.to(device=device, dtype=torch.float32)
                    * output.query_mask.to(torch.float32)
                ),
            )
            input_configuration = (
                "within_modality_masked"
                if not example.is_cross_modal
                else (
                    "source_plus_masked_target"
                    if prepared.target_observation_included
                    else "source_only"
                )
            )
            row: dict[str, Any] = {
                "subject_id": example.subject_id,
                "target_subject_id": example.target_subject_id,
                "relation": example.relation,
                "view_index": view,
                "is_cross_modal": example.is_cross_modal,
                "input_configuration": input_configuration,
                "target_observation_included": prepared.target_observation_included,
                "query_mask_digest": digest_object(
                    prepared.query_mask.detach().cpu().tolist()
                ),
                "loss": float(objective.loss.detach().cpu()),

                **representation,
                **{f"node_feature_{key}": value for key, value in node_feature_representation.items()},
                **{f"latent_field_{key}": value for key, value in latent_representation.items()},
                **{f"decoder_input_{key}": value for key, value in query_representation.items()},
                "node_feature_coverage": output.node_feature_field.diagnostics()["coverage_fraction"],
                "core_output_rms": core.output_rms,
                "core_output_channel_variance_mean": core.output_channel_variance_mean,
                "core_output_hard_threshold_rank": float(core.output_hard_threshold_rank),
                "decoder_input_all_token_entropy_effective_rank": float(
                    entropy_effective_rank(decoder_input_tokens).detach().cpu()
                ),
                "geometry_embedding_relative_rms": core.geometry_embedding_relative_rms,
                "mean_routing_matrix_abs": block_mean("routing_matrix_abs_mean"),
                "mean_cross_scale_routing_abs": block_mean(
                    "cross_scale_routing_abs_mean"
                ),
                "mean_routing_delta_abs": block_mean("routing_delta_abs_mean"),
                "mean_routing_spatial_std": block_mean(
                    "routing_spatial_std"
                ),
                "decoder_output_token_rms": float(
                    output.decoder_diagnostics.output_token_rms
                    if output.decoder_diagnostics
                    else 0.0
                ),
                "maximum_frame_reconstruction_relative_l2": block_max(
                    "reconstruction_relative_l2"
                ),
                "complement_relative_energy": (
                    float(scale_mean[0]) if scale_mean else 0.0
                ),
                "parseval_additive_energy_closure_relative_error": block_max(
                    "additive_energy_closure_relative_error"
                ),
                "centered_additive_energy_closure_relative_error": block_max(
                    "centered_additive_energy_closure_relative_error"
                ),
            }
            for scale_index, value in enumerate(scale_mean):
                row[f"scale_energy_fraction_{scale_index}"] = float(value)
            for scale_index, value in enumerate(centered_mean):
                row[f"centered_scale_energy_fraction_{scale_index}"] = float(value)
            for block_index, block in enumerate(blocks):
                for name in (
                    "input_entropy_effective_rank",
                    "output_entropy_effective_rank",
                    "input_centered_variance",
                    "output_centered_variance",
                    "input_spatial_mean_energy_fraction",
                    "output_spatial_mean_energy_fraction",
                ):
                    row[f"block_{block_index}_{name}"] = float(block[name])
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No supported validation examples for split={split}")
    nonnumeric = {
        "subject_id",
        "target_subject_id",
        "relation",
        "view_index",
        "is_cross_modal",
        "input_configuration",
        "target_observation_included",
    }
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in nonnumeric and isinstance(value, (int, float))
    ]
    subjects = sorted({str(row["subject_id"]) for row in rows})
    subject_rows: list[dict[str, Any]] = []
    for subject in subjects:
        selected = [row for row in rows if row["subject_id"] == subject]
        subject_row: dict[str, Any] = {
            "subject_id": subject,
            **{
                key: float(np.mean([float(row[key]) for row in selected]))
                for key in numeric_keys
            },
        }
        for kind, name in (
            ("source_only", "cross_modal_source_only_loss"),
            ("source_plus_masked_target", "cross_modal_fusion_loss"),
            ("within_modality_masked", "intra_modal_loss"),
        ):
            values = [
                float(row["loss"])
                for row in selected
                if row["input_configuration"] == kind
            ]
            if values:
                subject_row[name] = float(np.mean(values))
        if (
            "cross_modal_source_only_loss" in subject_row
            and "cross_modal_fusion_loss" in subject_row
        ):
            subject_row["cross_modal_additional_input_advantage"] = float(
                subject_row["cross_modal_source_only_loss"]
                - subject_row["cross_modal_fusion_loss"]
            )
        subject_rows.append(subject_row)

    summary = {
        key: float(np.mean([float(row[key]) for row in subject_rows]))
        for key in numeric_keys
    }
    summary.update(
        {
            "subject_count": len(subjects),
            "example_count": len(rows),
            "supported_example_count": len(sampler.supported_examples),
            "views_per_example": view_count,
            "complete_supported_census": len(examples) == len(sampler.supported_examples),
            "subject_rows": subject_rows,
            "view_rows": rows,
            "pairing_policy": pairing_policy,
            "sampler": sampler.sampling_summary(),
        }
    )
    for name in (
        "cross_modal_source_only_loss",
        "cross_modal_fusion_loss",
        "cross_modal_additional_input_advantage",
        "intra_modal_loss",
    ):
        values = [float(row[name]) for row in subject_rows if name in row]
        if values:
            summary[name] = float(np.mean(values))
    model.train(was_training)
    return summary


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup: int,
    total: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1.0e-4, float(step + 1) / float(warmup))
        progress = (step - warmup) / float(max(1, total - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _training_contract(
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    tables: ObservationTables,
    field_geometry: ModelGeometry,
    sampler: ObservationRelationSampler,
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": str(get(config, "project.method")),
        "experiment": dict(experiment),
        "verified_pair_retention_fraction": float(experiment.get("verified_pair_retention_fraction", 1.0)),
        "pairing_mode": str(experiment.get("pairing_mode", "verified")),
        "seed": int(seed),
        "model": dict(get(config, "model", {})),
        "sat3d": dict(get(config, "sat3d", {})),
        "geometry": dict(get(config, "geometry", {})),
        "geometry_embedding": dict(get(config, "geometry_embedding", {})),
        "masking": dict(get(config, "masking", {})),
        "inputs": dict(get(config, "inputs", {})),
        "training": {
            key: value
            for key, value in dict(get(config, "training", {})).items()
            if key not in {"validation_every", "checkpoint_every"}
        },
        "observation_digest": tables.contract_digest(),
        "sampler_digest": sampler.contract_digest(),
        "model_geometry_digest": field_geometry.contract_digest,
    }


def run_training(
    tables: ObservationTables,
    field_geometry: ModelGeometry,
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    *,
    seed: int,
    output_dir: str | Path,
    resume_checkpoint: str | Path | None = None,
    event_logger: EventLogger | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(get(config, "resources.device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    precision = str(get(config, "training.precision", "fp32"))
    run_seed = int(get(config, "project.seed", 20260806)) + 1009 * int(seed)
    _seed_everything(run_seed)
    model = build_foundation_model(
        config, experiment, field_geometry, seed=run_seed
    ).to(device)
    # Reset stochastic state after deterministic model construction. Resume loading
    # below restores its checkpointed RNG state when requested.
    _seed_everything(run_seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(
            encoder_learning_rate=float(get(config, "training.encoder_learning_rate")),
            model_learning_rate=float(get(config, "training.model_learning_rate")),
            weight_decay=float(get(config, "training.weight_decay")),
        )
    )
    maximum_updates = int(experiment["max_updates"])
    planned_total = int(experiment.get("planned_total_updates", maximum_updates))
    scheduler = _scheduler(
        optimizer,
        warmup=int(get(config, "training.warmup_updates", 0)),
        total=planned_total,
    )
    sampler = _make_sampler(
        tables, config, experiment, split="train", seed=run_seed
    )
    contract = _training_contract(
        config, experiment, tables, field_geometry, sampler, seed=seed
    )
    cache = ObservationVolumeCache(
        tables.observations, max_items=max(16, len(tables.observations))
    )
    accumulation = int(get(config, "training.gradient_accumulation", 1))
    if int(get(config, "training.microbatch_size", 1)) != 1:
        raise RuntimeError("Current online SAT3D engine requires physical batch one")
    validation_every = int(get(config, "training.validation_every", 100))
    checkpoint_every = int(get(config, "training.checkpoint_every", validation_every))
    validation_examples = int(get(config, "training.validation_max_examples", 20))
    history: list[dict[str, Any]] = []
    start_step = 0
    best_metric = float("inf")
    best_step = -1
    best_candidate = checkpoints / "best_candidate.pt"

    resume_source: Path | None = None
    if resume_checkpoint is not None:
        resume_source = Path(resume_checkpoint).resolve()
    elif (checkpoints / "resume.pt").is_file():
        resume_source = checkpoints / "resume.pt"
    if resume_source is not None:
        source_best = resume_source.parent / "best_candidate.pt"
        if source_best.is_file() and source_best.resolve() != best_candidate.resolve():
            materialize_checkpoint(source_best, best_candidate)
        payload = load_checkpoint(
            resume_source,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_contract=contract,
            restore_rng=True,
        )
        start_step = int(payload["step"])
        best_metric = float(payload["best_metric"])
        best_step = int(payload["best_step"])
        history = list(payload.get("history_tail") or [])
        if start_step > maximum_updates:
            raise RuntimeError(
                f"Resume step {start_step} exceeds requested maximum {maximum_updates}"
            )

    relation_counts = {name: 0 for name in sampler.active_relations}
    for completed_step in range(1, start_step + 1):
        for micro in range(accumulation):
            example = sampler.sample((completed_step - 1) * accumulation + micro, 1)[0]
            relation_counts[example.relation] += 1

    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, maximum_updates + 1):
        losses: list[float] = []

        target_observation_count = 0
        for micro in range(accumulation):
            example = sampler.sample((step - 1) * accumulation + micro, 1)[0]
            relation_counts[example.relation] += 1
            prepared = prepare_example(
                cache,
                example,
                grid=field_geometry.grid_shape,
                hidden_fraction=float(get(config, "masking.hidden_fraction", 0.30)),
                query_tokens=int(get(config, "masking.query_tokens", 128)),
                include_target_observation_probability=float(
                    get(config, "inputs.include_target_observation_probability", 0.5)
                ),
                metadata_dim=int(get(config, "model.metadata_dim", 4)),
                seed=_hash_seed(
                    run_seed, step, micro, example.source_observation_uid
                ),
            )
            target_observation_count += int(prepared.target_observation_included)
            _, objective = _forward_prepared(
                model,
                prepared,
                device=device,
                precision=precision,
                collect_diagnostics=False,
            )
            (objective.loss / accumulation).backward()
            losses.append(float(objective.loss.detach().cpu()))

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError("Missing or non-finite trainable gradients")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(get(config, "training.gradient_clip_norm", 1.0)),
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Gradient norm is non-finite")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        momentum = cosine_ema_momentum(
            step,
            planned_total,
            start=float(get(config, "training.ema_start", 0.996)),
            end=float(get(config, "training.ema_end", 0.9999)),
        )
        model.update_target(momentum)
        row: dict[str, Any] = {
            "step": step,
            "train_loss": float(np.mean(losses)),

            "gradient_norm": float(gradient_norm.detach().cpu()),
            "ema_momentum": momentum,
            "target_observation_fraction": target_observation_count / float(accumulation),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        }
        history.append(row)
        if event_logger is not None and (step == 1 or step % max(1, validation_every) == 0):
            event_logger.log(
                "geomc_training_progress",
                experiment=str(experiment["id"]),
                seed=int(seed),
                step=step,
                train_loss=row["train_loss"],
                gradient_norm=row["gradient_norm"],
            )

        should_validate = step == maximum_updates or step % validation_every == 0
        if should_validate:
            validation = validate_model(
                model,
                tables,
                cache,
                field_geometry,
                config,
                experiment,
                seed=run_seed,
                device=device,
                precision=precision,
                maximum_examples=validation_examples,
                views_per_example=int(get(config, "training.validation_views", 1)),
                pairing_policy="configured",
            )
            row["validation"] = {
                key: value
                for key, value in validation.items()
                if key not in {"subject_rows", "view_rows"}
            }
            metric = float(validation["loss"])
            if metric < best_metric:
                best_metric = metric
                best_step = step
                save_checkpoint(
                    best_candidate,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    step=step,
                    best_metric=best_metric,
                    best_step=step,
                    contract=contract,
                    history_tail=history,
                    run_state={"validation": validation},
                )
        should_checkpoint = step == maximum_updates or step % checkpoint_every == 0
        if should_checkpoint:
            if best_step < 1 or not best_candidate.is_file():
                # Validation must coincide with every checkpoint at least once.
                validation = validate_model(
                    model,
                    tables,
                    cache,
                    field_geometry,
                    config,
                    experiment,
                    seed=run_seed,
                    device=device,
                    precision=precision,
                    maximum_examples=validation_examples,
                    views_per_example=int(get(config, "training.validation_views", 1)),
                    pairing_policy="configured",
                )
                best_metric = float(validation["loss"])
                best_step = step
                save_checkpoint(
                    best_candidate,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    step=step,
                    best_metric=best_metric,
                    best_step=step,
                    contract=contract,
                    history_tail=history,
                    run_state={"validation": validation},
                )
            run_state = bind_best_checkpoint(
                {}, best_candidate, expected_contract=contract
            )
            save_checkpoint(
                checkpoints / "resume.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                step=step,
                best_metric=best_metric,
                best_step=best_step,
                contract=contract,
                history_tail=history,
                run_state=run_state,
            )

    if not (checkpoints / "resume.pt").is_file() or not best_candidate.is_file():
        raise RuntimeError("Training produced no resumable validation-bound checkpoint")
    resume_payload = torch.load(
        checkpoints / "resume.pt", map_location="cpu", weights_only=False
    )
    bound_candidate, best_identity = validate_best_checkpoint_binding(
        resume_payload["run_state"], checkpoints, expected_contract=contract
    )
    best_path = materialize_checkpoint(bound_candidate, checkpoints / "best.pt")
    final_path = materialize_checkpoint(checkpoints / "resume.pt", checkpoints / "final.pt")
    final_identity = checkpoint_identity(final_path, expected_contract=contract)
    if int(final_identity["step"]) != maximum_updates:
        raise RuntimeError(
            "The fixed-final checkpoint does not match the declared update budget: "
            f"{final_identity}"
        )
    best_payload = torch.load(bound_candidate, map_location="cpu", weights_only=False)
    best_validation = dict(best_payload.get("run_state", {}).get("validation") or {})
    summary = {
        "schema_version": 1,
        "experiment": dict(experiment),
        "seed": int(seed),
        "status": "completed",
        "start_step": start_step,
        "completed_step": maximum_updates,
        "best_step": best_step,
        "best_validation_loss": best_metric,
        "best_validation": best_validation,
        "training_contract": contract,
        "training_contract_digest": digest_object(contract),
        "sampler": sampler.sampling_summary(),
        "training_relation_counts": relation_counts,
        "model_specification": model.model_specification(),
        "encoder_provenance": model.online_encoder.provenance,
        "geometry_embedding_diagnostics": model.geometry_embedding.diagnostics().as_dict(),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_identity": best_identity,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": final_identity["checkpoint_sha256"],
        "final_identity": final_identity,
        "history_tail": history[-100:],
    }
    summary_path = atomic_write_json(output / "training_summary.json", summary)
    return {**summary, "summary_path": str(summary_path)}


@torch.no_grad()
def build_fixed_reference_targets(
    tables: ObservationTables,
    field_geometry: ModelGeometry,
    config: Mapping[str, Any],
    *,
    seed: int,
    splits: Sequence[str] = ("train", "test"),
) -> dict[str, Tensor]:
    """Encode the fixed target space used by main model evaluation.

    The neutral target encoder/projector is initialized before training. Its
    CPU-resident tokens define an immutable evaluation reference without keeping
    a second model resident in accelerator memory.
    """

    fixed_reference_name = str(get(config, "evaluation.fixed_reference", ""))
    if fixed_reference_name != "initial_target_encoder_v1":
        raise ValueError("Unsupported fixed reference contract")
    primary_identifier = str(
        get(config, "evaluation.primary_experiment_id", "geomc")
    )
    main_identifier = primary_identifier
    main = next(
        (
            dict(item)
            for item in get(config, "experiments", [])
            if str(item.get("id")) == main_identifier
        ),
        None,
    )
    if main is None:
        raise RuntimeError(
            f"Fixed reference requires main experiment {main_identifier!r}"
        )
    device = torch.device(str(get(config, "resources.device", "cpu")))
    precision = str(get(config, "training.precision", "fp32"))
    run_seed = int(get(config, "project.seed", 20260806)) + 1009 * int(seed)
    model = build_foundation_model(
        config, main, field_geometry, seed=run_seed
    ).to(device)
    model.eval()
    cache = ObservationVolumeCache(
        tables.observations, max_items=max(16, len(tables.observations))
    )
    accepted = {str(value) for value in splits}
    selected = sorted(
        item.observation_uid
        for item in tables.observations
        if tables.subject_splits.get(item.subject_id) in accepted
    )
    targets: dict[str, Tensor] = {}
    try:
        for uid in selected:
            observation_volume = cache.get(uid)
            with _autocast(device, precision):
                tokens, grid = model.encode_target(
                    observation_volume.volume.unsqueeze(0).to(device, non_blocking=True),
                    _observation_metadata(
                        observation_volume, int(get(config, "model.metadata_dim", 4))
                    ).to(device, non_blocking=True),
                )
            if tuple(grid) != tuple(field_geometry.grid_shape):
                raise RuntimeError("Fixed reference target grid differs from field grid")
            value = tokens.squeeze(0).detach().float().cpu().contiguous()
            if value.ndim != 2 or not torch.isfinite(value).all():
                raise RuntimeError(f"Invalid fixed reference target for {uid}")
            targets[uid] = value
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(targets) != len(selected):
        raise RuntimeError("Fixed reference cache is incomplete")
    return targets


@torch.no_grad()
def _collect_fixed_reference_rows(
    model: GeoMCFoundationModel,
    tables: ObservationTables,
    cache: ObservationVolumeCache,
    field_geometry: ModelGeometry,
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    reference_targets: Mapping[str, Tensor],
    *,
    split: str,
    seed: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    """Collect fixed query rows for train-only alignment or held-out score."""

    sampler = _make_sampler(
        tables,
        config,
        experiment,
        split=split,
        seed=seed + 9817,
        use_configured_pairing=False,
    )
    view_count = int(get(config, "evaluation.query_mask_views", 2))
    examples = _validation_subset(sampler.supported_examples, 1_000_000_000)
    predictions: list[Tensor] = []
    references: list[Tensor] = []
    raw_weights: list[Tensor] = []
    subject_ids: list[str] = []
    group_ids: list[str] = []
    relation_counts: dict[str, int] = {}
    was_training = model.training
    model.eval()
    for index, example in enumerate(examples):
        for view in range(view_count):
            target_observation_probability = (
                float(view % 2)
                if example.is_cross_modal and view_count >= 2
                else float(
                    get(config, "inputs.include_target_observation_probability", 0.5)
                )
            )
            prepared = prepare_example(
                cache,
                example,
                grid=field_geometry.grid_shape,
                hidden_fraction=float(get(config, "masking.hidden_fraction", 0.30)),
                query_tokens=int(get(config, "masking.query_tokens", 128)),
                include_target_observation_probability=target_observation_probability,
                metadata_dim=int(get(config, "model.metadata_dim", 4)),
                seed=_hash_seed(
                    seed,
                    split,
                    index,
                    view,
                    example.source_observation_uid,
                ),
            )
            output, _ = _forward_prepared(
                model,
                prepared,
                device=device,
                precision=precision,
                collect_diagnostics=False,
            )
            mask = output.query_mask[0].detach().cpu()
            reference = reference_targets.get(example.target_observation_uid)
            if reference is None:
                raise RuntimeError(
                    f"Fixed reference cache lacks {example.target_observation_uid}"
                )
            prediction = output.predicted_latents[0].detach().float().cpu()
            if prediction.shape != reference.shape or mask.shape != prediction.shape[:1]:
                raise RuntimeError("Candidate/fixed-reference token contracts differ")
            weight = prepared.query_weight[0].detach().float().cpu()[mask]
            if int(mask.sum()) < 2 or float(weight.sum()) <= 0:
                raise RuntimeError("Fixed-reference query group has insufficient weight")
            selected_prediction = prediction[mask]
            predictions.append(selected_prediction)
            references.append(reference[mask])
            raw_weights.append(weight)
            group = (
                f"{example.relation}|{example.source_observation_uid}|"
                f"{example.target_observation_uid}|view{view}"
            )
            subject_ids.extend([example.subject_id] * selected_prediction.shape[0])
            group_ids.extend([group] * selected_prediction.shape[0])
            relation_counts[example.relation] = relation_counts.get(example.relation, 0) + 1
    model.train(was_training)
    if not predictions:
        raise RuntimeError(f"No fixed-reference rows were collected for split={split}")
    return {
        "prediction": torch.cat(predictions, dim=0),
        "reference": torch.cat(references, dim=0),
        "raw_weight": torch.cat(raw_weights, dim=0),
        "subject_ids": tuple(subject_ids),
        "group_ids": tuple(group_ids),
        "group_count": len(predictions),
        "relation_group_counts": relation_counts,
    }

def _fixed_reference_subject_scores(
    collection: Mapping[str, Any],
    transform: WeightedSimilarityTransform,
) -> tuple[list[dict[str, float | str]], float, float]:
    prediction = torch.as_tensor(collection["prediction"]).float()
    reference = torch.as_tensor(collection["reference"]).float()
    raw_weight = torch.as_tensor(collection["raw_weight"]).float()
    subjects = tuple(str(value) for value in collection["subject_ids"])
    groups = tuple(str(value) for value in collection["group_ids"])
    aligned = transform.apply(prediction)
    aligned_element = F.smooth_l1_loss(
        aligned, reference, beta=1.0, reduction="none"
    ).mean(dim=-1)
    unaligned_element = F.smooth_l1_loss(
        prediction, reference, beta=1.0, reduction="none"
    ).mean(dim=-1)
    rows: list[dict[str, float | str]] = []
    for subject in sorted(set(subjects)):
        indices = [index for index, value in enumerate(subjects) if value == subject]
        index = torch.tensor(indices, dtype=torch.long)
        within = subject_balanced_weights(
            raw_weight[index],
            [subject] * len(indices),
            [groups[position] for position in indices],
        ).to(torch.float32)
        rows.append(
            {
                "subject_id": subject,
                "fixed_reference_aligned_huber_loss": float(
                    torch.sum(within * aligned_element[index])
                ),
                "fixed_reference_unaligned_huber_loss": float(
                    torch.sum(within * unaligned_element[index])
                ),
            }
        )
    aligned_mean = float(
        np.mean([float(row["fixed_reference_aligned_huber_loss"]) for row in rows])
    )
    unaligned_mean = float(
        np.mean([float(row["fixed_reference_unaligned_huber_loss"]) for row in rows])
    )
    return rows, aligned_mean, unaligned_mean



def _fixed_reference_digest(reference_targets: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for uid in sorted(reference_targets):
        value = torch.as_tensor(reference_targets[uid]).detach().float().cpu().contiguous()
        digest.update(uid.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


@torch.no_grad()
def evaluate_fixed_reference(
    model: GeoMCFoundationModel,
    tables: ObservationTables,
    cache: ObservationVolumeCache,
    field_geometry: ModelGeometry,
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    reference_targets: Mapping[str, Tensor],
    *,
    seed: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    if str(get(config, "evaluation.reference_alignment", "")) != (
        "global_subject_weighted_similarity_procrustes"
    ):
        raise ValueError("Unsupported fixed-reference alignment contract")
    if str(get(config, "evaluation.alignment_fit_split", "")) != "train":
        raise ValueError("Fixed-reference alignment must be fitted on train only")
    train = _collect_fixed_reference_rows(
        model,
        tables,
        cache,
        field_geometry,
        config,
        experiment,
        reference_targets,
        split="train",
        seed=seed + 314159,
        device=device,
        precision=precision,
    )
    train_weight = subject_balanced_weights(
        train["raw_weight"], train["subject_ids"], train["group_ids"]
    )
    transform = WeightedSimilarityTransform.fit(
        train["prediction"], train["reference"], train_weight
    )
    test = _collect_fixed_reference_rows(
        model,
        tables,
        cache,
        field_geometry,
        config,
        experiment,
        reference_targets,
        split="test",
        seed=seed + 271828,
        device=device,
        precision=precision,
    )
    subject_rows, aligned, unaligned = _fixed_reference_subject_scores(test, transform)
    transform_state = transform.state_dict()
    serializable_state = {
        key: (value.tolist() if isinstance(value, Tensor) else value)
        for key, value in transform_state.items()
    }
    return {
        "schema_version": 1,
        "fixed_reference": str(get(config, "evaluation.fixed_reference", "")),
        "reference_target_cache_sha256": _fixed_reference_digest(reference_targets),
        "fit_split": "train",
        "score_split": "test",
        "alignment": "global_subject_weighted_similarity_procrustes",
        "aggregate_by_subject": True,
        "fixed_reference_aligned_huber_loss": aligned,
        "fixed_reference_unaligned_huber_loss": unaligned,
        "subject_rows": subject_rows,
        "alignment_diagnostics": transform.diagnostics(),
        "alignment_state": serializable_state,
        "alignment_state_digest": digest_object(serializable_state),
        "alignment_fit_prediction_scope": "fixed_query_predictions",

        "train_group_count": int(train["group_count"]),
        "test_group_count": int(test["group_count"]),
        "train_relation_group_counts": train["relation_group_counts"],
        "test_relation_group_counts": test["relation_group_counts"],
    }


def evaluate_checkpoint(
    tables: ObservationTables,
    field_geometry: ModelGeometry,
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_path: str | Path,
    fixed_reference_targets: Mapping[str, Tensor] | None = None,
    split: str = "test",
    maximum_examples: int = 100000,
) -> dict[str, Any]:
    device = torch.device(str(get(config, "resources.device", "cpu")))
    precision = str(get(config, "training.precision", "fp32"))
    run_seed = int(get(config, "project.seed", 20260806)) + 1009 * int(seed)
    sampler = _make_sampler(
        tables, config, experiment, split="train", seed=run_seed
    )
    contract = _training_contract(
        config, experiment, tables, field_geometry, sampler, seed=seed
    )
    model = build_foundation_model(
        config, experiment, field_geometry, seed=run_seed
    ).to(device)
    load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        expected_contract=contract,
        restore_rng=False,
    )
    cache = ObservationVolumeCache(
        tables.observations, max_items=max(16, len(tables.observations))
    )
    ema_target_metrics = validate_model(
        model,
        tables,
        cache,
        field_geometry,
        config,
        experiment,
        seed=run_seed + 104729,
        device=device,
        precision=precision,
        maximum_examples=maximum_examples,
        views_per_example=int(get(config, "evaluation.query_mask_views", 2)),
        split=split,
        pairing_policy="verified",
    )
    result: dict[str, Any] = {
        "ema_target_metrics": ema_target_metrics,
    }
    if fixed_reference_targets is not None:
        result["fixed_reference"] = evaluate_fixed_reference(
            model,
            tables,
            cache,
            field_geometry,
            config,
            experiment,
            fixed_reference_targets,
            seed=run_seed,
            device=device,
            precision=precision,
        )
    return result


def run_small_sample_overfit_check(
    tables: ObservationTables,
    field_geometry: ModelGeometry,
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    train_subjects = sorted(
        subject for subject, split in tables.subject_splits.items() if split == "train"
    )[:3]
    validation_subjects = sorted(
        subject for subject, split in tables.subject_splits.items() if split == "validation"
    )[:1]
    selected = set(train_subjects + validation_subjects)
    if len(train_subjects) < 3 or not validation_subjects:
        raise RuntimeError("Small-sample overfit check requires 3 train and 1 validation subject")
    subset = ObservationTables(
        observations=tuple(item for item in tables.observations if item.subject_id in selected),
        links=tuple(item for item in tables.links if item.subject_id in selected),
        subject_splits={
            subject: split
            for subject, split in tables.subject_splits.items()
            if subject in selected
        },
    )
    primary_id = str(get(config, "evaluation.primary_experiment_id"))
    primary = next(
        (
            dict(item)
            for item in get(config, "experiments", [])
            if str(item.get("id")) == primary_id
        ),
        None,
    )
    if primary is None:
        raise RuntimeError("Small-sample overfit check cannot resolve the primary experiment")
    # Exercise the exact primary GeoMC contract. Reconstructing a partial
    # model here could silently change the core/decoder and make this
    # diagnostic test a different model from the one it claimed to test.
    experiment = json.loads(json.dumps(primary))
    experiment.update(
        {
            "id": "small_sample_overfit_check",
            "seeds": [0],
            "max_updates": int(get(config, "overfit.max_updates", 8)),
            "planned_total_updates": int(get(config, "overfit.max_updates", 8)),
        }
    )
    local_config = json.loads(json.dumps(config))
    local_config["data"].pop("frozen_verified_pair_retention_fraction_subject_order", None)
    local_config["training"]["gradient_accumulation"] = 1
    local_config["training"]["validation_every"] = int(experiment["max_updates"])
    local_config["training"]["checkpoint_every"] = int(experiment["max_updates"])
    local_config["training"]["validation_max_examples"] = 4
    result = run_training(
        subset,
        field_geometry,
        local_config,
        experiment,
        seed=0,
        output_dir=output_dir,
    )
    history = list(result.get("history_tail") or [])
    initial = float(history[0]["train_loss"])
    final = float(history[-1]["train_loss"])
    relative_drop = (initial - final) / max(abs(initial), 1.0e-8)
    initial_total = float(history[0]["train_loss"])
    final_total = float(history[-1]["train_loss"])
    threshold = float(
        get(config, "overfit.minimum_relative_drop_report_only", 0.0)
    )
    finite_drop = bool(math.isfinite(relative_drop))
    payload = {
        "schema_version": 2,
        "status": "completed",
        # Backwards-compatible diagnostic flag; never an execution gate.
        "passed": bool(finite_drop and relative_drop >= threshold),
        "initial_train_loss": initial,
        "final_train_loss": final,
        "loss_used_for_relative_drop": "train_loss",
        "initial_total_objective_loss": initial_total,
        "final_total_objective_loss": final_total,
        "relative_drop": relative_drop,
        "report_only_threshold": threshold,
        "meets_report_only_threshold": bool(
            finite_drop and relative_drop >= threshold
        ),
        "supports_scientific_claims": False,
        "findings_block_execution": False,
        "execution_action": "record_and_continue",
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "primary_experiment_id": primary_id,
        "exercised_experiment_contract": experiment,
        "training_summary": result["summary_path"],
    }
    atomic_write_json(Path(output_dir) / "small_sample_overfit_check_diagnostics.json", payload)
    return payload


__all__ = [
    "ModelGeometry",
    "PreparedExample",
    "PreparedObservation",
    "_forward_prepared",
    "_hash_seed",
    "_make_sampler",
    "_seed_everything",
    "build_fixed_reference_targets",
    "build_model_geometry",
    "build_foundation_model",
    "evaluate_checkpoint",
    "evaluate_fixed_reference",
    "run_small_sample_overfit_check",
    "prepare_example",
    "run_training",
    "validate_model",
]
