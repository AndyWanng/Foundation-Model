"""Configuration loading and validation for the formal workstation workflow."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


class FormalConfigError(ValueError):
    """Raised when a formal-run configuration violates a structural contract."""


_PATH_KEYS = {
    "metadata_root",
    "fomo_root",
    "adni_pet_root",
    "cache_root",
    "run_root",
    "sat3d_code_root",
    "sat3d_checkpoint",
    "source_reference",
    "source_reference_mask",
}


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand(item) for key, item in value.items()}
    return value


def _resolve_paths(config: MutableMapping[str, Any], *, project_root: Path) -> None:
    paths = config.get("paths")
    if not isinstance(paths, MutableMapping):
        raise FormalConfigError("paths must be a mapping")
    for key in _PATH_KEYS:
        raw = paths.get(key)
        if raw in (None, ""):
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = project_root / path
        paths[key] = str(path.resolve(strict=False))


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise FormalConfigError(f"{key} must be a mapping")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FormalConfigError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FormalConfigError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormalConfigError(f"{name} must be finite") from exc
    if isinstance(value, bool) or not math.isfinite(parsed):
        raise FormalConfigError(f"{name} must be finite")
    return parsed


def _positive_float(value: Any, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if not parsed > 0.0:
        raise FormalConfigError(f"{name} must be positive")
    return parsed


def validate_formal_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != 2:
        raise FormalConfigError("schema_version must be 2")
    project = _mapping(config, "project")
    if str(project.get("mode")) != "workstation_strict_unpaired_v1":
        raise FormalConfigError("project.mode must be workstation_strict_unpaired_v1")

    paths = _mapping(config, "paths")
    for key in (
        "metadata_root",
        "fomo_root",
        "adni_pet_root",
        "cache_root",
        "run_root",
        "sat3d_code_root",
        "sat3d_checkpoint",
        "source_reference",
        "source_reference_mask",
    ):
        if not str(paths.get(key, "")).strip():
            raise FormalConfigError(f"paths.{key} is required")

    data = _mapping(config, "data")
    if bool(data.get("include_adni_mri", False)):
        raise FormalConfigError("ADNI MRI is not part of the physical inventory")
    if bool(data.get("allow_mri_pet_edges", False)):
        raise FormalConfigError("MRI-PET relation edges are forbidden in this run")
    if not bool(data.get("physical_inner_join_only", False)):
        raise FormalConfigError("the formal catalog must be a physical inner join")
    sources = data.get("fomo_sources")
    expected_fomo_sources = [
        "PT001_ClevelandCCF",
        "PT005_ADHD_200",
        "PT014_CoRR",
        "PT015_MSD_BrainTumor",
        "PT021_IXI",
        "PT028_OASIS1",
    ]
    if list(sources or []) != expected_fomo_sources:
        raise FormalConfigError("data.fomo_sources differs from the frozen catalog parser")
    expected_pet_manifests = [
        "Fully_Processed_Amyloid_PET_Manifest_26Jul2026.csv",
        "Fully_Processed_FDG_PET_Manifest_26Jul2026.csv",
        "Fully_Processed_Tau_PET_Manifest_26Jul2026.csv",
    ]
    if list(data.get("pet_manifests") or []) != expected_pet_manifests:
        raise FormalConfigError("data.pet_manifests differs from the frozen catalog parser")
    split = _mapping(data, "split")
    fractions = [float(split.get(name, -1.0)) for name in ("train", "validation", "test")]
    if any(value <= 0.0 for value in fractions) or abs(sum(fractions) - 1.0) > 1.0e-9:
        raise FormalConfigError("data.split fractions must be positive and sum to one")
    if fractions != [0.8, 0.1, 0.1]:
        raise FormalConfigError("the formal split is fixed to 80/10/10")
    if str(split.get("group_key")) != "canonical_subject_id":
        raise FormalConfigError("the formal split must group by canonical_subject_id")
    if not bool(split.get("balance_by_dataset", False)):
        raise FormalConfigError("the formal subject split is dataset-stratified")
    if "soft_balance_fields" in split:
        raise FormalConfigError(
            "data.split.soft_balance_fields is unsupported and must not claim unused balancing"
        )

    preprocessing = _mapping(config, "preprocessing")
    if "quality_control" in preprocessing:
        raise FormalConfigError(
            "preprocessing.quality_control was removed; structural validity checks are not QC"
        )
    shape = preprocessing.get("target_shape")
    if list(shape or []) != [128, 128, 128]:
        raise FormalConfigError("preprocessing.target_shape must stay [128, 128, 128]")
    spacing = preprocessing.get("target_spacing_mm")
    if not isinstance(spacing, list) or len(spacing) != 3:
        raise FormalConfigError("preprocessing.target_spacing_mm must contain three values")
    for index, value in enumerate(spacing):
        _positive_float(value, name=f"preprocessing.target_spacing_mm[{index}]")
    if not bool(preprocessing.get("continue_on_case_error", False)):
        raise FormalConfigError("preprocessing must record a failed case and continue")
    if str(preprocessing.get("registration", {}).get("transform")) not in {
        "rigid_affine",
        "affine",
    }:
        raise FormalConfigError("formal preprocessing supports rigid_affine or affine")
    registration = _mapping(preprocessing, "registration")
    if str(registration.get("mode")) != "simpleitk_rigid_affine":
        raise FormalConfigError(
            "preprocessing.registration.mode must be simpleitk_rigid_affine"
        )
    _positive_int(
        registration.get("threads_per_worker"),
        name="preprocessing.registration.threads_per_worker",
    )
    if bool(registration.get("nonlinear", False)):
        raise FormalConfigError("nonlinear registration is outside the first formal run")

    conditioning = _mapping(config, "metadata_conditioning")
    _positive_int(conditioning.get("dimension"), name="metadata_conditioning.dimension")
    metadata_dropout = _finite_float(
        conditioning.get("dropout_probability"), name="metadata_conditioning.dropout_probability"
    )
    if not 0.0 <= metadata_dropout <= 1.0:
        raise FormalConfigError("metadata_conditioning.dropout_probability must lie in [0,1]")
    expected_categorical = [
        "modality",
        "mri_sequence",
        "mri_product",
        "pet_family",
        "exact_tracer",
        "dwi_representation",
        "mp2rage_component",
        "derived_flag",
    ]
    expected_continuous = [
        "native_spacing_x",
        "native_spacing_y",
        "native_spacing_z",
        "native_resolution_x",
        "native_resolution_y",
        "native_resolution_z",
    ]
    if list(conditioning.get("categorical_fields") or []) != expected_categorical:
        raise FormalConfigError(
            "metadata_conditioning.categorical_fields differs from the model contract"
        )
    if list(conditioning.get("continuous_fields") or []) != expected_continuous:
        raise FormalConfigError(
            "metadata_conditioning.continuous_fields differs from the model contract"
        )

    model = _mapping(config, "model")
    if list(model.get("input_shape") or []) != [128, 128, 128]:
        raise FormalConfigError("model.input_shape must stay [128,128,128]")
    # Width is a model choice, not part of the spatial FEM/cache contract.
    latent_dim = _positive_int(model.get("latent_dim"), name="model.latent_dim")
    sat3d = _mapping(model, "sat3d")
    if str(sat3d.get("builder")) != "swin2":
        raise FormalConfigError("model.sat3d.builder must match the bundled swin2 loader")
    if str(sat3d.get("initialization")) != "pretrained":
        raise FormalConfigError("formal SAT3D initialization must be pretrained")
    if str(sat3d.get("trainability")) != "all":
        raise FormalConfigError("the first formal SAT3D run trains all encoder stages")
    if not bool(sat3d.get("full_volume_input", False)):
        raise FormalConfigError("formal SAT3D must receive the full 128^3 volume")
    if not isinstance(sat3d.get("activation_checkpointing"), bool):
        raise FormalConfigError("model.sat3d.activation_checkpointing must be a boolean")
    adapters = _mapping(model, "adapters")
    if not bool(adapters.get("modality_stems", False)):
        raise FormalConfigError("MRI/PET near-identity stems must be enabled")
    if not bool(adapters.get("stem_zero_initialized", False)):
        raise FormalConfigError("MRI/PET stems must start as near-identity residuals")
    if bool(adapters.get("stage_conditioned_bottlenecks", False)):
        raise FormalConfigError(
            "unsafe internal SAT3D stage hooks are not part of the formal model"
        )
    if not bool(adapters.get("final_feature_conditioned_bottleneck", False)):
        raise FormalConfigError("the safe final SAT3D feature adapter must be enabled")
    if bool(adapters.get("additive_condition_bias", False)):
        raise FormalConfigError("additive condition bias is outside the formal adapter")
    _positive_int(adapters.get("stem_hidden_channels"), name="model.adapters.stem_hidden_channels")
    adapter_ratio = _positive_float(
        adapters.get("bottleneck_ratio"), name="model.adapters.bottleneck_ratio"
    )
    if adapter_ratio > 1.0:
        raise FormalConfigError("model.adapters.bottleneck_ratio must lie in (0,1]")
    projector = _mapping(model, "projector")
    if (
        str(projector.get("type")) != "content_gated_residual"
        or int(projector.get("input_dim", -1)) != 384
        or bool(projector.get("additive_condition_bias", False))
    ):
        raise FormalConfigError("model.projector differs from the implemented formal projector")
    # The builder derives the output from latent_dim.  Explicit values remain
    # checked so a stale YAML entry cannot silently describe another model.
    output_dim = projector.get("output_dim", "auto")
    if output_dim not in (None, "auto"):
        if _positive_int(output_dim, name="model.projector.output_dim") != latent_dim:
            raise FormalConfigError("model.projector.output_dim must equal model.latent_dim (or be auto)")
    fusion = _mapping(model, "fusion")
    if "reliability_weighting" in fusion:
        raise FormalConfigError(
            "model.fusion.reliability_weighting was removed from the formal model"
        )
    if str(fusion.get("type")) != "anchor_centric_gated":
        raise FormalConfigError("model.fusion must be anchor_centric_gated")
    if int(fusion.get("maximum_companions", -1)) != 1:
        raise FormalConfigError("model.fusion.maximum_companions must be one")
    for key in ("relation_embedding_dim", "hidden_dim"):
        _positive_int(fusion.get(key), name=f"model.fusion.{key}")
    geometry = _mapping(model, "geometry")
    if list(geometry.get("token_grid") or []) != [8, 8, 8]:
        raise FormalConfigError("model.geometry.token_grid must be [8,8,8]")
    _positive_int(
        geometry.get("interpolation_neighbors"),
        name="model.geometry.interpolation_neighbors",
    )
    _positive_float(
        geometry.get("interpolation_sigma_mm"),
        name="model.geometry.interpolation_sigma_mm",
    )
    for key in ("nodes_target", "modes", "geomc_blocks", "hidden_dim", "scale_embedding_dim"):
        _positive_int(geometry.get(key), name=f"model.geometry.{key}")
    if geometry["nodes_target"] < 8 or not 2 <= geometry["modes"] < geometry["nodes_target"]:
        raise FormalConfigError("FEM requires nodes_target >= 8 and 2 <= modes < nodes_target")
    if geometry["geomc_blocks"] < 3:
        raise FormalConfigError("model.geometry.geomc_blocks must be at least 3 for GeoMCCore")
    _finite_float(geometry.get("residual_scale"), name="model.geometry.residual_scale")
    radii = geometry.get("resolvent_radii")
    if not isinstance(radii, (list, tuple)) or not radii:
        raise FormalConfigError("model.geometry.resolvent_radii must be a non-empty sequence")
    parsed_radii = [
        _positive_float(value, name=f"model.geometry.resolvent_radii[{index}]")
        for index, value in enumerate(radii)
    ]
    if any(right <= left for left, right in zip(parsed_radii, parsed_radii[1:])):
        raise FormalConfigError("model.geometry.resolvent_radii must be strictly increasing")
    embedding = _mapping(geometry, "embedding")
    _positive_int(embedding.get("hidden_dim"), name="model.geometry.embedding.hidden_dim")
    _finite_float(embedding.get("initial_scale"), name="model.geometry.embedding.initial_scale")
    if str(embedding.get("type")) not in {"spectral", "spectral_xyz"}:
        raise FormalConfigError("model.geometry.embedding.type must be spectral or spectral_xyz")
    predictor = _mapping(model, "predictor")
    if str(predictor.get("type")) != "query_aware_3d_vit":
        raise FormalConfigError("model.predictor must be query_aware_3d_vit")
    if not bool(predictor.get("physical_position_encoding", False)):
        raise FormalConfigError("the JEPA predictor requires physical position encoding")
    if bool(predictor.get("raw_sat3d_skip", False)):
        raise FormalConfigError("raw SAT3D skips are forbidden after GeoMC")
    predictor_dim = _positive_int(predictor.get("embed_dim"), name="model.predictor.embed_dim")
    heads = _positive_int(predictor.get("heads"), name="model.predictor.heads")
    _positive_int(predictor.get("depth"), name="model.predictor.depth")
    max_queries = _positive_int(predictor.get("max_queries"), name="model.predictor.max_queries")
    if predictor_dim < 6 or predictor_dim % heads:
        raise FormalConfigError("model.predictor.embed_dim must be >= 6 and divisible by heads")
    if max_queries > math.prod(geometry["token_grid"]):
        raise FormalConfigError("model.predictor.max_queries cannot exceed the token grid size")
    mlp_ratio = _positive_float(predictor.get("mlp_ratio"), name="model.predictor.mlp_ratio")
    if not math.isfinite(predictor_dim * mlp_ratio) or round(predictor_dim * mlp_ratio) < 1:
        raise FormalConfigError("model.predictor.mlp_ratio must produce a positive finite hidden dimension")
    drop_path = _finite_float(predictor.get("drop_path"), name="model.predictor.drop_path")
    if not 0.0 <= drop_path < 1.0:
        raise FormalConfigError("model.predictor.drop_path must lie in [0,1)")
    masking = _mapping(config, "masking")
    if _positive_int(masking.get("query_tokens"), name="masking.query_tokens") != max_queries:
        raise FormalConfigError("masking.query_tokens must equal model.predictor.max_queries")
    hidden_fraction = _finite_float(masking.get("hidden_fraction"), name="masking.hidden_fraction")
    if not 0.0 < hidden_fraction < 1.0:
        raise FormalConfigError("masking.hidden_fraction must lie in (0,1)")
    _positive_int(masking.get("blocks"), name="masking.blocks")
    ema_scope = _mapping(model, "ema")
    expected_ema_includes = [
        "stems",
        "condition_encoder",
        "encoder",
        "feature_adapter",
        "projector",
    ]
    expected_ema_excludes = [
        "anchor_companion_fusion",
        "geometry_embedding",
        "geomc",
        "predictor",
        "relation_embedding",
    ]
    if list(ema_scope.get("includes") or []) != expected_ema_includes:
        raise FormalConfigError("model.ema.includes differs from OnlineEMABundle")
    if list(ema_scope.get("excludes") or []) != expected_ema_excludes:
        raise FormalConfigError("model.ema.excludes differs from OnlineEMABundle")

    training = _mapping(config, "training")
    if bool(training.get("early_stopping", False)):
        raise FormalConfigError("performance early stopping is disabled")
    if training.get("performance_gate") not in (None, "none"):
        raise FormalConfigError("training.performance_gate must be none")
    epochs = _positive_int(training.get("epochs"), name="training.epochs")
    scheduler = _mapping(training, "scheduler")
    if (
            _positive_int(
                scheduler.get("horizon_coverages"),
                name="training.scheduler.horizon_coverages",
            )
            != epochs
    ):
        raise FormalConfigError(
            "training.scheduler.horizon_coverages must equal training.epochs"
        )
    warmup_coverages = _nonnegative_int(
        scheduler.get("warmup_coverages"),
        name="training.scheduler.warmup_coverages",
    )
    if warmup_coverages >= epochs:
        raise FormalConfigError(
            "training.scheduler.warmup_coverages must be less than training.epochs"
        )
    microbatch = _positive_int(training.get("microbatch_size"), name="training.microbatch_size")
    accumulation = _positive_int(
        training.get("gradient_accumulation"), name="training.gradient_accumulation"
    )
    expected_global = _positive_int(training.get("global_batch_size"), name="training.global_batch_size")
    if microbatch * accumulation != expected_global:
        raise FormalConfigError("single-GPU microbatch * accumulation must equal global batch")
    if int(training.get("max_companions", -1)) != 1:
        raise FormalConfigError("the first formal run permits at most one companion per anchor")
    if not bool(training.get("finish_fixed_horizon", False)):
        raise FormalConfigError("training.finish_fixed_horizon must be true")
    loss = _mapping(training, "loss")
    if str(loss.get("companion_policy")) != "context_only_for_anchor_self_jepa":
        raise FormalConfigError("companions must remain context-only for anchor self-JEPA")
    equality_weights = [
        float(value)
        for key, value in loss.items()
        if str(key).endswith("_weight") and "latent_equality" in str(key)
    ]
    if any(value != 0.0 for value in equality_weights):
        raise FormalConfigError("companion relations are context-only, not latent equality")
    for key in ("cross_tracer_weight", "mri_pet_weight"):
        if float(loss.get(key, 0.0)) != 0.0:
            raise FormalConfigError(f"training.loss.{key} must be zero")
    schedule = _mapping(training, "relation_schedule")
    mixture = _mapping(schedule, "steady_mixture")
    mixture_values = [
        _finite_float(value, name=f"training.relation_schedule.steady_mixture.{key}")
        for key, value in mixture.items()
    ]
    if any(value < 0 for value in mixture_values) or abs(sum(mixture_values) - 1.0) > 1.0e-9:
        raise FormalConfigError("training.relation_schedule.steady_mixture must sum to one")
    if "mri_pet" in " ".join(str(key).lower() for key in mixture):
        raise FormalConfigError("MRI-PET mixture entries are forbidden")
    expected_mixture = {
        "self_only": 0.50,
        "same_session_cross_sequence": 0.30,
        "longitudinal_same_acquisition": 0.15,
        "same_session_repeat": 0.05,
    }
    if set(mixture) != set(expected_mixture) or any(
        abs(float(mixture[key]) - expected) > 1.0e-12
        for key, expected in expected_mixture.items()
    ):
        raise FormalConfigError("the first formal relation mixture is fixed at 50/30/15/5")
    if list(schedule.get("self_only_coverages") or []) != [1, 2]:
        raise FormalConfigError("relation self-only coverages must be [1,2]")
    if list(schedule.get("ramp_coverages") or []) != [3, 4, 5]:
        raise FormalConfigError("relation ramp coverages must be [3,4,5]")
    if int(schedule.get("steady_start_coverage", -1)) != 6:
        raise FormalConfigError("the fixed relation mixture must start at coverage 6")
    checkpoint = _mapping(training, "checkpoint")
    if bool(checkpoint.get("keep_best", False)):
        raise FormalConfigError("performance-selected checkpoints are disabled")
    _positive_int(checkpoint.get("keep_last"), name="training.checkpoint.keep_last")
    _positive_int(checkpoint.get("every_updates"), name="training.checkpoint.every_updates")
    _positive_int(
        checkpoint.get("keep_every_coverages"),
        name="training.checkpoint.keep_every_coverages",
    )

    validation = _mapping(config, "validation")
    if bool(validation.get("stop_training", False)):
        raise FormalConfigError("validation must be observational only")
    if bool(validation.get("select_best_checkpoint", False)):
        raise FormalConfigError("the fixed-horizon first run does not select by performance")
    if not bool(validation.get("test_only_after_training", False)):
        raise FormalConfigError("the held-out test is allowed only after the fixed horizon")
    if not bool(validation.get("health_every_coverage", False)):
        raise FormalConfigError("observational health validation must run every coverage")
    _positive_int(
        validation.get("full_every_coverages"),
        name="validation.full_every_coverages",
    )
    if not bool(validation.get("aggregate_by_subject", False)):
        raise FormalConfigError("validation/test metrics must aggregate by subject first")
    modality_weights = _mapping(validation, "modality_macro_weights")
    if set(modality_weights) != {"mri", "pet"} or any(
        abs(float(modality_weights[key]) - 0.5) > 1.0e-12
        for key in ("mri", "pet")
    ):
        raise FormalConfigError("validation modality macro weights must be MRI/PET 0.5/0.5")

    logging = _mapping(config, "logging")
    if not bool(logging.get("rich_progress", False)):
        raise FormalConfigError("the formal terminal UI requires Rich progress")
    if not bool(logging.get("jsonl_metrics", False)):
        raise FormalConfigError("synchronous JSONL metrics are required")
    if int(logging.get("step_log_every", -1)) != 1:
        raise FormalConfigError("the formal run records every optimizer step")
    if not bool(logging.get("flush_each_event", False)):
        raise FormalConfigError("every text/JSONL log event must be flushed synchronously")

    resources = _mapping(config, "resources")
    if not bool(resources.get("single_gpu_first", False)):
        raise FormalConfigError("the first formal run is fixed to one GPU")
    if not bool(resources.get("vram_soft_limit_is_warning_only", False)):
        raise FormalConfigError("the VRAM planning value must remain warning-only")


def config_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_formal_config(path: str | Path) -> dict[str, Any]:
    """Load, resolve, validate, and hash a formal-run YAML configuration."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise FormalConfigError("configuration root must be a mapping")
    config: dict[str, Any] = copy.deepcopy(_expand(raw))
    project_root = source.parent.parent if source.parent.name == "configs" else source.parent
    _resolve_paths(config, project_root=project_root)
    validate_formal_config(config)
    config["_resolved"] = {
        "config_path": str(source),
        "project_root": str(project_root.resolve()),
    }
    config["_resolved"]["digest"] = config_digest(config)
    return config


__all__ = [
    "FormalConfigError",
    "config_digest",
    "load_formal_config",
    "validate_formal_config",
]
