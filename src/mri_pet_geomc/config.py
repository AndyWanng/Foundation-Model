"""Configuration loading and validation for the main GeoMC pipeline."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


METHOD_NAME = "shared_sat3d_geomc"
INPUT_NORMALIZATION_METHOD = "brain_mask_zscore_v1"
RELATIONS = {"mri->mri", "pet->pet", "mri->pet", "pet->mri"}

REQUIRED_EXPERIMENT_SETTINGS = {
    "id": "geomc",
    "verified_pair_retention_fraction": 1.0,
    "pairing_mode": "verified",
    "seeds": [0],
}


# YAML keys are part of the public run contract. Unknown keys are rejected so
# a typo cannot survive into a resolved config while being ignored by code.
SECTION_KEYS: dict[str, set[str]] = {
    "project": {"name", "method", "seed", "scientific_use", "claim_scope"},
    "run": {"mode", "launch_dir"},
    "execution": {
        "policy",
        "scientific_thresholds_block_execution",
        "qc_findings_block_execution",
        "smoke_test_findings_block_execution",
        "overfit_findings_block_execution",
        "experiment_failure_blocks_other_experiments",
    },
    "paths": {
        "artifacts_root",
        "derivatives_root",
        "cache_root",
        "dataset_root",
        "legacy_pairings",
        "legacy_splits",
        "legacy_pet128_manifest",
        "sat3d_code_root",
        "sat3d_checkpoint",
        "model_reference_128",
        "model_mask_128",
    },
    "data": {
        "cohort_mode",
        "expected_subjects",
        "split_counts",
        "synthetic_input_size",
        "input_normalization",
        "modalities",
        "allow_pseudo_pairs",
        "checksum",
        "frozen_verified_pair_retention_fraction_subject_order",
    },
    "preprocessing": {"write_legacy_roots", "pet128"},
    "registration_qc": {
        "policy",
        "manual_review_required",
        "statistical_flags_block_training",
        "affine_tolerance",
        "minimum_support_voxels",
        "diagnostic_maximum_axis",
        "histogram_bins",
        "mask_dice_flag_below",
        "centroid_distance_mm_flag_above",
        "nmi_shift_margin_flag_below",
    },
    "sat3d": {
        "builder",
        "strict_checkpoint",
        "expected_checkpoint_sha256",
        "expected_source_tree_sha256",
        "expected_feature_shape",
        "trainability",
        "initialization",
        "use_native_activation_checkpoint",
        "tiny_hidden_channels",
        "tiny_output_channels",
    },
    "geometry": {
        "domain",
        "nodes_target",
        "modes",
        "token_grid",
        "interpolation_neighbors",
        "interpolation_sigma_mm",
        "resolvent_radii_mm",
        "normalize_spectrum",
        "exact_complement",
    },
    "model": {
        "latent_dim",
        "metadata_dim",
        "geomc_hidden_dim",
        "geomc_scale_embedding_dim",
        "geomc_residual_scale",
        "geomc_blocks",
    },
    "geometry_embedding": {
        "type",
        "hidden_dim",
        "initial_scale",
        "trainable_scale",
    },
    "inputs": {
        "max_observations",
        "include_target_observation_probability",
        "aggregation",
    },
    "masking": {
        "hidden_fraction",
        "query_tokens",
        "source_mask_before_encoder",
        "normalization_uses_hidden_voxels",
    },
    "overfit": {"max_updates", "minimum_relative_drop_report_only"},
    "training": {
        "objective",
        "microbatch_size",
        "gradient_accumulation",
        "relation_cycle_length",
        "precision",
        "optimizer",
        "encoder_learning_rate",
        "model_learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "warmup_updates",
        "validation_every",
        "checkpoint_every",
        "validation_max_examples",
        "validation_views",
        "validation_pairing_policy",
        "huber_beta",
        "ema_start",
        "ema_end",
        "relation_sampling_weights",
    },
    "evaluation": {
        "aggregate_by_subject",
        "validation_best_only",
        "checkpoint_name",
        "query_mask_views",
        "required_model_diagnostics",
        "all_thresholds_report_only",
        "bootstrap_resamples",
        "comparisons",
        "primary_experiment_id",
        "checkpoint_policy",
        "fixed_reference",
        "reference_alignment",
        "alignment_fit_split",
    },
    "resources": {
        "device",
        "encoder_backend",
        "require_cuda",
        "min_free_vram_gib",
        "min_free_ram_gib",
        "min_free_disk_gib",
    },
    "runtime": {"config_path", "project_root"},
}

EXPERIMENT_KEYS = {
    "id",
    "verified_pair_retention_fraction",
    "pairing_mode",
    "seeds",
    "max_updates",
    "planned_total_updates",
    "allowed_relations",
    "trainability",
    "initialization",
}


class ConfigurationError(ValueError):
    pass


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Top-level YAML must be a mapping: {source}")
    return _expand(payload)


def load_config(path: str | Path, *, project_root: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = load_yaml(source)
    if "extends" in raw:
        parent = Path(str(raw.pop("extends")))
        if not parent.is_absolute():
            parent = source.parent / parent
        raw = deep_merge(load_config(parent, project_root=project_root), raw)
    raw.setdefault("runtime", {})
    raw["runtime"].update(
        {"config_path": str(source), "project_root": str(Path(project_root).resolve())}
    )
    validate_config(raw)
    return raw


def get(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def require(config: Mapping[str, Any], dotted: str) -> Any:
    value = get(config, dotted, None)
    if value in (None, ""):
        raise ConfigurationError(f"Missing required configuration value: {dotted}")
    return value


def _sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise ConfigurationError(f"{name} must be a 64-character SHA-256")


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], *, name: str
) -> None:
    unknown = sorted(set(str(key) for key in value) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown {name} configuration keys: {unknown}")


def _require_mapping(config: Mapping[str, Any], dotted: str) -> Mapping[str, Any]:
    value = require(config, dotted)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{dotted} must be a mapping")
    return value


def _finite(config: Mapping[str, Any], dotted: str) -> float:
    try:
        value = float(require(config, dotted))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{dotted} must be numeric") from error
    if not math.isfinite(value):
        raise ConfigurationError(f"{dotted} must be finite")
    return value


def _positive_int(config: Mapping[str, Any], dotted: str) -> int:
    try:
        raw = require(config, dotted)
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{dotted} must be an integer") from error
    if isinstance(raw, bool) or value != raw or value < 1:
        raise ConfigurationError(f"{dotted} must be a positive integer")
    return value


def _require_equal(config: Mapping[str, Any], dotted: str, expected: Any) -> None:
    actual = get(config, dotted, None)
    if actual != expected:
        raise ConfigurationError(
            f"{dotted} is an implementation invariant and must equal {expected!r}"
        )


def _validate_main_experiment(config: Mapping[str, Any]) -> None:
    experiments = get(config, "experiments", [])
    if not isinstance(experiments, list) or len(experiments) != 1:
        raise ConfigurationError("experiments must contain exactly one main GeoMC model")
    experiment = experiments[0]
    if not isinstance(experiment, Mapping):
        raise ConfigurationError("The main experiment must be a mapping")
    _reject_unknown_keys(experiment, EXPERIMENT_KEYS, name="experiment")

    for key, expected in REQUIRED_EXPERIMENT_SETTINGS.items():
        actual = experiment.get(key)
        if actual != expected:
            raise ConfigurationError(
                f"Main experiment field {key!r} must equal {expected!r}; "
                f"found {actual!r}"
            )

    max_updates = _positive_int({"experiment": experiment}, "experiment.max_updates")
    planned = _positive_int(
        {"experiment": experiment}, "experiment.planned_total_updates"
    )
    if planned < max_updates:
        raise ConfigurationError(
            "experiment.planned_total_updates must cover experiment.max_updates"
        )

    allowed_relations = experiment.get("allowed_relations")
    if allowed_relations is not None:
        values = [str(value) for value in allowed_relations]
        if not values or len(values) != len(set(values)) or not set(values) <= RELATIONS:
            raise ConfigurationError("Main experiment has invalid allowed_relations")

    trainability = str(experiment.get("trainability", get(config, "sat3d.trainability")))
    if trainability not in {"frozen", "last_stage", "all"}:
        raise ConfigurationError("Main experiment has invalid trainability")
    initialization = str(
        experiment.get("initialization", get(config, "sat3d.initialization"))
    )
    if initialization not in {"pretrained", "scratch"}:
        raise ConfigurationError("Main experiment has invalid initialization")

    _require_equal(config, "evaluation.primary_experiment_id", "geomc")
    comparisons = get(config, "evaluation.comparisons", None)
    if comparisons != []:
        raise ConfigurationError(
            "evaluation.comparisons must be empty for the single main model"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    top_level = set(SECTION_KEYS) | {"schema_version", "experiments"}
    _reject_unknown_keys(config, top_level, name="top-level")
    _require_equal(config, "schema_version", 1)
    for section, allowed in SECTION_KEYS.items():
        section_value = _require_mapping(config, section)
        _reject_unknown_keys(section_value, allowed, name=section)

    mode = str(get(config, "run.mode", ""))
    if mode not in {"smoke", "gpu_smoke_test", "full"}:
        raise ConfigurationError("run.mode must be smoke, gpu_smoke_test, or full")
    _require_equal(config, "project.method", METHOD_NAME)
    for dotted in (
        "project.name",
        "project.seed",
        "project.claim_scope",
        "paths.artifacts_root",
        "paths.derivatives_root",
        "paths.cache_root",
    ):
        require(config, dotted)
    _positive_int(config, "project.seed")
    if not isinstance(get(config, "project.scientific_use", None), bool):
        raise ConfigurationError("project.scientific_use must be boolean")

    _require_equal(config, "execution.policy", "exhaustive_nonblocking_v1")
    for dotted in (
        "execution.scientific_thresholds_block_execution",
        "execution.qc_findings_block_execution",
        "execution.smoke_test_findings_block_execution",
        "execution.overfit_findings_block_execution",
        "execution.experiment_failure_blocks_other_experiments",
    ):
        _require_equal(config, dotted, False)

    _require_equal(config, "data.input_normalization", INPUT_NORMALIZATION_METHOD)
    _require_equal(config, "data.allow_pseudo_pairs", False)
    _require_equal(config, "data.modalities", ["mri", "pet"])
    _require_equal(config, "data.checksum", True)
    _require_equal(
        config,
        "data.cohort_mode",
        "legacy_frozen_30" if mode == "full" else "synthetic",
    )
    _positive_int(config, "data.synthetic_input_size")

    _require_equal(config, "preprocessing.write_legacy_roots", False)
    pet128 = _require_mapping(config, "preprocessing.pet128")
    _reject_unknown_keys(
        pet128,
        {"one_resample_only", "relative_normalization", "quantification_claim"},
        name="preprocessing.pet128",
    )
    _require_equal(config, "preprocessing.pet128.one_resample_only", True)
    _require_equal(
        config,
        "preprocessing.pet128.relative_normalization",
        "eroded_brain_relative_median_after_p005_p995_clip",
    )
    _require_equal(
        config, "preprocessing.pet128.quantification_claim", "relative_fdg_only"
    )

    _require_equal(config, "registration_qc.policy", "automatic_nonblocking_v1")
    _require_equal(config, "registration_qc.manual_review_required", False)
    _require_equal(config, "registration_qc.statistical_flags_block_training", False)
    if _finite(config, "registration_qc.affine_tolerance") <= 0:
        raise ConfigurationError("registration_qc.affine_tolerance must be positive")
    _positive_int(config, "registration_qc.minimum_support_voxels")
    _positive_int(config, "registration_qc.diagnostic_maximum_axis")
    if _positive_int(config, "registration_qc.histogram_bins") < 2:
        raise ConfigurationError("registration_qc.histogram_bins must be at least 2")
    dice_threshold = _finite(config, "registration_qc.mask_dice_flag_below")
    if not 0.0 <= dice_threshold <= 1.0:
        raise ConfigurationError("registration_qc.mask_dice_flag_below must lie in [0,1]")
    if _finite(config, "registration_qc.centroid_distance_mm_flag_above") < 0:
        raise ConfigurationError(
            "registration_qc.centroid_distance_mm_flag_above must be non-negative"
        )
    _finite(config, "registration_qc.nmi_shift_margin_flag_below")

    if str(get(config, "sat3d.trainability", "")) not in {"frozen", "last_stage", "all"}:
        raise ConfigurationError("sat3d.trainability must be frozen, last_stage, or all")
    _require_equal(config, "sat3d.builder", "swin2")
    _require_equal(config, "sat3d.use_native_activation_checkpoint", False)
    _require_equal(config, "sat3d.strict_checkpoint", True)
    _require_equal(config, "sat3d.expected_feature_shape", [384, 8, 8, 8])
    if str(get(config, "sat3d.initialization", "")) not in {"pretrained", "scratch"}:
        raise ConfigurationError("sat3d.initialization must be pretrained or scratch")
    for dotted in ("sat3d.tiny_hidden_channels", "sat3d.tiny_output_channels"):
        if get(config, dotted, None) is not None:
            _positive_int(config, dotted)
    _sha256(
        str(require(config, "sat3d.expected_checkpoint_sha256")),
        name="sat3d.expected_checkpoint_sha256",
    )
    _sha256(
        str(require(config, "sat3d.expected_source_tree_sha256")),
        name="sat3d.expected_source_tree_sha256",
    )

    radii = [float(value) for value in get(config, "geometry.resolvent_radii_mm", [])]
    if (
        not radii
        or any(not math.isfinite(value) or value <= 0 for value in radii)
        or any(right <= left for left, right in zip(radii, radii[1:]))
    ):
        raise ConfigurationError(
            "geometry.resolvent_radii_mm must be positive and strictly increasing"
        )
    spectral_bands = len(radii) + 1
    modes = _positive_int(config, "geometry.modes")
    nodes = _positive_int(config, "geometry.nodes_target")
    if modes < spectral_bands:
        raise ConfigurationError("geometry.modes must cover every spectral band")
    if nodes < modes:
        raise ConfigurationError("geometry.nodes_target must be at least geometry.modes")
    _require_equal(config, "geometry.normalize_spectrum", False)
    _require_equal(
        config,
        "geometry.domain",
        "fixed_template_volumetric_fem" if mode == "full" else "synthetic_volumetric_fem",
    )
    _require_equal(config, "geometry.exact_complement", True)
    token_grid = list(require(config, "geometry.token_grid"))
    if len(token_grid) != 3 or any(
        isinstance(value, bool) or int(value) != value or int(value) < 1
        for value in token_grid
    ):
        raise ConfigurationError("geometry.token_grid must contain three positive integers")
    _positive_int(config, "geometry.interpolation_neighbors")
    if _finite(config, "geometry.interpolation_sigma_mm") <= 0:
        raise ConfigurationError("geometry.interpolation_sigma_mm must be positive")
    if _positive_int(config, "model.geomc_blocks") < 3:
        raise ConfigurationError("GeoMC requires at least three repeated cells")
    if _positive_int(config, "model.latent_dim") < 8:
        raise ConfigurationError("model.latent_dim is implausibly small")
    if _positive_int(config, "model.metadata_dim") < 4:
        raise ConfigurationError("model.metadata_dim must encode modality and metadata availability")
    _positive_int(config, "model.geomc_hidden_dim")
    _positive_int(config, "model.geomc_scale_embedding_dim")
    if _finite(config, "model.geomc_residual_scale") <= 0:
        raise ConfigurationError("model.geomc_residual_scale must be positive")

    _require_equal(config, "geometry_embedding.type", "spectral")
    _positive_int(config, "geometry_embedding.hidden_dim")
    _finite(config, "geometry_embedding.initial_scale")
    if not isinstance(get(config, "geometry_embedding.trainable_scale", None), bool):
        raise ConfigurationError("geometry_embedding.trainable_scale must be boolean")

    _require_equal(config, "inputs.max_observations", 2)
    _require_equal(config, "inputs.aggregation", "reliability_weighted_mean")
    target_observation_probability = _finite(config, "inputs.include_target_observation_probability")
    if not 0.0 <= target_observation_probability <= 1.0:
        raise ConfigurationError("include_target_observation_probability must lie in [0,1]")

    hidden_fraction = _finite(config, "masking.hidden_fraction")
    if not 0.0 < hidden_fraction < 1.0:
        raise ConfigurationError("masking.hidden_fraction must lie strictly inside (0,1)")
    query_tokens = _positive_int(config, "masking.query_tokens")
    if query_tokens < 2 or query_tokens >= math.prod(int(value) for value in token_grid):
        raise ConfigurationError(
            "masking.query_tokens must be at least 2 and smaller than the token grid"
        )
    _require_equal(config, "masking.source_mask_before_encoder", True)
    _require_equal(config, "masking.normalization_uses_hidden_voxels", False)

    _positive_int(config, "overfit.max_updates")
    _finite(config, "overfit.minimum_relative_drop_report_only")

    _require_equal(
        config,
        "training.objective",
        "masked_huber_v1",
    )
    if _positive_int(config, "training.microbatch_size") != 1:
        raise ConfigurationError("Current online SAT3D requires microbatch_size=1")
    _positive_int(config, "training.gradient_accumulation")
    _positive_int(config, "training.relation_cycle_length")
    if str(get(config, "training.precision", "")) not in {"fp32", "bf16"}:
        raise ConfigurationError("training.precision must be fp32 or bf16")
    _require_equal(config, "training.optimizer", "adamw")
    _require_equal(config, "training.huber_beta", 1.0)
    for dotted in ("training.encoder_learning_rate", "training.model_learning_rate"):
        if _finite(config, dotted) <= 0:
            raise ConfigurationError(f"{dotted} must be positive")
    if _finite(config, "training.weight_decay") < 0:
        raise ConfigurationError("training.weight_decay must be non-negative")
    if _finite(config, "training.gradient_clip_norm") <= 0:
        raise ConfigurationError("training.gradient_clip_norm must be positive")
    for dotted in ("training.warmup_updates",):
        raw = require(config, dotted)
        value = int(raw)
        if isinstance(raw, bool) or value != raw or value < 0:
            raise ConfigurationError(f"{dotted} must be a non-negative integer")
    for dotted in (
        "training.validation_every",
        "training.checkpoint_every",
        "training.validation_max_examples",
        "training.validation_views",
    ):
        _positive_int(config, dotted)
    _require_equal(
        config,
        "training.validation_pairing_policy",
        "experiment_relations",
    )
    ema_start = _finite(config, "training.ema_start")
    ema_end = _finite(config, "training.ema_end")
    if not 0.0 <= ema_start <= ema_end < 1.0:
        raise ConfigurationError("training EMA values must satisfy 0 <= start <= end < 1")
    relation_sampling_weights = _require_mapping(config, "training.relation_sampling_weights")
    if set(relation_sampling_weights) != {
        "mri_to_mri",
        "pet_to_pet",
        "mri_to_pet",
        "pet_to_mri",
    }:
        raise ConfigurationError(
            "training.relation_sampling_weights must define exactly the four MRI/PET relations"
        )
    relation_weights = [float(value) for value in relation_sampling_weights.values()]
    if any(not math.isfinite(value) or value < 0 for value in relation_weights):
        raise ConfigurationError("relation sampling weights must be finite and non-negative")
    if not math.isclose(sum(relation_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ConfigurationError("relation sampling weights must sum to one")

    _require_equal(config, "evaluation.aggregate_by_subject", True)
    _require_equal(config, "evaluation.validation_best_only", False)
    _require_equal(config, "evaluation.checkpoint_name", "final.pt")
    _require_equal(config, "evaluation.checkpoint_policy", "fixed_final_equal_update")
    _require_equal(config, "evaluation.fixed_reference", "initial_target_encoder_v1")
    _require_equal(
        config,
        "evaluation.reference_alignment",
        "global_subject_weighted_similarity_procrustes",
    )
    _require_equal(config, "evaluation.alignment_fit_split", "train")
    _require_equal(config, "evaluation.all_thresholds_report_only", True)
    if _positive_int(config, "evaluation.query_mask_views") < 2:
        raise ConfigurationError("evaluation.query_mask_views must be at least 2")
    expected_diagnostics = {
        "signed_router",
        "parseval_additive_energy",
        "centered_scale_energy",
        "frame_closure",
        "complement_residual",
        "representation_retention",
    }
    actual_diagnostics = frozenset(
        str(value)
        for value in get(config, "evaluation.required_model_diagnostics", [])
    )
    if actual_diagnostics != frozenset(expected_diagnostics):
        raise ConfigurationError(
            "evaluation.required_model_diagnostics must match the main "
            "GeoMC model diagnostics"
        )
    _positive_int(config, "evaluation.bootstrap_resamples")

    device = str(get(config, "resources.device", ""))
    backend = str(get(config, "resources.encoder_backend", ""))
    require_cuda = get(config, "resources.require_cuda", None)
    if device not in {"cpu", "cuda"}:
        raise ConfigurationError("resources.device must be cpu or cuda")
    if backend not in {"tiny", "sat3d"}:
        raise ConfigurationError("resources.encoder_backend must be tiny or sat3d")
    if not isinstance(require_cuda, bool) or require_cuda != (device == "cuda"):
        raise ConfigurationError(
            "resources.require_cuda must be true exactly when resources.device is cuda"
        )
    for dotted in (
        "resources.min_free_vram_gib",
        "resources.min_free_ram_gib",
        "resources.min_free_disk_gib",
    ):
        if _finite(config, dotted) < 0:
            raise ConfigurationError(f"{dotted} must be non-negative")

    _validate_main_experiment(config)

    if mode == "full":
        for dotted in (
            "paths.legacy_pairings",
            "paths.legacy_splits",
            "paths.legacy_pet128_manifest",
            "paths.sat3d_code_root",
            "paths.sat3d_checkpoint",
            "paths.model_reference_128",
            "paths.model_mask_128",
        ):
            require(config, dotted)
        _require_equal(config, "resources.device", "cuda")
        _require_equal(config, "resources.encoder_backend", "sat3d")
        _require_equal(config, "training.precision", "bf16")
        _require_equal(config, "sat3d.trainability", "last_stage")
        _require_equal(config, "sat3d.initialization", "pretrained")
        _require_equal(config, "geometry.token_grid", [8, 8, 8])
        _require_equal(config, "geometry.modes", 128)
        if _positive_int(config, "training.validation_views") < 2:
            raise ConfigurationError("Full validation must include both query views")

    counts = get(config, "data.split_counts", {})
    if not isinstance(counts, Mapping):
        raise ConfigurationError("data.split_counts must be a mapping")
    expected_subjects = _positive_int(config, "data.expected_subjects")
    split_total = sum(
        int(counts.get(name, 0)) for name in ("train", "validation", "test")
    )
    if split_total != expected_subjects or any(
        int(counts.get(name, 0)) < 1 for name in ("train", "validation", "test")
    ):
        raise ConfigurationError(
            "split_counts must be positive and sum to expected_subjects"
        )


__all__ = [
    "REQUIRED_EXPERIMENT_SETTINGS",
    "ConfigurationError",
    "METHOD_NAME",
    "get",
    "load_config",
    "require",
    "validate_config",
]
