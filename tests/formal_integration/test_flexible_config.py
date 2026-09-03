"""Config-only coverage for configurable formal widths and training controls.

The two asset-contract/weight-shape tests import their runtime dependencies
inside the test.  No test builds SAT3D, reads medical images or loads a real
checkpoint; the shape check instantiates only two small CPU projectors.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from mri_pet_geomc.formal.config import (
    FormalConfigError,
    config_digest,
    validate_formal_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def formal_config() -> dict[str, Any]:
    with (PROJECT_ROOT / "configs" / "workstation_formal.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def _set(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _wide_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["model"]["latent_dim"] = 384
    result["model"]["projector"]["output_dim"] = 384
    result["metadata_conditioning"]["dimension"] = 192
    result["model"]["geometry"]["hidden_dim"] = 512
    result["model"]["geometry"]["geomc_blocks"] = 6
    result["model"]["fusion"]["hidden_dim"] = 256
    result["model"]["predictor"].update(
        embed_dim=384, depth=8, heads=8, max_queries=256
    )
    result["masking"]["query_tokens"] = 256
    return result


def test_legacy_128_config_remains_valid(formal_config: dict[str, Any]) -> None:
    assert formal_config["model"]["latent_dim"] == 128
    assert formal_config["model"]["projector"]["output_dim"] == 128
    validate_formal_config(formal_config)


def test_384_latent_and_wider_deeper_predictor_are_valid(
    formal_config: dict[str, Any],
) -> None:
    wide = _wide_config(formal_config)
    validate_formal_config(wide)
    assert config_digest(wide) != config_digest(formal_config)


@pytest.mark.parametrize("latent_dim,condition_dim", [(1, 1), (256, 64), (384, 192)])
def test_latent_and_condition_widths_are_independent(
    formal_config: dict[str, Any], latent_dim: int, condition_dim: int
) -> None:
    formal_config["model"]["latent_dim"] = latent_dim
    formal_config["model"]["projector"]["output_dim"] = latent_dim
    formal_config["metadata_conditioning"]["dimension"] = condition_dim
    validate_formal_config(formal_config)


@pytest.mark.parametrize("output_dim", [128, 256])
def test_projector_output_must_match_latent_width(
    formal_config: dict[str, Any], output_dim: int
) -> None:
    formal_config["model"]["latent_dim"] = 384
    formal_config["model"]["projector"]["output_dim"] = output_dim
    with pytest.raises(FormalConfigError):
        validate_formal_config(formal_config)


def test_projector_input_stays_at_sat3d_feature_width(
    formal_config: dict[str, Any],
) -> None:
    formal_config["model"]["projector"]["input_dim"] = 512
    with pytest.raises(FormalConfigError):
        validate_formal_config(formal_config)


@pytest.mark.parametrize(
    "path,value",
    [
        ("model.latent_dim", 0),
        ("model.latent_dim", True),
        ("model.latent_dim", 128.5),
        ("metadata_conditioning.dimension", 0),
        ("model.projector.output_dim", 0),
        ("model.predictor.embed_dim", 4),
        ("model.predictor.depth", 0),
        ("model.predictor.heads", 0),
        ("model.predictor.heads", 7),
        ("model.predictor.max_queries", 0),
        ("model.predictor.max_queries", 513),
        ("masking.query_tokens", 64),
        ("model.geometry.geomc_blocks", 2),
        ("model.geometry.hidden_dim", 0),
        ("model.geometry.scale_embedding_dim", 0),
        ("model.geometry.embedding.hidden_dim", 0),
        ("model.fusion.hidden_dim", 0),
        ("model.fusion.relation_embedding_dim", 0),
        ("preprocessing.registration.threads_per_worker", 0),
        ("training.checkpoint.keep_last", 0),
        ("training.checkpoint.keep_every_coverages", 0),
        ("validation.full_every_coverages", 0),
    ],
)
def test_invalid_structural_controls_are_rejected(
    formal_config: dict[str, Any], path: str, value: Any
) -> None:
    _set(formal_config, path, value)
    with pytest.raises(FormalConfigError):
        validate_formal_config(formal_config)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "path",
    [
        "model.latent_dim",
        "metadata_conditioning.dimension",
        "model.geometry.interpolation_sigma_mm",
        "model.geometry.residual_scale",
        "model.geometry.embedding.initial_scale",
        "model.predictor.mlp_ratio",
        "model.predictor.drop_path",
        "metadata_conditioning.dropout_probability",
    ],
)
def test_nonfinite_controls_raise_config_errors(
    formal_config: dict[str, Any], path: str, value: float
) -> None:
    _set(formal_config, path, value)
    with pytest.raises(FormalConfigError):
        validate_formal_config(formal_config)


@pytest.mark.parametrize("query_count", [1, 256, 512])
def test_query_limit_is_configurable_with_matching_mask_contract(
    formal_config: dict[str, Any], query_count: int
) -> None:
    formal_config["model"]["predictor"]["max_queries"] = query_count
    formal_config["masking"]["query_tokens"] = query_count
    validate_formal_config(formal_config)


def test_runtime_controls_and_zero_geometry_initial_scale_are_configurable(
    formal_config: dict[str, Any],
) -> None:
    formal_config["model"]["sat3d"]["activation_checkpointing"] = False
    formal_config["preprocessing"]["registration"]["threads_per_worker"] = 2
    formal_config["model"]["geometry"]["geomc_blocks"] = 3
    formal_config["model"]["geometry"]["embedding"]["initial_scale"] = 0.0
    formal_config["training"]["checkpoint"].update(
        keep_last=5, keep_every_coverages=10
    )
    formal_config["validation"]["full_every_coverages"] = 10
    formal_config["training"].update(
        microbatch_size=4, gradient_accumulation=2, global_batch_size=8
    )
    validate_formal_config(formal_config)


def test_cache_contracts_do_not_change_when_only_model_widths_change(
    formal_config: dict[str, Any],
) -> None:
    from mri_pet_geomc.formal.workflows import (
        _geometry_contract_digest,
        _preprocessing_contract,
    )

    wide = _wide_config(formal_config)
    reference_contract = "unchanged-reference-contract"
    assert _preprocessing_contract(wide) == _preprocessing_contract(formal_config)
    assert _geometry_contract_digest(wide, reference_contract) == (
        _geometry_contract_digest(formal_config, reference_contract)
    )
    # This is deliberately not a license to reuse a genuinely different FEM.
    wide["model"]["geometry"]["modes"] += 1
    assert _geometry_contract_digest(wide, reference_contract) != (
        _geometry_contract_digest(formal_config, reference_contract)
    )


def test_old_128_checkpoint_cannot_strictly_resume_a_384_projector() -> None:
    from mri_pet_geomc.formal.model.projector import ContentGatedTokenProjector

    old = ContentGatedTokenProjector(input_dim=384, latent_dim=128, condition_dim=8)
    wide = ContentGatedTokenProjector(input_dim=384, latent_dim=384, condition_dim=8)
    assert tuple(old.projection.weight.shape) == (128, 384)
    assert tuple(wide.projection.weight.shape) == (384, 384)
    with pytest.raises(RuntimeError, match="size mismatch"):
        wide.load_state_dict(old.state_dict(), strict=True)


@pytest.mark.parametrize("output_dim", [None, "auto"])
def test_projector_can_infer_its_output_width(
    formal_config: dict[str, Any], output_dim: Any,
) -> None:
    formal_config["model"]["latent_dim"] = 384
    formal_config["model"]["projector"]["output_dim"] = output_dim
    validate_formal_config(formal_config)
    del formal_config["model"]["projector"]["output_dim"]
    validate_formal_config(formal_config)


def test_zero_warmup_is_a_valid_configured_schedule(formal_config: dict[str, Any]) -> None:
    formal_config["training"]["scheduler"]["warmup_coverages"] = 0
    validate_formal_config(formal_config)


def test_nondefault_batch_preserves_relation_curriculum_semantics() -> None:
    from mri_pet_geomc.formal.runtime.schedule import (
        BatchContract,
        FormalTrainingSchedule,
        RelationKind,
    )

    schedule = FormalTrainingSchedule(
        total_coverages=50,
        batch=BatchContract(micro_batch_anchors=4, gradient_accumulation_steps=2),
    )
    assert schedule.batch.global_batch_anchors == 8
    assert schedule.contract()["global_batch_anchors"] == 8
    first = schedule.relation_mixture(1).as_dict()
    assert first[RelationKind.SELF_ONLY] == 1.0
    ramp = schedule.relation_mixture(3).as_dict()
    assert ramp[RelationKind.SELF_ONLY] == pytest.approx(0.875)
    assert ramp[RelationKind.SAME_SESSION_CROSS_SEQUENCE] == pytest.approx(0.075)
    assert schedule.relation_mixture(6) == schedule.fixed_mixture


def test_relation_mixture_remains_fixed_until_runtime_is_config_driven(
    formal_config: dict[str, Any],
) -> None:
    mixture = formal_config["training"]["relation_schedule"]["steady_mixture"]
    mixture.update(self_only=0.60, same_session_cross_sequence=0.20)
    with pytest.raises(FormalConfigError):
        validate_formal_config(formal_config)


@pytest.mark.parametrize(
    "path,value",
    [
        ("model.geometry.nodes_target", 1),
        ("model.geometry.modes", 1),
        ("model.geometry.modes", 1024),
        ("model.geometry.resolvent_radii", [6.0, 3.0]),
        ("model.geometry.resolvent_radii", [3.0, 3.0]),
        ("model.predictor.mlp_ratio", 1.0e-10),
        ("training.relation_schedule.steady_mixture.self_only", float("nan")),
    ],
)
def test_structural_errors_are_reported_before_the_real_builder(
    formal_config: dict[str, Any], path: str, value: Any,
) -> None:
    _set(formal_config, path, value)
    with pytest.raises(FormalConfigError):
        validate_formal_config(formal_config)
