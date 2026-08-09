from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from mri_pet_geomc.config import (
    REQUIRED_EXPERIMENT_SETTINGS,
    METHOD_NAME,
    ConfigurationError,
    load_config,
    validate_config,
)
from mri_pet_geomc.pipeline import build_plan, plan_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = ("smoke.yaml", "local_cuda_smoke.yaml", "server.yaml")


def _config(name: str = "smoke.yaml") -> dict:
    return load_config(PROJECT_ROOT / "configs" / name, project_root=PROJECT_ROOT)


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_all_public_configs_resolve_to_one_main_model(name: str) -> None:
    config = _config(name)

    assert config["schema_version"] == 1
    assert config["project"]["method"] == METHOD_NAME
    assert len(config["experiments"]) == 1
    experiment = config["experiments"][0]
    for key, expected in REQUIRED_EXPERIMENT_SETTINGS.items():
        assert experiment[key] == expected
    assert config["evaluation"]["primary_experiment_id"] == "geomc"
    assert config["evaluation"]["comparisons"] == []


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_plan_contains_exactly_one_training_stage(name: str) -> None:
    config = _config(name)
    plan = build_plan(config)
    training = [stage for stage in plan if stage.stage_id.startswith("08.experiment.")]

    assert [stage.stage_id for stage in training] == ["08.experiment.geomc.seed0"]
    assert training[0].params["experiment"]["id"] == "geomc"
    assert plan_payload(plan)["schema_version"] == 1


def test_unknown_top_level_section_and_experiment_keys_are_rejected() -> None:
    top_level = deepcopy(_config())
    top_level["model_family"] = "geomc"
    with pytest.raises(ConfigurationError, match="Unknown top-level"):
        validate_config(top_level)

    section = deepcopy(_config())
    section["model"]["layers"] = 3
    with pytest.raises(ConfigurationError, match="Unknown model"):
        validate_config(section)

    experiment = deepcopy(_config())
    experiment["experiments"][0]["family"] = "geomc"
    with pytest.raises(ConfigurationError, match="Unknown experiment"):
        validate_config(experiment)


def test_main_experiment_id_is_fixed() -> None:
    config = deepcopy(_config())
    config["experiments"][0]["id"] = "another-model"
    with pytest.raises(ConfigurationError, match="Main experiment field"):
        validate_config(config)


def test_exactly_one_experiment_is_required() -> None:
    missing = deepcopy(_config())
    missing["experiments"] = []
    with pytest.raises(ConfigurationError, match="exactly one"):
        validate_config(missing)

    duplicate = deepcopy(_config())
    duplicate["experiments"].append(deepcopy(duplicate["experiments"][0]))
    with pytest.raises(ConfigurationError, match="exactly one"):
        validate_config(duplicate)


def test_single_model_requires_empty_comparisons() -> None:
    config = deepcopy(_config())
    config["evaluation"]["comparisons"] = [["geomc", "geomc"]]
    with pytest.raises(ConfigurationError, match="comparisons must be empty"):
        validate_config(config)


def test_unknown_method_is_rejected_by_config_and_plan() -> None:
    config = deepcopy(_config())
    config["project"]["method"] = "another_method"
    with pytest.raises(ConfigurationError, match="project.method"):
        validate_config(config)
    with pytest.raises(ValueError, match="implements only"):
        build_plan(config)


def test_schema_and_protocol_versions_are_not_model_rename_targets() -> None:
    config = deepcopy(_config())
    config["schema_version"] = 2
    with pytest.raises(ConfigurationError, match="schema_version"):
        validate_config(config)

    assert _config()["data"]["input_normalization"] == "brain_mask_zscore_v1"
    assert _config()["registration_qc"]["policy"] == "automatic_nonblocking_v1"
