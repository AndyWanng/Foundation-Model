from __future__ import annotations

from pathlib import Path

import pytest

from mri_pet_geomc import stages
from mri_pet_geomc.pipeline import StageContext, StageSpec
from mri_pet_geomc.utils import EventLogger, atomic_write_json, read_json


def _experiment(*, max_updates: int = 12) -> dict:
    return {
        "id": "geomc",
        "verified_pair_retention_fraction": 1.0,
        "pairing_mode": "verified",
        "seeds": [0],
        "max_updates": max_updates,
        "planned_total_updates": max_updates,
    }


def _config(*, comparisons: list | None = None) -> dict:
    return {
        "project": {
            "method": "shared_sat3d_geomc",
            "scientific_use": False,
            "seed": 20260809,
        },
        "run": {"mode": "smoke"},
        "evaluation": {
            "primary_experiment_id": "geomc",
            "comparisons": [] if comparisons is None else comparisons,
        },
        "experiments": [_experiment()],
    }


def _context(
    project: Path,
    launch: Path,
    stage_id: str,
    handler: str,
    config: dict,
    *,
    params: dict | None = None,
) -> StageContext:
    stage_dir = launch / "stages" / stage_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        project_root=project,
        launch_dir=launch,
        stage_dir=stage_dir,
        config=config,
        spec=StageSpec(stage_id, handler, params=params or {}),
        event_logger=EventLogger(launch / "logs" / "events.jsonl"),
    )


def test_main_experiment_is_the_single_main_model() -> None:
    experiment = stages._main_experiment(_config())
    assert experiment == _experiment()
    assert experiment["id"] == "geomc"
    assert experiment["pairing_mode"] == "verified"


def test_training_failure_is_materialized_without_blocking_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    launch = project / "artifacts" / "launch"
    project.mkdir()
    config = _config()
    experiment = config["experiments"][0]

    monkeypatch.setattr(stages, "_load_tables", lambda _: object())
    monkeypatch.setattr(stages, "_model_geometry", lambda _ctx, _tables: object())

    def fail_training(*_args, **_kwargs):
        raise RuntimeError("deliberate main training failure")

    monkeypatch.setattr(stages, "run_training", fail_training)
    context = _context(
        project,
        launch,
        "08.experiment.geomc.seed0",
        "geomc_train",
        config,
        params={"experiment": experiment, "seed": 0},
    )
    result = stages.geomc_train(context)

    assert result.metrics["execution_succeeded"] == 0.0
    payload = read_json(context.stage_dir / "experiment_result.json")
    assert payload["status"] == "execution_failed_nonblocking"
    assert payload["kind"] == "experiment"
    assert payload["error"]["type"] == "RuntimeError"
    assert payload["experiment"]["id"] == "geomc"


def test_single_model_evaluation_records_ema_target_and_fixed_reference_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    launch = project / "artifacts" / "launch"
    project.mkdir()
    config = _config()
    training_dir = launch / "stages" / "08.experiment.geomc.seed0"
    checkpoint = training_dir / "training" / "final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-placeholder")
    atomic_write_json(
        training_dir / "experiment_result.json",
        {
            "schema_version": 2,
            "status": "completed",
            "execution_succeeded": True,
            "final_checkpoint": str(checkpoint),
            "final_step": 12,
        },
    )

    tables = object()
    geometry = object()
    reference_targets = {"frozen": object()}
    monkeypatch.setattr(stages, "_load_tables", lambda _: tables)
    monkeypatch.setattr(stages, "_model_geometry", lambda _ctx, _tables: geometry)
    monkeypatch.setattr(
        stages,
        "build_fixed_reference_targets",
        lambda *_args, **_kwargs: reference_targets,
    )

    def evaluate(*args, **kwargs):
        assert args[0] is tables
        assert args[1] is geometry
        assert kwargs["checkpoint_path"] == checkpoint.resolve()
        assert kwargs["fixed_reference_targets"] is reference_targets
        assert kwargs["split"] == "test"
        return {
            "ema_target_metrics": {
                "loss": 0.20,
                "mae": 0.10,
            },
            "fixed_reference": {
                "fixed_reference_aligned_huber_loss": 0.30,
                "fixed_reference_unaligned_huber_loss": 0.40,
            },
            "model_specification": {
                "name": "shared_sat3d_geomc"
            },
        }

    monkeypatch.setattr(stages, "evaluate_checkpoint", evaluate)
    context = _context(project, launch, "09.evaluate", "geomc_evaluate", config)
    result = stages.geomc_evaluate(context)
    payload = read_json(context.stage_dir / "evaluation.json")

    assert result.metrics["execution_succeeded"] == 1.0
    assert payload["status"] == "complete"
    assert payload["single_model"] is True
    assert payload["comparisons"] == []
    assert list(payload["results"]) == ["geomc"]
    assert payload["results"]["geomc"]["checkpoint_contract"] == {
        "policy": "fixed_final_checkpoint",
        "actual_step": 12,
        "planned_step": 12,
        "matches_planned_updates": True,
    }
    assert payload["absolute_metrics"] == {
        "ema_target_huber_loss": 0.20,
        "ema_target_mean_absolute_error": 0.10,
        "fixed_reference_aligned_huber_loss": 0.30,
        "fixed_reference_unaligned_huber_loss": 0.40,
    }
    assert payload["metric_spaces"]["ema_target"]["cross_model_comparable"] is False
    assert payload["metric_spaces"]["fixed_reference"]["available"] is True


def test_single_model_evaluation_preserves_training_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    launch = project / "artifacts" / "launch"
    project.mkdir()
    config = _config()
    training_dir = launch / "stages" / "08.experiment.geomc.seed0"
    atomic_write_json(
        training_dir / "experiment_result.json",
        {
            "schema_version": 1,
            "status": "execution_failed_nonblocking",
            "execution_succeeded": False,
            "error": {"type": "RuntimeError", "message": "training failed"},
        },
    )
    monkeypatch.setattr(stages, "_load_tables", lambda _: object())
    monkeypatch.setattr(stages, "_model_geometry", lambda _ctx, _tables: object())
    monkeypatch.setattr(
        stages,
        "build_fixed_reference_targets",
        lambda *_args, **_kwargs: {},
    )

    context = _context(project, launch, "09.evaluate", "geomc_evaluate", config)
    result = stages.geomc_evaluate(context)
    payload = read_json(context.stage_dir / "evaluation.json")

    assert result.metrics["execution_succeeded"] == 0.0
    assert payload["status"] == "evaluation_failed"
    assert payload["comparisons"] == []
    assert (
        payload["results"]["geomc"]["status"]
        == "not_evaluated_training_failed"
    )


def test_evaluation_rejects_comparison_arms(tmp_path: Path) -> None:
    project = tmp_path / "project"
    launch = project / "artifacts" / "launch"
    project.mkdir()
    config = _config(comparisons=[["geomc", "another-model"]])
    context = _context(project, launch, "09.evaluate", "geomc_evaluate", config)
    with pytest.raises(RuntimeError, match="comparisons=\\[\\]"):
        stages.geomc_evaluate(context)


def _write_complete_report_inputs(launch: Path) -> None:
    atomic_write_json(
        launch
        / "stages"
        / "01.observations"
        / "tables"
        / "observation_summary.json",
        {
            "schema_version": 1,
            "subject_count": 8,
            "observation_count": 16,
            "verified_pair_count": 8,
        },
    )
    atomic_write_json(
        launch / "stages" / "03.registration_qc" / "automatic_registration_qc.json",
        {"schema_version": 1, "status": "completed"},
    )
    atomic_write_json(
        launch / "stages" / "04.geometry" / "geometry_summary.json",
        {"schema_version": 1, "status": "completed"},
    )
    atomic_write_json(
        launch / "stages" / "05.model_check" / "model_check.json",
        {
            "schema_version": 1,
            "status": "completed",
            "execution_succeeded": True,
            "supports_scientific_claims": False,
            "model_check_scope": "single aggregated observation field and GeoMC core",
        },
    )
    atomic_write_json(
        launch / "stages" / "06.end_to_end_smoke_test" / "smoke_test.json",
        {
            "schema_version": 1,
            "status": "completed",
            "execution_succeeded": True,
            "loss": 0.50,
            "model_specification": {
                "name": "shared_sat3d_geomc"
            },
        },
    )
    atomic_write_json(
        launch / "stages" / "07.small_sample_overfit_check" / "small_sample_overfit_check.json",
        {
            "schema_version": 1,
            "status": "completed",
            "execution_succeeded": True,
            "relative_drop": 0.25,
        },
    )
    atomic_write_json(
        launch / "stages" / "08.experiment.geomc.seed0" / "experiment_result.json",
        {
            "schema_version": 2,
            "status": "completed",
            "execution_succeeded": True,
            "final_step": 12,
            "best_validation_loss": 0.22,
            "final_checkpoint": "final.pt",
        },
    )
    atomic_write_json(
        launch / "stages" / "09.evaluate" / "evaluation.json",
        {
            "schema_version": 1,
            "status": "complete",
            "execution_succeeded": True,
            "primary_experiment_id": "geomc",
            "single_model": True,
            "comparisons": [],
            "results": {
                "geomc": {
                    "status": "evaluated",
                    "model_specification": {
                        "name": "shared_sat3d_geomc"
                    },
                    "ema_target_metrics": {"loss": 0.20, "mae": 0.10},
                    "fixed_reference": {
                        "fixed_reference_aligned_huber_loss": 0.30
                    },
                }
            },
            "absolute_metrics": {
                "ema_target_huber_loss": 0.20,
                "ema_target_mean_absolute_error": 0.10,
                "fixed_reference_aligned_huber_loss": 0.30,
                "fixed_reference_unaligned_huber_loss": 0.40,
            },
            "shared_infrastructure_errors": [],
        },
    )


def test_report_emits_execution_absolute_metrics_and_interpretation_limits(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    launch = project / "artifacts" / "launch"
    project.mkdir()
    _write_complete_report_inputs(launch)

    context = _context(project, launch, "10.report", "geomc_report", _config())
    result = stages.geomc_report(context)
    report = read_json(context.stage_dir / "feasibility_report.json")
    markdown = (context.stage_dir / "feasibility_report.md").read_text(
        encoding="utf-8"
    )

    assert result.metrics["training_executed"] == 1.0
    assert result.metrics["evaluation_executed"] == 1.0
    assert report["decision"] == "SINGLE_MODEL_EVALUATION_COMPLETE"
    assert report["method"] == "shared_sat3d_geomc"
    assert report["experiment_id"] == "geomc"
    assert report["single_model"] is True
    assert report["comparisons"] == []
    assert report["absolute_metrics"]["ema_target_huber_loss"] == 0.20
    assert report["absolute_metrics"]["fixed_reference_aligned_huber_loss"] == 0.30
    assert report["model_check"]["supports_scientific_claims"] is False
    assert any(
        item["claim"] == "relative_improvement"
        and item["status"] == "not_identified"
        for item in report["interpretation_limits"]
    )
    assert "Performance improvement over another architecture." in report[
        "claims_not_allowed"
    ]
    assert "no comparison arms" in markdown


def test_report_remains_available_when_upstream_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    launch = project / "artifacts" / "launch"
    project.mkdir()

    context = _context(project, launch, "10.report", "geomc_report", _config())
    result = stages.geomc_report(context)
    report = read_json(context.stage_dir / "feasibility_report.json")

    assert result.metrics["training_executed"] == 0.0
    assert result.metrics["evaluation_executed"] == 0.0
    assert report["decision"] == "EXECUTION_INCOMPLETE"
    assert report["comparisons"] == []
    assert report["absolute_metrics"] == {
        "ema_target_huber_loss": None,
        "ema_target_mean_absolute_error": None,
        "fixed_reference_aligned_huber_loss": None,
        "fixed_reference_unaligned_huber_loss": None,
    }
    assert set(report["artifact_status"]) == {
        "observations",
        "registration_qc",
        "geometry",
        "model_check",
        "smoke_test",
        "small_sample_overfit_check",
        "training",
        "evaluation",
    }
    assert all(
        value["status"] == "unavailable"
        for value in report["artifact_status"].values()
    )
