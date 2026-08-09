from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from mri_pet_geomc.pipeline import (
    PipelineRunner,
    StageContext,
    StageResult,
    launch_lock,
    verify_launch,
)


def _hold_launch_lock(
    launch_dir: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with launch_lock(Path(launch_dir)):
        ready.set()
        if not release.wait(timeout=10):
            raise RuntimeError("Parent did not release lock-holder process")


def test_legacy_plain_lock_file_does_not_block_resume(tmp_path: Path) -> None:
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / ".pipeline.lock").write_text(
        '{"pid":999999,"hostname":"retired-node"}\n', encoding="utf-8"
    )

    with launch_lock(launch):
        assert (launch / ".pipeline.lock.owner.json").is_file()


def test_live_concurrent_launch_lock_fails_closed(tmp_path: Path) -> None:
    launch = tmp_path / "launch"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_launch_lock,
        args=(str(launch), ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=10), "Lock-holder process did not become ready"
        with pytest.raises(RuntimeError, match="Launch lock is active"):
            with launch_lock(launch):
                pass
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_failed_experiment_does_not_block_evaluate_or_report(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    launch = project / "artifacts" / "launch"
    experiment = {
        "id": "deliberate-failure",
        "verified_pair_retention_fraction": 1.0,
        "pairing_mode": "verified",
        "seeds": [0],
        "max_updates": 1,
    }
    config = {
        "project": {"method": "shared_sat3d_geomc"},
        "execution": {"policy": "exhaustive_nonblocking_v1"},
        "experiments": [experiment],
        "runtime": {"project_root": str(project.resolve())},
    }
    executed: list[str] = []

    def complete(ctx: StageContext) -> StageResult:
        executed.append(ctx.spec.stage_id)
        output = ctx.stage_dir / "output.json"
        output.write_text('{"schema_version":1}\n', encoding="utf-8")
        return StageResult(outputs=[output])

    def fail(_: StageContext) -> StageResult:
        raise RuntimeError("deliberate independent-arm failure")

    handlers = {
        "geomc_preflight": complete,
        "geomc_observations": complete,
        "geomc_asset_snapshot": complete,
        "geomc_registration_qc": complete,
        "geomc_geometry": complete,
        "geomc_model_check": complete,
        "geomc_smoke_test": complete,
        "geomc_small_sample_overfit_check": complete,
        "geomc_train": fail,
        "geomc_evaluate": complete,
        "geomc_report": complete,
    }
    runner = PipelineRunner(
        project_root=project,
        launch_dir=launch,
        config=config,
        handlers=handlers,
    )
    state = runner.run()
    failed_stage = state["stages"]["08.experiment.deliberate-failure.seed0"]
    assert state["status"] == "completed"
    assert failed_stage["status"] == "completed"
    assert failed_stage["nonblocking_failure"] is True
    assert "09.evaluate" in executed
    assert "10.report" in executed
    verification = verify_launch(launch)
    assert verification["artifact_integrity_passed"] is True
    assert verification["pipeline_exhausted_plan"] is True
    assert verification["exhaustive_execution_completed"] is True
    assert verification["all_requested_computations_succeeded"] is False
    assert verification["passed"] is False
    assert verification["computation_failures"] == [
        "requested computation failed: 08.experiment.deliberate-failure.seed0"
    ]
