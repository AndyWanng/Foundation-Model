from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import mri_pet_geomc.formal.runtime.checkpoint as checkpoint_module

from mri_pet_geomc.formal.runtime import (
    AtomicCheckpointManager,
    CheckpointCursor,
    Microbatch,
    ModelContractError,
    NonFiniteMetricError,
    PickleSerializer,
    ResourceSnapshot,
    ResumeMismatchError,
    RunJournal,
    TelemetryRecorder,
    TerminalProgress,
    TrainingRuntimeController,
    TrainingStepResult,
)


class _Stateful:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> Mapping[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state_dict: Mapping[str, Any], **_: Any) -> None:
        self.value = int(state_dict["value"])


class _Resources:
    def sample(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            cpu_rss_gib=1.25,
            gpu_allocated_gib=12.0,
            gpu_reserved_gib=16.0,
            gpu_peak_allocated_gib=13.0,
            gpu_peak_reserved_gib=17.0,
        )


@dataclass
class _Backend:
    model: _Stateful
    optimizer: _Stateful
    ema: _Stateful
    scheduler: _Stateful | None = None
    scaler: _Stateful | None = None

    def extra_checkpoint_stateful(self) -> Mapping[str, _Stateful]:
        return {}

    def train_accumulated_step(
        self,
        microbatches: Any,
        *,
        coverage: int,
        relation_mixture: Any,
        expected_global_batch_anchors: int,
    ) -> TrainingStepResult:
        del coverage, relation_mixture
        anchors = sum(item.anchor_count for item in microbatches)
        assert expected_global_batch_anchors == 16
        return TrainingStepResult(
            anchors_processed=anchors,
            next_anchor_index=microbatches[-1].next_anchor_index,
            elapsed_seconds=2.0,
            loader_wait_seconds=sum(item.loader_wait_seconds for item in microbatches),
            loss_total=1.0,
            modality_losses={"mri": 1.0, "pet": 1.0},
            modality_counts={"mri": 8, "pet": 8},
            relation_losses={"total": 0.0},
            relation_counts={"self_only": 16},
            learning_rates={"main": 1.0e-4},
            ema_decay=0.996,
            grad_norm=1.0,
        )

    def validate(self, request: Any) -> Mapping[str, Any]:
        return {"loss": 999.0, "coverage": request.coverage}


def test_atomic_checkpoint_restores_all_required_state_rng_and_cursor(tmp_path: Path) -> None:
    model = _Stateful(1)
    optimizer = _Stateful(2)
    ema = _Stateful(3)
    manager = AtomicCheckpointManager(tmp_path, serializer=PickleSerializer())
    cursor = CheckpointCursor(
        epoch=7,
        coverage=7,
        next_anchor_index=320,
        global_step=40,
        optimizer_step=20,
    )
    random.seed(123)
    manager.save(
        cursor=cursor,
        hashes={"manifest": "m", "config": "c", "split": "s"},
        model=model,
        optimizer=optimizer,
        ema=ema,
        metrics={"loss.total": 1.5},
    )
    expected_next_random = random.random()
    model.value = optimizer.value = ema.value = -1
    random.seed(999)

    loaded = manager.load_latest(
        expected_hashes={"manifest": "m", "config": "c", "split": "s"},
        model=model,
        optimizer=optimizer,
        ema=ema,
    )

    assert loaded.cursor == cursor
    assert (model.value, optimizer.value, ema.value) == (1, 2, 3)
    assert random.random() == expected_next_random
    with pytest.raises(ResumeMismatchError):
        manager.load_latest(
            expected_hashes={"manifest": "changed", "config": "c", "split": "s"},
            model=model,
            optimizer=optimizer,
            ema=ema,
        )
    loaded.path.write_bytes(loaded.path.read_bytes() + b"tamper")
    with pytest.raises(ResumeMismatchError, match="checksum mismatch"):
        manager.load_latest(
            expected_hashes={"manifest": "m", "config": "c", "split": "s"},
            model=model,
            optimizer=optimizer,
            ema=ema,
        )


def test_same_cursor_commit_cannot_clobber_latest_target_on_pointer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Stateful(1)
    optimizer = _Stateful(2)
    ema = _Stateful(3)
    manager = AtomicCheckpointManager(tmp_path, serializer=PickleSerializer())
    cursor = CheckpointCursor(
        epoch=4,
        coverage=4,
        next_anchor_index=24,
        global_step=4,
        optimizer_step=2,
    )
    hashes = {"manifest": "m", "config": "c"}
    first = manager.save(
        cursor=cursor,
        hashes=hashes,
        model=model,
        optimizer=optimizer,
        ema=ema,
    )
    pointer_before = manager.latest_path.read_bytes()
    first_before = first.read_bytes()

    model.value = 11
    optimizer.value = 12
    ema.value = 13
    first_commit = first.stem.rsplit("-k", maxsplit=1)[1]
    commit_ids = iter((first_commit, "f" * 32))
    monkeypatch.setattr(
        checkpoint_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=next(commit_ids)),
    )

    def fail_pointer_update(*_: Any, **__: Any) -> None:
        raise RuntimeError("simulated pointer commit interruption")

    monkeypatch.setattr(manager, "_atomic_write_json", fail_pointer_update)
    with pytest.raises(RuntimeError, match="pointer commit interruption"):
        manager.save(
            cursor=cursor,
            hashes=hashes,
            model=model,
            optimizer=optimizer,
            ema=ema,
        )

    assert manager.latest_path.read_bytes() == pointer_before
    assert first.read_bytes() == first_before
    commits = sorted(tmp_path.glob("formal-training-c004-s000000002-*.pkl"))
    assert len(commits) == 2
    assert all("-pstep-k" in path.name for path in commits)
    assert any(("-k" + "f" * 32) in path.name for path in commits)

    model.value = optimizer.value = ema.value = -1
    loaded = manager.load_latest(
        expected_hashes=hashes,
        model=model,
        optimizer=optimizer,
        ema=ema,
        restore_rng=False,
    )
    assert loaded.path == first.resolve()
    assert (model.value, optimizer.value, ema.value) == (1, 2, 3)


def test_legacy_checkpoint_name_remains_restore_and_retention_compatible(
    tmp_path: Path,
) -> None:
    model = _Stateful(1)
    optimizer = _Stateful(2)
    ema = _Stateful(3)
    manager = AtomicCheckpointManager(tmp_path, serializer=PickleSerializer())
    hashes = {"manifest": "m", "config": "c"}
    first_cursor = CheckpointCursor(1, 1, 16, 2, 1)
    modern = manager.save(
        cursor=first_cursor,
        hashes=hashes,
        model=model,
        optimizer=optimizer,
        ema=ema,
    )
    legacy = tmp_path / "formal-training-c001-s000000001.pkl"
    modern.replace(legacy)
    pointer = json.loads(manager.latest_path.read_text(encoding="utf-8"))
    pointer["checkpoint_name"] = legacy.name
    manager._atomic_write_json(manager.latest_path, pointer)

    model.value = optimizer.value = ema.value = -1
    loaded = manager.load_latest(
        expected_hashes=hashes,
        model=model,
        optimizer=optimizer,
        ema=ema,
        restore_rng=False,
    )
    assert loaded.path == legacy.resolve()
    assert (model.value, optimizer.value, ema.value) == (1, 2, 3)

    second_cursor = CheckpointCursor(2, 2, 32, 4, 2)
    manager.save(
        cursor=second_cursor,
        hashes=hashes,
        model=model,
        optimizer=optimizer,
        ema=ema,
    )
    assert manager.prune(
        keep_last=1,
        keep_every_coverages=None,
        keep_names=(legacy.name,),
    ) == ()
    removed = manager.prune(keep_last=1, keep_every_coverages=None)
    assert removed == (legacy.resolve(),)


def test_telemetry_logs_all_requested_fields_and_nan_is_fatal(tmp_path: Path) -> None:
    stream = io.StringIO()
    with RunJournal(
        tmp_path / "run.log", tmp_path / "metrics.jsonl", run_id="run"
    ) as journal:
        progress = TerminalProgress(
            journal=journal,
            stream=stream,
            force_plain=True,
            plain_update_every=1,
        )
        progress.begin_stage("training", total=32, unit="anchors")
        recorder = TelemetryRecorder(
            journal=journal,
            progress=progress,
            resource_provider=_Resources(),
        )
        telemetry = recorder.record(
            coverage=1,
            global_step=2,
            optimizer_step=1,
            next_anchor_index=16,
            anchors_processed=16,
            elapsed_seconds=4.0,
            loader_wait_seconds=0.25,
            loss_total=1.0,
            loss_components={"self": 0.8, "relation": 0.2},
            modality_losses={"mri": 0.9, "pet": 1.1},
            modality_counts={"mri": 10, "pet": 6},
            relation_losses={"total": 0.2, "self_only": 0.8},
            relation_counts={"self_only": 12, "same_session_cross_sequence": 4},
            learning_rates={"predictor": 2.0e-4, "encoder": 1.0e-5},
            ema_decay=0.996,
            grad_norm=2.5,
            progress_completed=16,
        )
        assert telemetry.throughput_anchors_s == 4.0
        assert telemetry.resources.gpu_reserved_gib == 16.0
        with pytest.raises(NonFiniteMetricError):
            recorder.record(
                coverage=1,
                global_step=4,
                optimizer_step=2,
                next_anchor_index=32,
                anchors_processed=16,
                elapsed_seconds=4.0,
                loader_wait_seconds=0.1,
                loss_total=float("nan"),
                loss_components={},
                modality_losses={},
                modality_counts={},
                relation_losses={},
                relation_counts={},
                learning_rates={"predictor": 2.0e-4},
                ema_decay=0.996,
                grad_norm=1.0,
            )

    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    step = next(row for row in rows if row["event"] == "training_optimizer_step")
    metrics = step["metrics"]
    assert metrics["loss.modality.mri"] == 0.9
    assert metrics["loss.modality.pet"] == 1.1
    assert metrics["loss.relation.total"] == 0.2
    assert metrics["cpu_rss_gib"] == 1.25
    assert metrics["gpu_allocated_gib"] == 12.0
    assert metrics["gpu_reserved_gib"] == 16.0


def test_training_controller_has_fixed_batch_and_observational_validation(
    tmp_path: Path,
) -> None:
    backend = _Backend(_Stateful(1), _Stateful(2), _Stateful(3))
    stream = io.StringIO()
    with RunJournal(
        tmp_path / "run.log", tmp_path / "metrics.jsonl", run_id="run"
    ) as journal:
        progress = TerminalProgress(
            journal=journal,
            stream=stream,
            force_plain=True,
            plain_update_every=1,
        )
        telemetry = TelemetryRecorder(
            journal=journal,
            progress=progress,
            resource_provider=_Resources(),
        )
        controller = TrainingRuntimeController(
            run_id="run",
            hashes={"manifest": "m", "config": "c"},
            journal=journal,
            progress=progress,
            telemetry=telemetry,
            checkpoints=AtomicCheckpointManager(
                tmp_path / "checkpoints",
                serializer=PickleSerializer(),
                journal=journal,
            ),
        )
        controller.initialize(resume_requested=False)
        mixture = controller.begin_coverage(1, total_anchors=18)
        assert mixture.self_only == 1.0
        cursor = CheckpointCursor(
            epoch=1,
            coverage=1,
            next_anchor_index=16,
            global_step=2,
            optimizer_step=1,
        )
        result = controller.train_accumulated_step(
            backend=backend,
            microbatches=[
                Microbatch(None, anchor_count=8, next_anchor_index=8, loader_wait_seconds=0.05),
                Microbatch(None, anchor_count=8, next_anchor_index=16, loader_wait_seconds=0.05),
            ],
            coverage=1,
        )
        controller.record_optimizer_step(cursor=cursor, result=result)
        request = controller.validation_requests(coverage=1, optimizer_step=1)[0]
        assert controller.run_validation(backend=backend, request=request)["loss"] == 999.0
        assert controller.checkpoint_due(optimizer_step=1, end_of_coverage=True)
        with pytest.raises(ModelContractError):
            controller.record_optimizer_step(
                cursor=cursor,
                result=TrainingStepResult(
                    anchors_processed=8,
                    next_anchor_index=16,
                    elapsed_seconds=1.0,
                    loader_wait_seconds=0.0,
                    loss_total=1.0,
                ),
            )

    text = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"no_performance_early_stop":true' in text
    assert '"affects_training_continuation":false' in text
