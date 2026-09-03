from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from mri_pet_geomc.formal.runtime import (
    AcceptableCaseError,
    AtomicStateStore,
    CompletedRanges,
    NonFiniteMetricError,
    PreprocessResult,
    PreprocessingResumeState,
    PreprocessingRunner,
    ResumeMismatchError,
    RunJournal,
    TerminalProgress,
)


def test_completed_ranges_merge_and_round_trip() -> None:
    ranges = CompletedRanges().add(2).add(0).add(1).add(4)

    assert ranges.ranges == ((0, 2), (4, 4))
    assert ranges.count == 4
    assert ranges.first_pending(6) == 3
    assert CompletedRanges.from_json(ranges.as_json()) == ranges


def test_atomic_state_rejects_hash_drift(tmp_path: Path) -> None:
    store = AtomicStateStore(tmp_path / "resume.json")
    state = PreprocessingResumeState(
        run_id="run",
        total_cases=2,
        hashes={"manifest": "abc", "config": "def"},
    )
    store.save(state.to_resume_state())

    loaded = store.load(
        expected_kind="preprocessing",
        expected_run_id="run",
        expected_hashes={"manifest": "abc", "config": "def"},
    )
    assert PreprocessingResumeState.from_resume_state(loaded).total_cases == 2
    with pytest.raises(ResumeMismatchError):
        store.load(expected_hashes={"manifest": "changed", "config": "def"})


@dataclass
class _Callbacks:
    fail_once_at: int | None = None
    acceptable_at: int | None = None

    def case_id(self, item: str) -> str:
        return item

    def process(self, item: str, *, index: int) -> PreprocessResult:
        if self.fail_once_at == index:
            self.fail_once_at = None
            raise RuntimeError("simulated interruption")
        if self.acceptable_at == index:
            raise AcceptableCaseError("declared unreadable case", code="declared_skip")
        return PreprocessResult(metrics={"voxels": 8.0}, outputs={"cache": item + ".npy"})


def _runner(tmp_path: Path, stream: io.StringIO) -> tuple[PreprocessingRunner[str], RunJournal]:
    journal = RunJournal(
        tmp_path / "run.log",
        tmp_path / "metrics.jsonl",
        run_id="run",
    )
    progress = TerminalProgress(
        journal=journal,
        stream=stream,
        force_plain=True,
        plain_update_every=1,
    )
    runner = PreprocessingRunner(
        state_store=AtomicStateStore(tmp_path / "resume.json"),
        journal=journal,
        progress=progress,
    )
    return runner, journal


def test_run_journal_sequence_remains_monotonic_across_resume(tmp_path: Path) -> None:
    text_path = tmp_path / "resume.log"
    metrics_path = tmp_path / "resume.jsonl"
    with RunJournal(text_path, metrics_path, run_id="stable-run") as journal:
        assert journal.event("first")["sequence"] == 1
        assert journal.event("second")["sequence"] == 2
    with RunJournal(text_path, metrics_path, run_id="stable-run") as journal:
        assert journal.event("resumed")["sequence"] == 3


def test_run_journal_truncates_partial_tail_before_append_and_records_recovery(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "resume.log"
    metrics_path = tmp_path / "resume.jsonl"
    with RunJournal(text_path, metrics_path, run_id="stable-run") as journal:
        assert journal.event("first")["sequence"] == 1
    damaged_tail = b'{"sequence":2,"event":"interrupted"'
    with metrics_path.open("ab") as handle:
        handle.write(damaged_tail)

    with RunJournal(text_path, metrics_path, run_id="stable-run") as journal:
        assert journal.event("resumed")["sequence"] == 3

    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert [row["event"] for row in rows] == [
        "first",
        "journal_tail_recovered",
        "resumed",
    ]
    recovery = rows[1]
    assert recovery["level"] == "WARNING"
    assert recovery["action"] == "truncate_invalid_tail"
    assert recovery["invalid_reason"] == "invalid_json"
    assert recovery["truncated_bytes"] == len(damaged_tail)
    assert recovery["truncated_sha256"] == hashlib.sha256(damaged_tail).hexdigest()


def test_run_journal_completes_valid_final_row_missing_newline(tmp_path: Path) -> None:
    text_path = tmp_path / "resume.log"
    metrics_path = tmp_path / "resume.jsonl"
    with RunJournal(text_path, metrics_path, run_id="stable-run") as journal:
        journal.event("first")
    complete = metrics_path.read_bytes()
    assert complete.endswith(b"\n")
    metrics_path.write_bytes(complete[:-1])

    with RunJournal(text_path, metrics_path, run_id="stable-run") as journal:
        assert journal.event("resumed")["sequence"] == 3

    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert rows[1]["event"] == "journal_tail_recovered"
    assert rows[1]["action"] == "append_missing_newline"
    assert rows[1]["appended_bytes"] == 1


def test_run_journal_refuses_interior_corruption_without_truncation(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "resume.log"
    metrics_path = tmp_path / "resume.jsonl"
    with RunJournal(text_path, metrics_path, run_id="stable-run") as journal:
        journal.event("first")
    valid_second = json.dumps({"sequence": 2, "event": "second"}).encode("utf-8")
    with metrics_path.open("ab") as handle:
        handle.write(b"not-json\n" + valid_second + b"\n")
    before = metrics_path.read_bytes()

    journal = RunJournal(text_path, metrics_path, run_id="stable-run")
    with pytest.raises(ResumeMismatchError, match="not confined to its tail"):
        journal.open()
    assert metrics_path.read_bytes() == before


def test_preprocessing_resume_is_exact_and_declared_case_error_is_warning(
    tmp_path: Path,
) -> None:
    items = ["case-0", "case-1", "case-2", "case-3"]
    hashes = {"manifest": "m1", "preprocess_config": "p1"}
    runner, journal = _runner(tmp_path, io.StringIO())
    with journal, pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run(
            items,
            _Callbacks(fail_once_at=2),
            run_id="run",
            hashes=hashes,
        )

    saved = AtomicStateStore(tmp_path / "resume.json").load()
    partial = PreprocessingResumeState.from_resume_state(saved)
    assert partial.completed.ranges == ((0, 1),)

    stream = io.StringIO()
    resumed, resumed_journal = _runner(tmp_path, stream)
    with resumed_journal:
        final = resumed.run(
            items,
            _Callbacks(acceptable_at=3),
            run_id="run",
            hashes=hashes,
        )

    assert final.status == "completed"
    assert final.completed.count == 4
    assert final.skipped_case_ids == ("case-3",)
    assert "WARNING case-3" in stream.getvalue()
    events = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["event"] == "preprocessing_resumed" for row in events)
    assert any(row["event"] == "preprocessing_case_skipped" for row in events)


def test_journal_refuses_nan_before_writing(tmp_path: Path) -> None:
    journal = RunJournal(tmp_path / "run.log", tmp_path / "metrics.jsonl", run_id="run")
    with journal, pytest.raises(NonFiniteMetricError):
        journal.metric("bad", {"loss": float("nan")})

    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8") == ""
