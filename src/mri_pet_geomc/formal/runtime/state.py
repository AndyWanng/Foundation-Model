from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .errors import ManifestContractError, ResumeMismatchError, ensure_finite_metrics


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _validate_hashes(hashes: Mapping[str, str]) -> dict[str, str]:
    result = {str(key): str(value) for key, value in hashes.items()}
    if not result or any(not key or not value for key, value in result.items()):
        raise ManifestContractError("Resume hash contract must contain non-empty keys and values")
    return result


@dataclass(frozen=True)
class CompletedRanges:
    """Compact inclusive ranges for completed deterministic item indices."""

    ranges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        previous_end = -2
        for start, end in self.ranges:
            if start < 0 or end < start:
                raise ValueError(f"Invalid completed range: {(start, end)}")
            if start <= previous_end + 1:
                raise ValueError("Completed ranges must be sorted, disjoint, and merged")
            previous_end = end

    @property
    def count(self) -> int:
        return sum(end - start + 1 for start, end in self.ranges)

    def contains(self, index: int) -> bool:
        value = int(index)
        for start, end in self.ranges:
            if value < start:
                return False
            if value <= end:
                return True
        return False

    def add(self, index: int) -> CompletedRanges:
        value = int(index)
        if value < 0:
            raise ValueError("Completed index must be non-negative")
        expanded = list(self.ranges) + [(value, value)]
        expanded.sort()
        merged: list[tuple[int, int]] = []
        for start, end in expanded:
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return CompletedRanges(tuple(merged))

    def first_pending(self, total: int) -> int | None:
        if total < 0:
            raise ValueError("total must be non-negative")
        candidate = 0
        for start, end in self.ranges:
            if candidate < start:
                return candidate if candidate < total else None
            candidate = max(candidate, end + 1)
        return candidate if candidate < total else None

    def as_json(self) -> list[list[int]]:
        return [[start, end] for start, end in self.ranges]

    @classmethod
    def from_json(cls, value: object) -> CompletedRanges:
        if not isinstance(value, list):
            raise ResumeMismatchError("completed_ranges must be a list")
        try:
            ranges = tuple((int(item[0]), int(item[1])) for item in value)
        except (TypeError, ValueError, IndexError) as error:
            raise ResumeMismatchError("completed_ranges contains an invalid range") from error
        return cls(ranges)


@dataclass(frozen=True)
class ResumeState:
    kind: Literal["preprocessing", "training"]
    run_id: str
    hashes: Mapping[str, str]
    status: Literal["running", "completed"] = "running"
    cursor: Mapping[str, int] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Only resume-state schema version 1 is supported")
        if self.kind not in {"preprocessing", "training"}:
            raise ValueError(f"Unsupported resume state kind: {self.kind}")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        _validate_hashes(self.hashes)
        if self.status not in {"running", "completed"}:
            raise ValueError(f"Unsupported resume status: {self.status}")
        if any(int(value) < 0 for value in self.cursor.values()):
            raise ValueError("Resume cursor values must be non-negative")
        ensure_finite_metrics(self.payload, context="resume-state payload")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "hashes": dict(self.hashes),
            "status": self.status,
            "cursor": {key: int(value) for key, value in self.cursor.items()},
            "payload": dict(self.payload),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResumeState:
        try:
            return cls(
                schema_version=int(value["schema_version"]),
                kind=str(value["kind"]),  # type: ignore[arg-type]
                run_id=str(value["run_id"]),
                hashes=dict(value["hashes"]),
                status=str(value["status"]),  # type: ignore[arg-type]
                cursor={key: int(item) for key, item in dict(value["cursor"]).items()},
                payload=dict(value["payload"]),
                updated_at=str(value["updated_at"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ResumeMismatchError("Resume-state payload is malformed") from error


@dataclass(frozen=True)
class PreprocessingResumeState:
    run_id: str
    total_cases: int
    hashes: Mapping[str, str]
    completed: CompletedRanges = CompletedRanges()
    skipped_case_ids: tuple[str, ...] = ()
    warning_count: int = 0
    status: Literal["running", "completed"] = "running"

    def __post_init__(self) -> None:
        if self.total_cases < 1:
            raise ValueError("total_cases must be positive")
        if self.completed.count > self.total_cases:
            raise ValueError("Completed case count exceeds total_cases")
        if self.warning_count < len(self.skipped_case_ids):
            raise ValueError("warning_count cannot be smaller than skipped-case count")
        _validate_hashes(self.hashes)
        if self.status == "completed" and self.completed.count != self.total_cases:
            raise ValueError("A completed preprocessing state must cover every case")

    def mark_case(
        self,
        index: int,
        *,
        skipped_case_id: str | None = None,
        warning_increment: int = 0,
    ) -> PreprocessingResumeState:
        if not 0 <= index < self.total_cases:
            raise ValueError(f"Case index {index} is outside the preprocessing manifest")
        already_complete = self.completed.contains(index)
        skipped = self.skipped_case_ids
        if skipped_case_id is not None and skipped_case_id not in skipped:
            skipped = skipped + (str(skipped_case_id),)
        warnings = self.warning_count + (0 if already_complete else int(warning_increment))
        completed = self.completed.add(index)
        status: Literal["running", "completed"] = (
            "completed" if completed.count == self.total_cases else "running"
        )
        return replace(
            self,
            completed=completed,
            skipped_case_ids=skipped,
            warning_count=warnings,
            status=status,
        )

    def to_resume_state(self) -> ResumeState:
        return ResumeState(
            kind="preprocessing",
            run_id=self.run_id,
            hashes=dict(self.hashes),
            status=self.status,
            cursor={
                "total_cases": self.total_cases,
                "completed_cases": self.completed.count,
                "warning_count": self.warning_count,
            },
            payload={
                "completed_ranges": self.completed.as_json(),
                "skipped_case_ids": list(self.skipped_case_ids),
            },
        )

    @classmethod
    def from_resume_state(cls, state: ResumeState) -> PreprocessingResumeState:
        if state.kind != "preprocessing":
            raise ResumeMismatchError(f"Expected preprocessing state, found {state.kind}")
        try:
            total = int(state.cursor["total_cases"])
            warning_count = int(state.cursor.get("warning_count", 0))
            completed = CompletedRanges.from_json(state.payload["completed_ranges"])
            skipped = tuple(str(item) for item in state.payload.get("skipped_case_ids", []))
        except (KeyError, TypeError, ValueError) as error:
            raise ResumeMismatchError("Preprocessing resume state is malformed") from error
        result = cls(
            run_id=state.run_id,
            total_cases=total,
            hashes=dict(state.hashes),
            completed=completed,
            skipped_case_ids=skipped,
            warning_count=warning_count,
            status=state.status,
        )
        recorded = int(state.cursor.get("completed_cases", completed.count))
        if recorded != completed.count:
            raise ResumeMismatchError(
                "Preprocessing completed count disagrees with completed ranges"
            )
        return result


class AtomicStateStore:
    """Durable atomic JSON state used by preprocessing and lightweight cursors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def save(self, state: ResumeState) -> Path:
        value = state.as_dict()
        ensure_finite_metrics(value, context="resume state")
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return self.path

    def load(
        self,
        *,
        expected_kind: Literal["preprocessing", "training"] | None = None,
        expected_run_id: str | None = None,
        expected_hashes: Mapping[str, str] | None = None,
    ) -> ResumeState:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as error:
            raise ResumeMismatchError(f"Cannot read atomic resume state: {self.path}") from error
        if not isinstance(value, Mapping):
            raise ResumeMismatchError("Resume-state root must be a JSON object")
        state = ResumeState.from_dict(value)
        if expected_kind is not None and state.kind != expected_kind:
            raise ResumeMismatchError(
                f"Resume kind mismatch: found={state.kind} expected={expected_kind}"
            )
        if expected_run_id is not None and state.run_id != expected_run_id:
            raise ResumeMismatchError(
                f"Resume run_id mismatch: found={state.run_id} expected={expected_run_id}"
            )
        if expected_hashes is not None:
            expected = _validate_hashes(expected_hashes)
            found = _validate_hashes(state.hashes)
            if found != expected:
                raise ResumeMismatchError(
                    f"Resume hash contract mismatch: found={found} expected={expected}"
                )
        return state
