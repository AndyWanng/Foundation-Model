from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from .errors import (
    AcceptableCaseError,
    FatalRuntimeError,
    ResumeMismatchError,
    ensure_finite_metrics,
)
from .journal import RunJournal
from .progress import TerminalProgress
from .state import AtomicStateStore, PreprocessingResumeState


ItemT = TypeVar("ItemT")


@dataclass(frozen=True)
class PreprocessResult:
    """Audit information returned after one idempotent preprocessing callback."""

    metrics: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    outputs: Mapping[str, str] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PreprocessingCallbacks(Protocol[ItemT]):
    """Concrete preprocessing adapter supplied by the data layer.

    ``process`` must be idempotent because work performed after the most recent
    atomic state write may be replayed after an abrupt process termination.
    Per-case errors are skippable only when raised explicitly as
    :class:`AcceptableCaseError`.
    """

    def case_id(self, item: ItemT) -> str: ...

    def process(self, item: ItemT, *, index: int) -> PreprocessResult: ...


class PreprocessingRunner(Generic[ItemT]):
    def __init__(
        self,
        *,
        state_store: AtomicStateStore,
        journal: RunJournal,
        progress: TerminalProgress,
        state_every_cases: int = 1,
    ) -> None:
        if state_every_cases < 1:
            raise ValueError("state_every_cases must be positive")
        self.state_store = state_store
        self.journal = journal
        self.progress = progress
        self.state_every_cases = int(state_every_cases)

    def run(
        self,
        items: Sequence[ItemT],
        callbacks: PreprocessingCallbacks[ItemT],
        *,
        run_id: str,
        hashes: Mapping[str, str],
        resume: bool = True,
    ) -> PreprocessingResumeState:
        total = len(items)
        if total < 1:
            raise ValueError("Preprocessing requires a non-empty deterministic manifest")
        if self.state_store.exists:
            if not resume:
                raise ResumeMismatchError(
                    f"Resume state already exists but resume=False: {self.state_store.path}"
                )
            raw_state = self.state_store.load(
                expected_kind="preprocessing",
                expected_run_id=run_id,
                expected_hashes=hashes,
            )
            state = PreprocessingResumeState.from_resume_state(raw_state)
            if state.total_cases != total:
                raise ResumeMismatchError(
                    f"Manifest length changed: found={state.total_cases} expected={total}"
                )
            self.journal.event(
                "preprocessing_resumed",
                "Resuming deterministic preprocessing manifest",
                completed_cases=state.completed.count,
                total_cases=total,
                warning_count=state.warning_count,
            )
        else:
            state = PreprocessingResumeState(
                run_id=run_id,
                total_cases=total,
                hashes=dict(hashes),
            )
            self.state_store.save(state.to_resume_state())
            self.journal.event(
                "preprocessing_initialized",
                "Created preprocessing resume state",
                total_cases=total,
                hashes=dict(hashes),
            )

        self.progress.begin_stage(
            "preprocessing",
            total=total,
            completed=state.completed.count,
            description="offline preprocessing",
            unit="cases",
        )
        if state.status == "completed":
            self.progress.finish_stage(status="already completed")
            return state

        unsaved = 0
        for index, item in enumerate(items):
            if state.completed.contains(index):
                continue
            case_id = str(callbacks.case_id(item))
            if not case_id:
                raise ResumeMismatchError(f"Empty case identifier at manifest index {index}")
            started = time.perf_counter()
            try:
                result = callbacks.process(item, index=index)
                if not isinstance(result, PreprocessResult):
                    raise TypeError(
                        "Preprocessing callback must return PreprocessResult, "
                        f"found {type(result).__name__}"
                    )
                ensure_finite_metrics(
                    result.metrics,
                    context=f"preprocessing metrics for {case_id}",
                )
            except AcceptableCaseError as error:
                elapsed = time.perf_counter() - started
                state = state.mark_case(
                    index,
                    skipped_case_id=case_id,
                    warning_increment=1,
                )
                self.journal.warning(
                    "preprocessing_case_skipped",
                    str(error),
                    case_id=case_id,
                    manifest_index=index,
                    code=error.code,
                    details=dict(error.details or {}),
                    elapsed_seconds=elapsed,
                )
                self.progress.warning(f"{case_id}: {error}")
            except FatalRuntimeError as error:
                self.journal.fatal(
                    "preprocessing_fatal",
                    str(error),
                    case_id=case_id,
                    manifest_index=index,
                    exception_type=type(error).__name__,
                )
                self.journal.sync()
                raise
            except Exception as error:
                self.journal.fatal(
                    "preprocessing_case_failed",
                    str(error),
                    case_id=case_id,
                    manifest_index=index,
                    exception_type=type(error).__name__,
                )
                self.journal.sync()
                raise
            else:
                elapsed = time.perf_counter() - started
                warning_count = len(result.warnings)
                state = state.mark_case(index, warning_increment=warning_count)
                self.journal.metric(
                    "preprocessing_case_completed",
                    {
                        **dict(result.metrics),
                        "elapsed_seconds": elapsed,
                    },
                    case_id=case_id,
                    manifest_index=index,
                    outputs=dict(result.outputs),
                    warnings=list(result.warnings),
                    details=dict(result.details),
                )
                for warning in result.warnings:
                    self.progress.warning(f"{case_id}: {warning}")

            unsaved += 1
            if unsaved >= self.state_every_cases or state.status == "completed":
                self.state_store.save(state.to_resume_state())
                self.journal.sync()
                unsaved = 0
            self.progress.update(
                completed=state.completed.count,
                status=(
                    f"running; warnings={state.warning_count}; "
                    f"skipped={len(state.skipped_case_ids)}"
                ),
                metrics={
                    "warnings": state.warning_count,
                    "skipped": len(state.skipped_case_ids),
                },
            )

        if unsaved:
            self.state_store.save(state.to_resume_state())
            self.journal.sync()
        if state.status != "completed":
            raise ResumeMismatchError("Preprocessing exhausted its manifest without completion")
        self.progress.finish_stage(
            status=(
                f"completed; warnings={state.warning_count}; "
                f"skipped={len(state.skipped_case_ids)}"
            )
        )
        self.journal.event(
            "preprocessing_completed",
            "Processed the full immutable manifest",
            total_cases=total,
            warning_count=state.warning_count,
            skipped_case_ids=list(state.skipped_case_ids),
        )
        self.journal.sync()
        return state
