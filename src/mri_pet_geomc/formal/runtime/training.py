from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from .checkpoint import (
    AtomicCheckpointManager,
    CheckpointCursor,
    LoadedCheckpoint,
    Stateful,
)
from .errors import FatalRuntimeError, ModelContractError, ensure_finite_metrics
from .journal import RunJournal
from .progress import TerminalProgress
from .schedule import FormalTrainingSchedule, RelationMixture
from .telemetry import StepTelemetry, TelemetryRecorder
from .validation import CoverageValidationCadence, ValidationCadence, ValidationRequest


BatchT = TypeVar("BatchT")


@dataclass(frozen=True)
class Microbatch(Generic[BatchT]):
    payload: BatchT
    anchor_count: int
    next_anchor_index: int
    loader_wait_seconds: float

    def __post_init__(self) -> None:
        if self.anchor_count < 1:
            raise ValueError("A microbatch must contain at least one anchor")
        if self.next_anchor_index < self.anchor_count:
            raise ValueError("next_anchor_index cannot precede the microbatch")
        if self.loader_wait_seconds < 0.0:
            raise ValueError("loader_wait_seconds cannot be negative")


@dataclass(frozen=True)
class TrainingStepResult:
    """Backend result for one optimizer step (normally two microbatches)."""

    anchors_processed: int
    next_anchor_index: int
    elapsed_seconds: float
    loader_wait_seconds: float
    loss_total: float
    loss_components: Mapping[str, float] = field(default_factory=dict)
    modality_losses: Mapping[str, float] = field(default_factory=dict)
    modality_counts: Mapping[str, int] = field(default_factory=dict)
    relation_losses: Mapping[str, float] = field(default_factory=dict)
    relation_counts: Mapping[str, int] = field(default_factory=dict)
    learning_rates: Mapping[str, float] = field(default_factory=dict)
    ema_decay: float = 0.0
    grad_norm: float = 0.0
    encoded_volumes_processed: int = 0
    is_coverage_tail: bool = False


@runtime_checkable
class CoverageDataSource(Protocol[BatchT]):
    """Deterministic anchor-order source; no replacement within a coverage."""

    def total_anchors(self, coverage: int) -> int: ...

    def iter_microbatches(
        self,
        *,
        coverage: int,
        start_anchor_index: int,
        micro_batch_anchors: int,
        relation_mixture: RelationMixture,
    ) -> Iterable[Microbatch[BatchT]]: ...


@runtime_checkable
class TrainingBackend(Protocol[BatchT]):
    """Concrete model/optimizer adapter used by a formal training command."""

    model: Stateful
    optimizer: Stateful
    ema: Stateful
    scheduler: Stateful | None
    scaler: Stateful | None

    def train_accumulated_step(
        self,
        microbatches: Sequence[Microbatch[BatchT]],
        *,
        coverage: int,
        relation_mixture: RelationMixture,
        expected_global_batch_anchors: int,
    ) -> TrainingStepResult: ...

    def validate(self, request: ValidationRequest) -> Mapping[str, Any]: ...

    def extra_checkpoint_stateful(self) -> Mapping[str, Stateful]: ...


class TrainingRuntimeController:
    """Policy-enforcing runtime facade for the concrete training loop.

    The controller records and checkpoints; it intentionally has no
    performance-based stop API.  The caller advances through every configured coverage
    unless a fatal numeric, manifest/hash, or model-state error is raised.
    """

    def __init__(
        self,
        *,
        run_id: str,
        hashes: Mapping[str, str],
        journal: RunJournal,
        progress: TerminalProgress,
        telemetry: TelemetryRecorder,
        checkpoints: AtomicCheckpointManager,
        schedule: FormalTrainingSchedule | None = None,
        validation_cadence: ValidationCadence | None = None,
        checkpoint_every_optimizer_steps: int = 500,
    ) -> None:
        if checkpoint_every_optimizer_steps < 1:
            raise ValueError("checkpoint cadence must be positive")
        self.run_id = str(run_id)
        self.hashes = {str(key): str(value) for key, value in hashes.items()}
        if not self.run_id or not self.hashes:
            raise ValueError("run_id and immutable hash contract are required")
        self.journal = journal
        self.progress = progress
        self.telemetry = telemetry
        self.checkpoints = checkpoints
        self.schedule = schedule or FormalTrainingSchedule()
        self.validation_cadence = validation_cadence or CoverageValidationCadence()
        self.checkpoint_every_optimizer_steps = int(checkpoint_every_optimizer_steps)
        self._coverage_total: int | None = None
        self._active_coverage: int | None = None

    def initialize(self, *, resume_requested: bool) -> None:
        self.journal.event(
            "training_runtime_initialized",
            "Formal run uses a fixed horizon and observational validation",
            resume_requested=bool(resume_requested),
            no_performance_early_stop=True,
            schedule=dict(self.schedule.contract()),
            checkpoint_every_optimizer_steps=self.checkpoint_every_optimizer_steps,
            hashes=self.hashes,
        )
        self.journal.sync()

    def relation_mixture(self, coverage: int) -> RelationMixture:
        return self.schedule.relation_mixture(coverage)

    def begin_coverage(
        self,
        coverage: int,
        *,
        total_anchors: int,
        start_anchor_index: int = 0,
    ) -> RelationMixture:
        mixture = self.schedule.relation_mixture(coverage)
        if total_anchors < 1:
            raise ValueError("A training coverage must contain anchors")
        if not 0 <= start_anchor_index <= total_anchors:
            raise ValueError("Coverage resume cursor is out of bounds")
        self._active_coverage = int(coverage)
        self._coverage_total = int(total_anchors)
        self.progress.begin_stage(
            "training",
            total=total_anchors,
            completed=start_anchor_index,
            description=f"coverage {coverage:03d}/{self.schedule.total_coverages:03d}",
            unit="anchors",
        )
        self.journal.event(
            "training_coverage_started",
            f"Coverage {coverage} started",
            coverage=coverage,
            total_anchors=total_anchors,
            start_anchor_index=start_anchor_index,
            relation_mixture={key.value: value for key, value in mixture.as_dict().items()},
        )
        return mixture

    def record_optimizer_step(
        self,
        *,
        cursor: CheckpointCursor,
        result: TrainingStepResult,
    ) -> StepTelemetry:
        if self._active_coverage != cursor.coverage or self._coverage_total is None:
            raise ModelContractError("No matching active coverage for optimizer-step telemetry")
        expected = self.schedule.batch.global_batch_anchors
        if result.is_coverage_tail:
            if not 1 <= result.anchors_processed <= expected:
                raise ModelContractError("Invalid tail optimizer-step anchor count")
        elif result.anchors_processed != expected:
            raise ModelContractError(
                f"Non-tail optimizer step used {result.anchors_processed} anchors; "
                f"expected exactly {expected}"
            )
        if result.next_anchor_index != cursor.next_anchor_index:
            raise ModelContractError("Training result and checkpoint cursor disagree")
        if result.next_anchor_index > self._coverage_total:
            raise ModelContractError("Training cursor moved beyond the active coverage")
        return self.telemetry.record(
            coverage=cursor.coverage,
            global_step=cursor.global_step,
            optimizer_step=cursor.optimizer_step,
            next_anchor_index=cursor.next_anchor_index,
            anchors_processed=result.anchors_processed,
            encoded_volumes_processed=(
                result.encoded_volumes_processed or result.anchors_processed
            ),
            elapsed_seconds=result.elapsed_seconds,
            loader_wait_seconds=result.loader_wait_seconds,
            loss_total=result.loss_total,
            loss_components=result.loss_components,
            modality_losses=result.modality_losses,
            modality_counts=result.modality_counts,
            relation_losses=result.relation_losses,
            relation_counts=result.relation_counts,
            learning_rates=result.learning_rates,
            ema_decay=result.ema_decay,
            grad_norm=result.grad_norm,
            progress_completed=cursor.next_anchor_index,
        )

    def train_accumulated_step(
        self,
        *,
        backend: TrainingBackend[BatchT],
        microbatches: Sequence[Microbatch[BatchT]],
        coverage: int,
        is_coverage_tail: bool = False,
    ) -> TrainingStepResult:
        """Validate 8 x 2 accumulation, then delegate one optimizer step."""

        if coverage != self._active_coverage:
            raise ModelContractError("Accumulated step does not match the active coverage")
        expected_micro = self.schedule.batch.micro_batch_anchors
        expected_accumulation = self.schedule.batch.gradient_accumulation_steps
        if not microbatches or len(microbatches) > expected_accumulation:
            raise ModelContractError("Invalid number of microbatches in accumulation group")
        if not is_coverage_tail and len(microbatches) != expected_accumulation:
            raise ModelContractError(
                f"Non-tail step requires {expected_accumulation} microbatches"
            )
        for position, microbatch in enumerate(microbatches):
            if microbatch.anchor_count > expected_micro:
                raise ModelContractError(
                    f"Microbatch {position} exceeds the {expected_micro}-anchor contract"
                )
            if not is_coverage_tail and microbatch.anchor_count != expected_micro:
                raise ModelContractError(
                    f"Non-tail microbatch {position} has {microbatch.anchor_count} anchors; "
                    f"expected {expected_micro}"
                )
        for previous, current in zip(microbatches, microbatches[1:]):
            if current.next_anchor_index <= previous.next_anchor_index:
                raise ModelContractError("Microbatch cursors are not strictly increasing")
        result = backend.train_accumulated_step(
            microbatches,
            coverage=coverage,
            relation_mixture=self.schedule.relation_mixture(coverage),
            expected_global_batch_anchors=self.schedule.batch.global_batch_anchors,
        )
        anchors = sum(microbatch.anchor_count for microbatch in microbatches)
        if result.anchors_processed != anchors:
            raise ModelContractError("Backend result disagrees with accumulated anchor count")
        if result.next_anchor_index != microbatches[-1].next_anchor_index:
            raise ModelContractError("Backend result disagrees with final microbatch cursor")
        if bool(result.is_coverage_tail) != bool(is_coverage_tail):
            raise ModelContractError("Backend and runtime disagree about coverage-tail status")
        return result

    def checkpoint_due(self, *, optimizer_step: int, end_of_coverage: bool = False) -> bool:
        if optimizer_step < 0:
            raise ValueError("optimizer_step cannot be negative")
        return bool(
            end_of_coverage
            or (
                optimizer_step > 0
                and optimizer_step % self.checkpoint_every_optimizer_steps == 0
            )
        )

    def save_checkpoint(
        self,
        *,
        cursor: CheckpointCursor,
        backend: TrainingBackend[Any],
        metrics: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        return self.checkpoints.save(
            cursor=cursor,
            hashes=self.hashes,
            model=backend.model,
            optimizer=backend.optimizer,
            ema=backend.ema,
            scheduler=backend.scheduler,
            scaler=backend.scaler,
            extra_stateful=backend.extra_checkpoint_stateful(),
            metrics=metrics,
            extra=extra,
        )

    def resume(self, *, backend: TrainingBackend[Any]) -> LoadedCheckpoint:
        return self.checkpoints.load_latest(
            expected_hashes=self.hashes,
            model=backend.model,
            optimizer=backend.optimizer,
            ema=backend.ema,
            scheduler=backend.scheduler,
            scaler=backend.scaler,
            extra_stateful=backend.extra_checkpoint_stateful(),
            restore_rng=True,
        )

    def validation_requests(
        self,
        *,
        coverage: int,
        optimizer_step: int,
    ) -> tuple[ValidationRequest, ...]:
        return self.validation_cadence.requests(
            coverage=coverage,
            optimizer_step=optimizer_step,
            is_final=coverage == self.schedule.total_coverages,
        )

    def run_validation(
        self,
        *,
        backend: TrainingBackend[Any],
        request: ValidationRequest,
    ) -> Mapping[str, Any]:
        metrics = dict(backend.validate(request))
        ensure_finite_metrics(metrics, context=f"{request.tier.value} validation")
        self.journal.metric(
            "validation_completed",
            metrics,
            coverage=request.coverage,
            optimizer_step=request.optimizer_step,
            tier=request.tier.value,
            reason=request.reason,
            sample_limit=request.sample_limit,
            masks_per_anchor=request.masks_per_anchor,
            observational_only=True,
            affects_training_continuation=False,
        )
        return metrics

    def finish_coverage(self, *, coverage: int, optimizer_step: int) -> None:
        if coverage != self._active_coverage or self._coverage_total is None:
            raise ModelContractError("Cannot finish a coverage that is not active")
        if self.progress.state.completed != self._coverage_total:
            raise ModelContractError(
                "Cannot finish coverage before every deterministic anchor is consumed"
            )
        self.progress.finish_stage(status="coverage completed")
        self.journal.event(
            "training_coverage_completed",
            f"Coverage {coverage} completed",
            coverage=coverage,
            optimizer_step=optimizer_step,
            total_anchors=self._coverage_total,
        )
        self._active_coverage = None
        self._coverage_total = None

    def warning(self, code: str, message: str, **details: Any) -> None:
        self.journal.warning(code, message, **details)
        self.progress.warning(message)

    def record_fatal(self, error: BaseException, *, event: str = "training_fatal") -> None:
        self.journal.fatal(
            event,
            str(error),
            exception_type=type(error).__name__,
        )
        self.journal.sync()
        if isinstance(error, FatalRuntimeError):
            raise error
        raise ModelContractError(str(error)) from error
