"""Formal 24k-run runtime services.

This package deliberately depends on protocols instead of the project's concrete
dataset and model implementations.  It owns run bookkeeping, not scientific
model behaviour.
"""

from .checkpoint import (
    AtomicCheckpointManager,
    CheckpointCursor,
    CheckpointSerializer,
    LoadedCheckpoint,
    PickleSerializer,
    Stateful,
    TorchSerializer,
    capture_rng_state,
    restore_rng_state,
)
from .errors import (
    AcceptableCaseError,
    FatalRuntimeError,
    ManifestContractError,
    ModelContractError,
    NonFiniteMetricError,
    ResumeMismatchError,
    ensure_finite_metrics,
)
from .journal import RunJournal
from .preprocessing import (
    PreprocessResult,
    PreprocessingCallbacks,
    PreprocessingRunner,
)
from .progress import TerminalProgress
from .schedule import (
    BatchContract,
    FormalTrainingSchedule,
    RelationKind,
    RelationMixture,
)
from .state import (
    AtomicStateStore,
    CompletedRanges,
    PreprocessingResumeState,
    ResumeState,
)
from .telemetry import (
    DefaultResourceProvider,
    ResourceProvider,
    ResourceSnapshot,
    StepTelemetry,
    TelemetryRecorder,
)
from .training import (
    CoverageDataSource,
    Microbatch,
    TrainingBackend,
    TrainingRuntimeController,
    TrainingStepResult,
)
from .validation import (
    CoverageValidationCadence,
    ValidationCadence,
    ValidationRequest,
    ValidationTier,
)

__all__ = [
    "AcceptableCaseError",
    "AtomicCheckpointManager",
    "AtomicStateStore",
    "BatchContract",
    "CheckpointCursor",
    "CheckpointSerializer",
    "CompletedRanges",
    "CoverageValidationCadence",
    "CoverageDataSource",
    "DefaultResourceProvider",
    "FatalRuntimeError",
    "FormalTrainingSchedule",
    "LoadedCheckpoint",
    "ManifestContractError",
    "ModelContractError",
    "Microbatch",
    "NonFiniteMetricError",
    "PickleSerializer",
    "PreprocessResult",
    "PreprocessingCallbacks",
    "PreprocessingResumeState",
    "PreprocessingRunner",
    "RelationKind",
    "RelationMixture",
    "ResourceSnapshot",
    "ResourceProvider",
    "ResumeMismatchError",
    "ResumeState",
    "RunJournal",
    "Stateful",
    "StepTelemetry",
    "TelemetryRecorder",
    "TerminalProgress",
    "TorchSerializer",
    "TrainingBackend",
    "TrainingRuntimeController",
    "TrainingStepResult",
    "ValidationCadence",
    "ValidationRequest",
    "ValidationTier",
    "capture_rng_state",
    "ensure_finite_metrics",
    "restore_rng_state",
]
