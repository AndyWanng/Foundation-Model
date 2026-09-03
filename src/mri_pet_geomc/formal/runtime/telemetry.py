from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from .errors import ensure_finite_metrics
from .journal import RunJournal
from .progress import TerminalProgress


_GIB = float(1024**3)


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_rss_gib: float
    gpu_allocated_gib: float = 0.0
    gpu_reserved_gib: float = 0.0
    gpu_peak_allocated_gib: float = 0.0
    gpu_peak_reserved_gib: float = 0.0


@runtime_checkable
class ResourceProvider(Protocol):
    def sample(self) -> ResourceSnapshot: ...


class DefaultResourceProvider:
    """Process RSS and current/peak CUDA memory without forcing CUDA startup."""

    def __init__(self, *, gpu_device: int | str | None = None) -> None:
        self.gpu_device = gpu_device

    def sample(self) -> ResourceSnapshot:
        try:
            import psutil

            cpu_rss = float(psutil.Process().memory_info().rss) / _GIB
        except ImportError:
            cpu_rss = 0.0
        allocated = reserved = peak_allocated = peak_reserved = 0.0
        try:
            import torch

            if torch.cuda.is_initialized():
                allocated = float(torch.cuda.memory_allocated(self.gpu_device)) / _GIB
                reserved = float(torch.cuda.memory_reserved(self.gpu_device)) / _GIB
                peak_allocated = (
                    float(torch.cuda.max_memory_allocated(self.gpu_device)) / _GIB
                )
                peak_reserved = (
                    float(torch.cuda.max_memory_reserved(self.gpu_device)) / _GIB
                )
        except ImportError:
            pass
        return ResourceSnapshot(
            cpu_rss_gib=cpu_rss,
            gpu_allocated_gib=allocated,
            gpu_reserved_gib=reserved,
            gpu_peak_allocated_gib=peak_allocated,
            gpu_peak_reserved_gib=peak_reserved,
        )


@dataclass(frozen=True)
class StepTelemetry:
    coverage: int
    global_step: int
    optimizer_step: int
    next_anchor_index: int
    anchors_processed: int
    encoded_volumes_processed: int
    elapsed_seconds: float
    loader_wait_seconds: float
    throughput_anchors_s: float
    loss_total: float
    loss_components: Mapping[str, float] = field(default_factory=dict)
    modality_losses: Mapping[str, float] = field(default_factory=dict)
    modality_counts: Mapping[str, int] = field(default_factory=dict)
    relation_losses: Mapping[str, float] = field(default_factory=dict)
    relation_counts: Mapping[str, int] = field(default_factory=dict)
    learning_rates: Mapping[str, float] = field(default_factory=dict)
    ema_decay: float = 0.0
    grad_norm: float = 0.0
    resources: ResourceSnapshot = ResourceSnapshot(cpu_rss_gib=0.0)

    def __post_init__(self) -> None:
        if self.coverage < 1:
            raise ValueError("coverage must be 1-based")
        if min(
            self.global_step,
            self.optimizer_step,
            self.next_anchor_index,
            self.anchors_processed,
            self.encoded_volumes_processed,
        ) < 0:
            raise ValueError("Training telemetry counters must be non-negative")
        if self.anchors_processed < 1:
            raise ValueError("anchors_processed must be positive")
        if self.encoded_volumes_processed < self.anchors_processed:
            raise ValueError("encoded volumes cannot be fewer than physical anchors")
        if self.elapsed_seconds <= 0.0:
            raise ValueError("elapsed_seconds must be positive")
        if self.loader_wait_seconds < 0.0:
            raise ValueError("loader_wait_seconds cannot be negative")
        if any(int(value) < 0 for value in self.modality_counts.values()):
            raise ValueError("modality counts cannot be negative")
        if any(int(value) < 0 for value in self.relation_counts.values()):
            raise ValueError("relation counts cannot be negative")
        ensure_finite_metrics(self.as_dict(), context="training telemetry")

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "global_step": self.global_step,
            "optimizer_step": self.optimizer_step,
            "next_anchor_index": self.next_anchor_index,
            "anchors_processed": self.anchors_processed,
            "encoded_volumes_processed": self.encoded_volumes_processed,
            "elapsed_seconds": self.elapsed_seconds,
            "loader_wait_seconds": self.loader_wait_seconds,
            "throughput_anchors_s": self.throughput_anchors_s,
            "throughput_encoded_volumes_s": (
                float(self.encoded_volumes_processed) / float(self.elapsed_seconds)
            ),
            "loss_total": self.loss_total,
            "loss_components": dict(self.loss_components),
            "modality_losses": dict(self.modality_losses),
            "modality_counts": {
                key: int(value) for key, value in self.modality_counts.items()
            },
            "relation_losses": dict(self.relation_losses),
            "relation_counts": {
                key: int(value) for key, value in self.relation_counts.items()
            },
            "learning_rates": dict(self.learning_rates),
            "ema_decay": self.ema_decay,
            "grad_norm": self.grad_norm,
            "resources": asdict(self.resources),
        }

    def flat_metrics(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {
            "loss.total": float(self.loss_total),
            "ema_decay": float(self.ema_decay),
            "grad_norm": float(self.grad_norm),
            "throughput_anchors_s": float(self.throughput_anchors_s),
            "throughput_encoded_volumes_s": (
                float(self.encoded_volumes_processed) / float(self.elapsed_seconds)
            ),
            "loader_wait_s": float(self.loader_wait_seconds),
            "cpu_rss_gib": float(self.resources.cpu_rss_gib),
            "gpu_allocated_gib": float(self.resources.gpu_allocated_gib),
            "gpu_reserved_gib": float(self.resources.gpu_reserved_gib),
            "gpu_peak_allocated_gib": float(self.resources.gpu_peak_allocated_gib),
            "gpu_peak_reserved_gib": float(self.resources.gpu_peak_reserved_gib),
        }
        values.update(
            {f"loss.component.{key}": float(value) for key, value in self.loss_components.items()}
        )
        values.update(
            {f"loss.modality.{key}": float(value) for key, value in self.modality_losses.items()}
        )
        values.update(
            {f"count.modality.{key}": int(value) for key, value in self.modality_counts.items()}
        )
        values.update(
            {f"loss.relation.{key}": float(value) for key, value in self.relation_losses.items()}
        )
        values.update(
            {f"count.relation.{key}": int(value) for key, value in self.relation_counts.items()}
        )
        values.update(
            {f"lr.{key}": float(value) for key, value in self.learning_rates.items()}
        )
        if self.learning_rates:
            values["lr"] = float(next(iter(self.learning_rates.values())))
        return values


class TelemetryRecorder:
    """Record every optimizer-step observation to JSONL and the live terminal."""

    def __init__(
        self,
        *,
        journal: RunJournal,
        progress: TerminalProgress,
        resource_provider: ResourceProvider | None = None,
        gpu_reserved_soft_limit_gib: float | None = None,
    ) -> None:
        self.journal = journal
        self.progress = progress
        self.resource_provider = resource_provider or DefaultResourceProvider()
        self.gpu_reserved_soft_limit_gib = (
            None
            if gpu_reserved_soft_limit_gib is None
            else float(gpu_reserved_soft_limit_gib)
        )
        if (
            self.gpu_reserved_soft_limit_gib is not None
            and self.gpu_reserved_soft_limit_gib <= 0.0
        ):
            raise ValueError("GPU reserved-memory soft limit must be positive")
        self._gpu_soft_limit_warned = False

    def record(
        self,
        *,
        coverage: int,
        global_step: int,
        optimizer_step: int,
        next_anchor_index: int,
        anchors_processed: int,
        elapsed_seconds: float,
        loader_wait_seconds: float,
        loss_total: float,
        loss_components: Mapping[str, float],
        modality_losses: Mapping[str, float],
        modality_counts: Mapping[str, int],
        relation_losses: Mapping[str, float],
        relation_counts: Mapping[str, int],
        learning_rates: Mapping[str, float],
        ema_decay: float,
        grad_norm: float,
        encoded_volumes_processed: int | None = None,
        progress_completed: int | None = None,
        progress_description: str | None = None,
    ) -> StepTelemetry:
        resources = self.resource_provider.sample()
        if (
            self.gpu_reserved_soft_limit_gib is not None
            and not self._gpu_soft_limit_warned
            and resources.gpu_reserved_gib > self.gpu_reserved_soft_limit_gib
        ):
            self._gpu_soft_limit_warned = True
            self.journal.warning(
                "gpu_reserved_memory_soft_limit",
                "CUDA reserved memory exceeded the warning-only planning value",
                gpu_reserved_gib=resources.gpu_reserved_gib,
                soft_limit_gib=self.gpu_reserved_soft_limit_gib,
                affects_training_continuation=False,
            )
            self.progress.warning(
                "CUDA reserved memory exceeded its warning-only planning value"
            )
        telemetry = StepTelemetry(
            coverage=int(coverage),
            global_step=int(global_step),
            optimizer_step=int(optimizer_step),
            next_anchor_index=int(next_anchor_index),
            anchors_processed=int(anchors_processed),
            encoded_volumes_processed=int(
                anchors_processed
                if encoded_volumes_processed is None
                else encoded_volumes_processed
            ),
            elapsed_seconds=float(elapsed_seconds),
            loader_wait_seconds=float(loader_wait_seconds),
            throughput_anchors_s=float(anchors_processed) / float(elapsed_seconds),
            loss_total=float(loss_total),
            loss_components=dict(loss_components),
            modality_losses=dict(modality_losses),
            modality_counts={key: int(value) for key, value in modality_counts.items()},
            relation_losses=dict(relation_losses),
            relation_counts={key: int(value) for key, value in relation_counts.items()},
            learning_rates=dict(learning_rates),
            ema_decay=float(ema_decay),
            grad_norm=float(grad_norm),
            resources=resources,
        )
        flat = telemetry.flat_metrics()
        self.journal.metric(
            "training_optimizer_step",
            flat,
            coverage=telemetry.coverage,
            global_step=telemetry.global_step,
            optimizer_step=telemetry.optimizer_step,
            next_anchor_index=telemetry.next_anchor_index,
            anchors_processed=telemetry.anchors_processed,
            elapsed_seconds=telemetry.elapsed_seconds,
            structured=telemetry.as_dict(),
        )
        visible = {
            key: value
            for key, value in flat.items()
            if key
            in {
                "loss.total",
                "loss.modality.mri",
                "loss.modality.pet",
                "loss.component.prediction_effective_rank",
                "loss.component.target_effective_rank",
                "lr",
                "ema_decay",
                "grad_norm",
                "throughput_anchors_s",
                "throughput_encoded_volumes_s",
                "loader_wait_s",
                "cpu_rss_gib",
                "gpu_allocated_gib",
                "gpu_reserved_gib",
                "gpu_peak_allocated_gib",
                "gpu_peak_reserved_gib",
            }
        }
        self.progress.update(
            completed=progress_completed,
            description=progress_description,
            metrics=visible,
        )
        return telemetry
