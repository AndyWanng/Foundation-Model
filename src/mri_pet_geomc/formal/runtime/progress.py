from __future__ import annotations

import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TextIO

from .journal import RunJournal


@dataclass
class _ProgressState:
    stage: str = "idle"
    description: str = "waiting"
    completed: int = 0
    total: int = 1
    unit: str = "items"
    status: str = "ready"
    metrics: dict[str, Any] = field(default_factory=dict)


class TerminalProgress:
    """One clean Rich Live view for preprocessing and training.

    Rich is imported lazily so metadata/catalog commands remain usable in a
    minimal environment.  Formal workstation environments should install Rich;
    when it is unavailable this class emits a throttled plain-text progress bar.
    """

    _PREFERRED_METRICS = (
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
    )

    def __init__(
        self,
        *,
        journal: RunJournal | None = None,
        stream: TextIO | None = None,
        force_plain: bool = False,
        refresh_per_second: float = 5.0,
        plain_update_every: int = 100,
        transient: bool = False,
    ) -> None:
        self.journal = journal
        self.stream = stream or sys.stderr
        self.force_plain = bool(force_plain)
        self.refresh_per_second = float(refresh_per_second)
        self.plain_update_every = max(1, int(plain_update_every))
        self.transient = bool(transient)
        self.state = _ProgressState()
        self._lock = threading.RLock()
        self._entered = False
        self._rich = False
        self._console: Any = None
        self._live: Any = None
        self._progress: Any = None
        self._task_id: Any = None
        self._plain_updates = 0

    @property
    def rich_enabled(self) -> bool:
        return self._rich

    def __enter__(self) -> TerminalProgress:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        with self._lock:
            if self._entered:
                return
            self._entered = True
            if not self.force_plain:
                try:
                    from rich.console import Console
                    from rich.live import Live
                    from rich.progress import (
                        BarColumn,
                        MofNCompleteColumn,
                        Progress,
                        SpinnerColumn,
                        TaskProgressColumn,
                        TextColumn,
                        TimeRemainingColumn,
                    )

                    self._console = Console(file=self.stream)
                    self._progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[bold cyan]{task.description}"),
                        BarColumn(bar_width=None),
                        TaskProgressColumn(),
                        MofNCompleteColumn(),
                        TextColumn("{task.fields[unit]}"),
                        TimeRemainingColumn(),
                        console=self._console,
                        expand=True,
                    )
                    self._live = Live(
                        self._render_rich(),
                        console=self._console,
                        refresh_per_second=self.refresh_per_second,
                        transient=self.transient,
                    )
                    self._live.start(refresh=True)
                    self._rich = True
                except ImportError:
                    self._rich = False

    def stop(self) -> None:
        with self._lock:
            if not self._entered:
                return
            if self._live is not None:
                self._live.update(self._render_rich(), refresh=True)
                self._live.stop()
            self._entered = False

    def begin_stage(
        self,
        stage: str,
        *,
        total: int,
        description: str | None = None,
        unit: str = "items",
        completed: int = 0,
    ) -> None:
        if total < 1:
            raise ValueError("Progress total must be positive")
        if not 0 <= completed <= total:
            raise ValueError("Progress completed value is out of bounds")
        with self._lock:
            self.start()
            self.state = _ProgressState(
                stage=str(stage),
                description=description or str(stage),
                completed=int(completed),
                total=int(total),
                unit=str(unit),
                status="running",
            )
            if self._rich:
                if self._task_id is not None:
                    self._progress.remove_task(self._task_id)
                self._task_id = self._progress.add_task(
                    self.state.description,
                    total=self.state.total,
                    completed=self.state.completed,
                    unit=self.state.unit,
                )
                self._refresh_rich()
            else:
                self._emit_plain(force=True)
            if self.journal is not None:
                self.journal.event(
                    "stage_started",
                    self.state.description,
                    stage=self.state.stage,
                    completed=self.state.completed,
                    total=self.state.total,
                    unit=self.state.unit,
                )

    def update(
        self,
        *,
        completed: int | None = None,
        advance: int = 0,
        description: str | None = None,
        status: str | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            next_completed = (
                self.state.completed + int(advance)
                if completed is None
                else int(completed)
            )
            if not 0 <= next_completed <= self.state.total:
                raise ValueError(
                    f"Progress update {next_completed} is outside [0, {self.state.total}]"
                )
            self.state.completed = next_completed
            if description is not None:
                self.state.description = str(description)
            if status is not None:
                self.state.status = str(status)
            if metrics is not None:
                self.state.metrics = dict(metrics)
            if self._rich:
                self._progress.update(
                    self._task_id,
                    completed=self.state.completed,
                    description=self.state.description,
                    unit=self.state.unit,
                )
                self._refresh_rich()
            else:
                self._plain_updates += 1
                self._emit_plain(force=self.state.completed == self.state.total)

    def finish_stage(self, *, status: str = "completed") -> None:
        with self._lock:
            self.update(completed=self.state.total, status=status)
            if self.journal is not None:
                self.journal.event(
                    "stage_finished",
                    self.state.description,
                    stage=self.state.stage,
                    status=self.state.status,
                    completed=self.state.completed,
                    total=self.state.total,
                )

    def warning(self, message: str) -> None:
        with self._lock:
            if self._rich and self._console is not None:
                self._console.print(f"[yellow]WARNING[/yellow] {message}")
            else:
                self.stream.write(f"WARNING {message}\n")
                self.stream.flush()

    def _render_rich(self) -> Any:
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        heading = Text()
        heading.append(self.state.stage, style="bold cyan")
        heading.append(f"  {self.state.status}", style="green")
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        ordered = []
        seen = set()
        for name in self._PREFERRED_METRICS:
            if name in self.state.metrics:
                ordered.append((name, self.state.metrics[name]))
                seen.add(name)
        ordered.extend(
            (name, value)
            for name, value in sorted(self.state.metrics.items())
            if name not in seen
        )
        cells = [f"[dim]{name}[/dim] {self._format_metric(value)}" for name, value in ordered]
        for index in range(0, len(cells), 3):
            row = cells[index : index + 3]
            table.add_row(*(row + [""] * (3 - len(row))))
        items = [Panel(heading, border_style="cyan")]
        if self._progress is not None:
            items.append(self._progress)
        if cells:
            items.append(Panel(table, title="live metrics", border_style="blue"))
        return Group(*items)

    def _refresh_rich(self) -> None:
        if self._live is not None:
            self._live.update(self._render_rich(), refresh=True)

    def _emit_plain(self, *, force: bool) -> None:
        if not force and self._plain_updates % self.plain_update_every:
            return
        fraction = self.state.completed / max(self.state.total, 1)
        width = 24
        filled = min(width, int(round(fraction * width)))
        bar = "#" * filled + "-" * (width - filled)
        metrics = " ".join(
            f"{key}={self._format_metric(value)}"
            for key, value in self.state.metrics.items()
        )
        line = (
            f"[{self.state.stage}] [{bar}] {self.state.completed}/{self.state.total} "
            f"{self.state.unit} {100.0 * fraction:5.1f}% {self.state.status}"
        )
        if metrics:
            line += " | " + metrics
        self.stream.write(line.rstrip() + "\n")
        self.stream.flush()

    @staticmethod
    def _format_metric(value: Any) -> str:
        if isinstance(value, float):
            magnitude = abs(value)
            if magnitude != 0.0 and (magnitude < 1.0e-3 or magnitude >= 1.0e4):
                return f"{value:.3e}"
            return f"{value:.4f}"
        return str(value)
