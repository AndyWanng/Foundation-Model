from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from .errors import ResumeMismatchError, ensure_finite_metrics


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    try:
        return value.item()
    except (AttributeError, TypeError, ValueError):
        return value


def _journal_row_sequence(raw: bytes, previous: int) -> tuple[int | None, str | None]:
    """Validate one physical JSONL row and its strictly increasing sequence."""

    content = raw[:-1] if raw.endswith(b"\n") else raw
    if content.endswith(b"\r"):
        content = content[:-1]
    if not content.strip():
        return None, "blank_row"
    try:
        row = json.loads(content.decode("utf-8"))
    except UnicodeDecodeError:
        return None, "invalid_utf8"
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(row, Mapping):
        return None, "non_object_row"
    sequence = row.get("sequence")
    if type(sequence) is not int or sequence < 0:  # bool is not a valid sequence
        return None, "invalid_sequence"
    if sequence <= previous:
        return None, "non_monotonic_sequence"
    return sequence, None


def _tail_digest(handle: Any, start: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    handle.seek(start)
    while chunk := handle.read(1024 * 1024):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def _repair_journal_tail(path: Path) -> tuple[int, dict[str, Any] | None]:
    """Validate JSONL and durably remove corruption confined to its tail.

    A malformed row followed by another valid row is interior corruption and
    fails closed.  A malformed final suffix is truncated to the last complete,
    strictly monotonic event.  A valid final row missing only its newline is
    completed in place so the next append cannot be glued to it.
    """

    if not path.is_file() or path.stat().st_size == 0:
        return 0, None
    with path.open("r+b") as handle:
        last_sequence = 0
        invalid_start: int | None = None
        invalid_reason: str | None = None
        final_row_missing_newline = False
        while True:
            row_start = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            sequence, reason = _journal_row_sequence(raw, last_sequence)
            if invalid_start is None:
                if reason is None:
                    assert sequence is not None
                    last_sequence = sequence
                    final_row_missing_newline = not raw.endswith(b"\n")
                    continue
                invalid_start = row_start
                invalid_reason = reason
                final_row_missing_newline = False
                continue
            # Once a malformed suffix starts, a later independently valid row
            # proves that corruption is not confined to the tail.  Do not
            # silently discard valid committed events in that situation.
            later_sequence, later_reason = _journal_row_sequence(raw, last_sequence)
            if later_reason is None and later_sequence is not None:
                raise ResumeMismatchError(
                    "Run journal corruption is not confined to its tail: "
                    f"a valid row follows invalid byte offset {invalid_start}"
                )

        if invalid_start is not None:
            truncated_bytes, truncated_sha256 = _tail_digest(handle, invalid_start)
            handle.seek(invalid_start)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
            return last_sequence, {
                "action": "truncate_invalid_tail",
                "invalid_byte_offset": invalid_start,
                "invalid_reason": invalid_reason,
                "previous_last_sequence": last_sequence,
                "truncated_bytes": truncated_bytes,
                "truncated_sha256": truncated_sha256,
            }
        if final_row_missing_newline:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            return last_sequence, {
                "action": "append_missing_newline",
                "appended_bytes": 1,
                "previous_last_sequence": last_sequence,
            }
    return last_sequence, None


class RunJournal:
    """Write human-readable and machine-readable logs on the calling thread.

    Every call serializes once, writes to both files, and flushes before it
    returns.  ``durable=True`` additionally fsyncs every event; the default
    avoids turning per-step metric logging into a storage bottleneck.  Critical
    callers can use :meth:`sync` at checkpoints.
    """

    def __init__(
        self,
        text_path: str | Path,
        metrics_path: str | Path,
        *,
        run_id: str,
        durable: bool = False,
    ) -> None:
        self.text_path = Path(text_path)
        self.metrics_path = Path(metrics_path)
        self.run_id = str(run_id)
        self.durable = bool(durable)
        self._text: TextIO | None = None
        self._jsonl: TextIO | None = None
        self._lock = threading.RLock()
        self._sequence = 0

    def __enter__(self) -> RunJournal:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        with self._lock:
            if self._text is not None:
                return
            self.text_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._sequence, recovery = _repair_journal_tail(self.metrics_path)
            self._text = self.text_path.open(
                "a", encoding="utf-8", newline="\n", buffering=1
            )
            self._jsonl = self.metrics_path.open(
                "a", encoding="utf-8", newline="\n", buffering=1
            )
            if recovery is not None:
                self.warning(
                    "journal_tail_recovered",
                    "Repaired the machine-readable journal before append",
                    **recovery,
                )

    def close(self) -> None:
        with self._lock:
            if self._text is None:
                return
            self.sync(durable=self.durable)
            self._text.close()
            assert self._jsonl is not None
            self._jsonl.close()
            self._text = None
            self._jsonl = None

    def event(
        self,
        event: str,
        message: str = "",
        *,
        level: str = "INFO",
        metrics: Mapping[str, Any] | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        """Append one synchronized event to the text and JSONL streams."""

        with self._lock:
            self.open()
            self._sequence += 1
            row = {
                "timestamp": _utc_now(),
                "sequence": self._sequence,
                "run_id": self.run_id,
                "level": str(level).upper(),
                "event": str(event),
                "message": str(message),
                **payload,
            }
            if metrics is not None:
                row["metrics"] = dict(metrics)
            safe_row = _jsonable(row)
            ensure_finite_metrics(safe_row, context=f"journal event {event!r}")
            encoded = json.dumps(
                safe_row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            summary = (
                f"{safe_row['timestamp']} [{safe_row['level']}] "
                f"{safe_row['event']}"
            )
            if safe_row["message"]:
                summary += f" | {safe_row['message']}"
            details = {
                key: value
                for key, value in safe_row.items()
                if key
                not in {"timestamp", "sequence", "run_id", "level", "event", "message"}
            }
            if details:
                summary += " | " + json.dumps(
                    details,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            assert self._text is not None and self._jsonl is not None
            self._text.write(summary + "\n")
            self._jsonl.write(encoded + "\n")
            self._text.flush()
            self._jsonl.flush()
            if self.durable:
                os.fsync(self._text.fileno())
                os.fsync(self._jsonl.fileno())
            return safe_row

    def metric(self, event: str, metrics: Mapping[str, Any], **payload: Any) -> dict[str, Any]:
        return self.event(event, metrics=metrics, **payload)

    def warning(self, event: str, message: str, **payload: Any) -> dict[str, Any]:
        return self.event(event, message, level="WARNING", **payload)

    def fatal(self, event: str, message: str, **payload: Any) -> dict[str, Any]:
        return self.event(event, message, level="FATAL", **payload)

    def sync(self, *, durable: bool = True) -> None:
        with self._lock:
            if self._text is None or self._jsonl is None:
                return
            self._text.flush()
            self._jsonl.flush()
            if durable:
                os.fsync(self._text.fileno())
                os.fsync(self._jsonl.fileno())
