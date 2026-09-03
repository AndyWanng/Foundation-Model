from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import re
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import ModelContractError, ResumeMismatchError, ensure_finite_metrics
from .journal import RunJournal


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_contract(hashes: Mapping[str, str]) -> dict[str, str]:
    result = {str(key): str(value) for key, value in hashes.items()}
    if not result or any(not key or not value for key, value in result.items()):
        raise ResumeMismatchError("Checkpoint hashes require non-empty keys and values")
    return result


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs: Any) -> Any: ...


class CheckpointSerializer(Protocol):
    suffix: str

    def dump(self, value: Mapping[str, Any], path: Path) -> None: ...

    def load(self, path: Path) -> Mapping[str, Any]: ...


class PickleSerializer:
    """Low-dependency serializer for tests and non-tensor protocol backends."""

    suffix = ".pkl"

    def dump(self, value: Mapping[str, Any], path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(dict(value), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path) -> Mapping[str, Any]:
        with path.open("rb") as handle:
            value = pickle.load(handle)  # noqa: S301 - trusted local run artifact
        if not isinstance(value, Mapping):
            raise ModelContractError("Checkpoint root is not a mapping")
        return value


class TorchSerializer:
    """Lazy PyTorch serializer used by the workstation training process."""

    suffix = ".pt"

    def dump(self, value: Mapping[str, Any], path: Path) -> None:
        import torch

        torch.save(dict(value), path)

    def load(self, path: Path) -> Mapping[str, Any]:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(value, Mapping):
            raise ModelContractError("Checkpoint root is not a mapping")
        return value


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    """Capture available RNGs without initializing CUDA solely for bookkeeping."""

    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if include_cuda and torch.cuda.is_initialized():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" not in state:
        raise ResumeMismatchError("Checkpoint RNG state has no Python RNG")
    random.setstate(state["python"])
    if "numpy" in state:
        try:
            import numpy as np
        except ImportError as error:
            raise ResumeMismatchError("Checkpoint requires NumPy RNG state") from error
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        try:
            import torch
        except ImportError as error:
            raise ResumeMismatchError("Checkpoint requires PyTorch RNG state") from error
        torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state:
            if not torch.cuda.is_available():
                raise ResumeMismatchError(
                    "Checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass(frozen=True)
class CheckpointCursor:
    """Exact next-work cursor, persisted only at accumulation boundaries."""

    epoch: int
    coverage: int
    next_anchor_index: int
    global_step: int
    optimizer_step: int
    accumulation_index: int = 0

    def __post_init__(self) -> None:
        values = self.as_dict()
        if any(value < 0 for value in values.values()):
            raise ValueError(f"Checkpoint cursor values must be non-negative: {values}")
        if self.accumulation_index != 0:
            raise ValueError(
                "Formal checkpoints are allowed only after an optimizer step; "
                "accumulation_index must be zero"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "epoch": int(self.epoch),
            "coverage": int(self.coverage),
            "next_anchor_index": int(self.next_anchor_index),
            "global_step": int(self.global_step),
            "optimizer_step": int(self.optimizer_step),
            "accumulation_index": int(self.accumulation_index),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointCursor:
        try:
            return cls(
                epoch=int(value["epoch"]),
                coverage=int(value["coverage"]),
                next_anchor_index=int(value["next_anchor_index"]),
                global_step=int(value["global_step"]),
                optimizer_step=int(value["optimizer_step"]),
                accumulation_index=int(value.get("accumulation_index", 0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ResumeMismatchError("Checkpoint cursor is malformed") from error


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    sha256: str
    cursor: CheckpointCursor
    metrics: Mapping[str, Any]
    extra: Mapping[str, Any]
    created_at: str


class AtomicCheckpointManager:
    """Atomic, hash-bound training checkpoints with a durable latest pointer."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        directory: str | Path,
        *,
        serializer: CheckpointSerializer | None = None,
        journal: RunJournal | None = None,
        prefix: str = "formal-training",
    ) -> None:
        self.directory = Path(directory)
        self.serializer = serializer or TorchSerializer()
        self.journal = journal
        self.prefix = str(prefix)
        if not self.prefix or Path(self.prefix).name != self.prefix:
            raise ValueError(f"Unsafe checkpoint prefix: {self.prefix!r}")
        self.latest_path = self.directory / "latest.json"

    def save(
        self,
        *,
        cursor: CheckpointCursor,
        hashes: Mapping[str, str],
        model: Stateful,
        optimizer: Stateful,
        ema: Stateful,
        scheduler: Stateful | None = None,
        scaler: Stateful | None = None,
        extra_stateful: Mapping[str, Stateful] | None = None,
        metrics: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        rng_state: Mapping[str, Any] | None = None,
    ) -> Path:
        contract = _hash_contract(hashes)
        recorded_metrics = dict(metrics or {})
        recorded_extra = dict(extra or {})
        ensure_finite_metrics(recorded_metrics, context="checkpoint metrics")
        ensure_finite_metrics(recorded_extra, context="checkpoint extra payload")
        named_stateful = dict(extra_stateful or {})
        if any(not name for name in named_stateful):
            raise ValueError("extra_stateful names must be non-empty")
        created_at = _utc_now()
        payload: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "kind": "formal_training_checkpoint",
            "created_at": created_at,
            "epoch": cursor.epoch,
            "coverage": cursor.coverage,
            "cursor": cursor.as_dict(),
            "hashes": contract,
            "rng": dict(rng_state or capture_rng_state()),
            "states": {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
                "extra": {
                    name: target.state_dict() for name, target in named_stateful.items()
                },
            },
            "metrics": recorded_metrics,
            "extra": recorded_extra,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        phase = (
            "final"
            if bool(recorded_extra.get("final_complete"))
            else "coverage"
            if bool(recorded_extra.get("coverage_complete"))
            else "anchors"
            if bool(recorded_extra.get("training_anchors_complete"))
            else "step"
        )
        # A committed checkpoint is immutable.  In particular, the
        # anchors-complete and validation-complete commits can share the same
        # optimizer cursor, so cursor-only names would overwrite the file
        # still referenced by latest.json before its pointer update commits.
        commit_id = uuid.uuid4().hex
        name = (
            f"{self.prefix}-c{cursor.coverage:03d}-"
            f"s{cursor.optimizer_step:09d}-p{phase}-k{commit_id}"
            f"{self.serializer.suffix}"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=self.directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            self.serializer.dump(payload, temporary)
            with temporary.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            checksum = _sha256(temporary)
            size_bytes = temporary.stat().st_size
            # Publish with an atomic no-clobber operation.  ``os.replace`` is
            # intentionally not used: if latest.json still names an older
            # same-cursor commit, overwriting that target before the pointer
            # update would make resume impossible after a crash.  The temp
            # file lives in this directory, so the hard-link commit remains
            # on one filesystem.  A practically impossible UUID collision is
            # still handled without ever replacing an existing checkpoint.
            while True:
                destination = self.directory / name
                try:
                    os.link(temporary, destination)
                    break
                except FileExistsError:
                    commit_id = uuid.uuid4().hex
                    name = (
                        f"{self.prefix}-c{cursor.coverage:03d}-"
                        f"s{cursor.optimizer_step:09d}-p{phase}-k{commit_id}"
                        f"{self.serializer.suffix}"
                    )
            pointer = {
                "schema_version": self.SCHEMA_VERSION,
                "checkpoint_name": destination.name,
                "checkpoint_sha256": checksum,
                "checkpoint_size_bytes": size_bytes,
                "cursor": cursor.as_dict(),
                "hashes": contract,
                "created_at": created_at,
            }
            self._atomic_write_json(self.latest_path, pointer)
        finally:
            temporary.unlink(missing_ok=True)
        if self.journal is not None:
            self.journal.event(
                "checkpoint_saved",
                "Saved atomic formal-training checkpoint",
                checkpoint_path=str(destination),
                checkpoint_sha256=checksum,
                checkpoint_size_bytes=size_bytes,
                cursor=cursor.as_dict(),
            )
            self.journal.sync()
        return destination

    def load_latest(
        self,
        *,
        expected_hashes: Mapping[str, str],
        model: Stateful,
        optimizer: Stateful,
        ema: Stateful,
        scheduler: Stateful | None = None,
        scaler: Stateful | None = None,
        extra_stateful: Mapping[str, Stateful] | None = None,
        restore_rng: bool = True,
    ) -> LoadedCheckpoint:
        pointer = self._read_pointer()
        expected = _hash_contract(expected_hashes)
        found_pointer_hashes = _hash_contract(pointer.get("hashes", {}))
        if found_pointer_hashes != expected:
            raise ResumeMismatchError(
                "Latest checkpoint pointer hash contract mismatch: "
                f"found={found_pointer_hashes} expected={expected}"
            )
        name = str(pointer.get("checkpoint_name", ""))
        if not name or Path(name).name != name:
            raise ResumeMismatchError(f"Unsafe checkpoint name in latest pointer: {name!r}")
        path = (self.directory / name).resolve()
        if path.parent != self.directory.resolve() or not path.is_file():
            raise ResumeMismatchError(f"Latest checkpoint is missing or escaped: {path}")
        checksum = _sha256(path)
        if checksum != str(pointer.get("checkpoint_sha256", "")):
            raise ResumeMismatchError(
                f"Checkpoint checksum mismatch: found={checksum} "
                f"expected={pointer.get('checkpoint_sha256')}"
            )
        payload = self.serializer.load(path)
        self._validate_payload(payload, expected, pointer)
        cursor = CheckpointCursor.from_dict(payload["cursor"])
        states = payload["states"]
        assert isinstance(states, Mapping)
        try:
            self._load_required(model, states["model"], strict=True)
            self._load_required(optimizer, states["optimizer"], strict=False)
            self._load_required(ema, states["ema"], strict=True)
            self._load_optional("scheduler", scheduler, states.get("scheduler"))
            self._load_optional("scaler", scaler, states.get("scaler"))
            saved_extra = states.get("extra", {})
            if not isinstance(saved_extra, Mapping):
                raise ModelContractError("Checkpoint extra state is not a mapping")
            expected_extra = dict(extra_stateful or {})
            if set(saved_extra) != set(expected_extra):
                raise ModelContractError(
                    "Checkpoint extra-state names differ: "
                    f"found={sorted(saved_extra)} expected={sorted(expected_extra)}"
                )
            for key, target in expected_extra.items():
                self._load_required(target, saved_extra[key], strict=True)
        except (ModelContractError, ResumeMismatchError):
            raise
        except Exception as error:
            raise ModelContractError("Strict checkpoint state restoration failed") from error
        if restore_rng:
            restore_rng_state(payload["rng"])
        loaded = LoadedCheckpoint(
            path=path,
            sha256=checksum,
            cursor=cursor,
            metrics=dict(payload.get("metrics", {})),
            extra=dict(payload.get("extra", {})),
            created_at=str(payload["created_at"]),
        )
        if self.journal is not None:
            self.journal.event(
                "checkpoint_loaded",
                "Strictly restored the latest atomic checkpoint",
                checkpoint_path=str(path),
                checkpoint_sha256=checksum,
                cursor=cursor.as_dict(),
            )
            self.journal.sync()
        return loaded

    def prune(
        self,
        *,
        keep_last: int = 3,
        keep_every_coverages: int | None = 5,
        keep_names: tuple[str, ...] = (),
    ) -> tuple[Path, ...]:
        """Safely remove redundant checkpoints after validating the latest one.

        Deletion is restricted to exact legacy
        ``prefix-cNNN-sNNN<suffix>`` files or immutable
        ``prefix-cNNN-sNNN-pPHASE-kCOMMIT<suffix>`` files in this manager's
        directory.  The latest checkpoint, the newest ``keep_last`` files,
        one final checkpoint at each requested coverage milestone, and
        explicitly named files are retained.
        """

        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        if keep_every_coverages is not None and keep_every_coverages < 1:
            raise ValueError("keep_every_coverages must be positive or None")
        pointer = self._read_pointer()
        latest_name = str(pointer.get("checkpoint_name", ""))
        if not latest_name or Path(latest_name).name != latest_name:
            raise ResumeMismatchError(
                f"Unsafe checkpoint name in latest pointer: {latest_name!r}"
            )
        root = self.directory.resolve()
        latest = (root / latest_name).resolve()
        if latest.parent != root or not latest.is_file():
            raise ResumeMismatchError(f"Latest checkpoint is missing or escaped: {latest}")
        latest_sha = _sha256(latest)
        if latest_sha != str(pointer.get("checkpoint_sha256", "")):
            raise ResumeMismatchError(
                "Refusing checkpoint retention because latest SHA256 does not match"
            )
        explicit = set()
        for name in keep_names:
            value = str(name)
            if not value or Path(value).name != value:
                raise ValueError(f"Unsafe explicit checkpoint name: {value!r}")
            explicit.add(value)
        pattern = re.compile(
            rf"^{re.escape(self.prefix)}-c(?P<coverage>\d{{3}})-"
            rf"s(?P<step>\d{{9}})"
            rf"(?:-p(?P<phase>step|anchors|coverage|final)-"
            rf"k(?P<commit>[0-9a-f]{{32}}))?"
            rf"{re.escape(self.serializer.suffix)}$"
        )
        phase_rank = {None: 0, "step": 1, "anchors": 2, "coverage": 3, "final": 4}
        candidates: list[tuple[int, int, int, int, Path]] = []
        if self.directory.is_dir():
            for path in self.directory.iterdir():
                if not path.is_file():
                    continue
                match = pattern.fullmatch(path.name)
                if match is None:
                    continue
                resolved = path.resolve()
                if resolved.parent != root:
                    raise ResumeMismatchError(
                        f"Checkpoint candidate escaped retention directory: {resolved}"
                    )
                candidates.append(
                    (
                        int(match.group("coverage")),
                        int(match.group("step")),
                        phase_rank[match.group("phase")],
                        int(path.stat().st_mtime_ns),
                        resolved,
                    )
                )
        candidates.sort(
            key=lambda item: (item[1], item[0], item[2], item[3], item[4].name)
        )
        retained = {latest.name, *explicit}
        retained.update(path.name for _, _, _, _, path in candidates[-keep_last:])
        if keep_every_coverages is not None:
            by_coverage: dict[int, tuple[int, int, int, str, Path]] = {}
            for coverage, step, rank, modified, path in candidates:
                if coverage % keep_every_coverages:
                    continue
                current = by_coverage.get(coverage)
                key = (step, rank, modified, path.name)
                if current is None or key > current[:4]:
                    by_coverage[coverage] = (*key, path)
            retained.update(value[4].name for value in by_coverage.values())
        removed: list[Path] = []
        for _, _, _, _, path in candidates:
            if path.name in retained:
                continue
            # Recheck the exact target immediately before the destructive call.
            if path.parent != root or pattern.fullmatch(path.name) is None:
                raise ResumeMismatchError(f"Unsafe checkpoint prune target: {path}")
            path.unlink()
            removed.append(path)
        if self.journal is not None:
            self.journal.event(
                "checkpoint_retention_completed",
                "Pruned redundant formal-training checkpoints",
                latest_checkpoint=latest.name,
                kept_checkpoint_count=len(candidates) - len(removed),
                removed_checkpoint_names=[path.name for path in removed],
                keep_last=keep_last,
                keep_every_coverages=keep_every_coverages,
                explicit_keep_names=sorted(explicit),
            )
            self.journal.sync()
        return tuple(removed)

    def _validate_payload(
        self,
        payload: Mapping[str, Any],
        expected_hashes: Mapping[str, str],
        pointer: Mapping[str, Any],
    ) -> None:
        if int(payload.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ResumeMismatchError("Unsupported formal-training checkpoint schema")
        if payload.get("kind") != "formal_training_checkpoint":
            raise ResumeMismatchError("Checkpoint kind is not formal training")
        found = _hash_contract(payload.get("hashes", {}))
        if found != dict(expected_hashes):
            raise ResumeMismatchError(
                f"Checkpoint hash contract mismatch: found={found} expected={expected_hashes}"
            )
        cursor = CheckpointCursor.from_dict(payload.get("cursor", {}))
        pointer_cursor = CheckpointCursor.from_dict(pointer.get("cursor", {}))
        if cursor != pointer_cursor:
            raise ResumeMismatchError("Latest pointer cursor differs from checkpoint cursor")
        if int(payload.get("epoch", -1)) != cursor.epoch:
            raise ResumeMismatchError("Checkpoint epoch disagrees with its cursor")
        if int(payload.get("coverage", -1)) != cursor.coverage:
            raise ResumeMismatchError("Checkpoint coverage disagrees with its cursor")
        states = payload.get("states")
        if not isinstance(states, Mapping):
            raise ModelContractError("Checkpoint has no state mapping")
        for required in ("model", "optimizer", "ema"):
            if required not in states or not isinstance(states[required], Mapping):
                raise ModelContractError(f"Checkpoint lacks required {required} state")
        if not isinstance(payload.get("rng"), Mapping):
            raise ResumeMismatchError("Checkpoint lacks RNG state")
        ensure_finite_metrics(payload.get("metrics", {}), context="loaded checkpoint metrics")
        ensure_finite_metrics(payload.get("extra", {}), context="loaded checkpoint extra")

    @staticmethod
    def _load_required(target: Stateful, state: Any, *, strict: bool) -> None:
        if not isinstance(state, Mapping):
            raise ModelContractError("Required checkpoint state is not a mapping")
        if strict:
            try:
                target.load_state_dict(state, strict=True)
                return
            except TypeError:
                pass
        target.load_state_dict(state)

    @classmethod
    def _load_optional(cls, name: str, target: Stateful | None, state: Any) -> None:
        if (target is None) != (state is None):
            raise ModelContractError(
                f"Checkpoint {name} presence differs from the live training backend"
            )
        if target is not None:
            cls._load_required(target, state, strict=False)

    def _read_pointer(self) -> Mapping[str, Any]:
        try:
            value = json.loads(self.latest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as error:
            raise ResumeMismatchError(
                f"Cannot read latest checkpoint pointer: {self.latest_path}"
            ) from error
        if not isinstance(value, Mapping):
            raise ResumeMismatchError("Latest checkpoint pointer is not a JSON object")
        return value

    @staticmethod
    def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
