from __future__ import annotations

import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..utils import atomic_torch_save, digest_object, sha256_file


BEST_BINDING_KEYS = {
    "best_candidate_name",
    "best_candidate_sha256",
    "best_candidate_step",
    "best_candidate_metric",
}


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    step: int,
    best_metric: float,
    best_step: int,
    contract: Mapping[str, Any],
    history_tail: list[dict[str, Any]],
    run_state: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "step": int(step),
        "best_metric": float(best_metric),
        "best_step": int(best_step),
        "contract": dict(contract),
        "contract_digest": digest_object(contract),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "rng": capture_rng_state(),
        "history_tail": history_tail[-100:],
        "run_state": dict(run_state or {}),
    }
    return atomic_torch_save(path, payload)


def checkpoint_identity(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Read immutable checkpoint identity without mutating a live model."""

    source = Path(path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise RuntimeError(f"Unsupported checkpoint payload: {source}")
    expected_digest = digest_object(expected_contract)
    if payload.get("contract_digest") != expected_digest:
        raise RuntimeError(
            f"Checkpoint contract mismatch: {source}; "
            f"found={payload.get('contract_digest')} expected={expected_digest}"
        )
    if not isinstance(payload.get("model"), Mapping):
        raise RuntimeError(f"Checkpoint has no model state mapping: {source}")
    try:
        step = int(payload["step"])
        best_step = int(payload["best_step"])
        best_metric = float(payload["best_metric"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Checkpoint identity fields are invalid: {source}") from error
    if step < 1 or best_step < 1 or not math.isfinite(best_metric):
        raise RuntimeError(
            f"Checkpoint identity is non-finite or non-positive: "
            f"step={step}, best_step={best_step}, best_metric={best_metric}"
        )
    return {
        "checkpoint_path": str(source),
        "checkpoint_sha256": sha256_file(source),
        "contract_digest": expected_digest,
        "step": step,
        "best_step": best_step,
        "best_metric": best_metric,
    }


def bind_best_checkpoint(
    run_state: Mapping[str, Any],
    candidate_path: str | Path,
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a last/final checkpoint to one immutable validation-best candidate."""

    source = Path(candidate_path).resolve()
    identity = checkpoint_identity(source, expected_contract=expected_contract)
    if identity["step"] != identity["best_step"]:
        raise RuntimeError(
            "A validation-best candidate must have step == best_step: "
            f"{identity}"
        )
    result = {key: value for key, value in dict(run_state).items() if key not in BEST_BINDING_KEYS}
    result.update(
        {
            "best_candidate_name": source.name,
            "best_candidate_sha256": identity["checkpoint_sha256"],
            "best_candidate_step": identity["step"],
            "best_candidate_metric": identity["best_metric"],
        }
    )
    return result


def validate_best_checkpoint_binding(
    run_state: Mapping[str, Any],
    checkpoint_dir: str | Path,
    *,
    expected_contract: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Fail closed unless the bound candidate matches hash, step, metric and contract."""

    missing = sorted(BEST_BINDING_KEYS - set(run_state))
    if missing:
        raise RuntimeError(f"Checkpoint run_state lacks validation-best binding: {missing}")
    name = str(run_state["best_candidate_name"])
    relative = Path(name)
    if relative.is_absolute() or relative.name != name or name in {"", ".", ".."}:
        raise RuntimeError(f"Unsafe validation-best candidate name: {name!r}")
    root = Path(checkpoint_dir).resolve()
    candidate = (root / relative).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise RuntimeError(f"Bound validation-best candidate is missing or escaped: {candidate}")
    identity = checkpoint_identity(candidate, expected_contract=expected_contract)
    expected_sha = str(run_state["best_candidate_sha256"])
    expected_step = int(run_state["best_candidate_step"])
    expected_metric = float(run_state["best_candidate_metric"])
    failures: list[str] = []
    if identity["checkpoint_sha256"] != expected_sha:
        failures.append(
            f"sha256 found={identity['checkpoint_sha256']} expected={expected_sha}"
        )
    if identity["step"] != expected_step or identity["best_step"] != expected_step:
        failures.append(
            f"step/best_step found={identity['step']}/{identity['best_step']} "
            f"expected={expected_step}"
        )
    if not math.isclose(
        float(identity["best_metric"]), expected_metric, rel_tol=0.0, abs_tol=1.0e-12
    ):
        failures.append(
            f"best_metric found={identity['best_metric']} expected={expected_metric}"
        )
    if failures:
        raise RuntimeError(
            "Validation-best checkpoint binding mismatch: " + "; ".join(failures)
        )
    return candidate, identity


def materialize_checkpoint(source_path: str | Path, destination_path: str | Path) -> Path:
    """Atomically copy an immutable candidate to the final checkpoint path."""

    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=4 * 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    expected_contract: Mapping[str, Any],
    restore_rng: bool,
) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    expected_digest = digest_object(expected_contract)
    if payload.get("contract_digest") != expected_digest:
        raise RuntimeError(
            f"Checkpoint contract mismatch: {source}; "
            f"found={payload.get('contract_digest')} expected={expected_digest}"
        )
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    payload["checkpoint_path"] = str(source)
    payload["checkpoint_sha256"] = sha256_file(source)
    return payload
