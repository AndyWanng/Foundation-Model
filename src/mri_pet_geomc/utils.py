from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def digest_object(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    return atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows),
    )


def atomic_torch_save(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, destination)
    return destination


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_tree_digest(project_root: str | Path) -> str:
    root = Path(project_root)
    selected: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(
            part
            in {
                ".pytest_cache",
                ".codex_test_deps",
                "__pycache__",
                ".git",
                "build",
                "dist",
            }
            or part.lower().endswith((".egg-info", ".dist-info"))
            or part.lower().startswith(("artifacts", "cache", "derivatives"))
            or part.lower().startswith((".venv", "venv"))
            for part in relative.parts
        ):
            continue
        if path.suffix.lower() not in {".py", ".yaml", ".yml", ".toml", ".txt", ".md", ".sh", ".ps1"}:
            continue
        selected.append((relative.as_posix(), sha256_file(path)))
    return digest_object(selected)


class EventLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        row = {"timestamp": utc_now(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
