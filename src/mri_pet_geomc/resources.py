from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

import psutil
import torch


def snapshot(path: str | Path | None = None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path(path or ".").resolve()))
    payload: dict[str, Any] = {
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "ram_available_gib": memory.available / 2**30,
        "ram_percent": memory.percent,
        "disk_free_gib": disk.free / 2**30,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        payload.update(
            {
                "gpu_index": index,
                "gpu_name": torch.cuda.get_device_name(index),
                "gpu_free_gib": free / 2**30,
                "gpu_total_gib": total / 2**30,
                "cuda_runtime": torch.version.cuda,
            }
        )
    return payload


def require_resources(
    payload: dict[str, Any],
    *,
    require_cuda: bool,
    min_free_vram_gib: float,
    min_free_ram_gib: float,
    min_free_disk_gib: float,
) -> None:
    failures: list[str] = []
    if require_cuda and not payload.get("cuda_available"):
        failures.append("CUDA is required but unavailable")
    if require_cuda and float(payload.get("gpu_free_gib", 0.0)) < min_free_vram_gib:
        failures.append(
            f"free VRAM {payload.get('gpu_free_gib', 0.0):.2f} GiB < {min_free_vram_gib:.2f} GiB"
        )
    if float(payload["ram_available_gib"]) < min_free_ram_gib:
        failures.append(
            f"free RAM {payload['ram_available_gib']:.2f} GiB < {min_free_ram_gib:.2f} GiB"
        )
    if float(payload["disk_free_gib"]) < min_free_disk_gib:
        failures.append(
            f"free disk {payload['disk_free_gib']:.2f} GiB < {min_free_disk_gib:.2f} GiB"
        )
    if failures:
        raise RuntimeError("Resource preflight failed: " + "; ".join(failures))
