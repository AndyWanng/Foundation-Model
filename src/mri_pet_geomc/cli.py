from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import get, load_config, validate_config
from .pipeline import PipelineRunner, build_plan, plan_payload, verify_launch
from .model.trainable_sat3d import sat3d_source_tree_sha256
from .resources import require_resources, snapshot
from .utils import atomic_write_json


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _launch_dir(config: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    configured = get(config, "run.launch_dir")
    if configured:
        return Path(str(configured)).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = str(config["project"]["name"])
    return Path(config["paths"]["artifacts_root"]).resolve() / f"{name}_{timestamp}"


def environment_check(config: dict[str, Any], *, project_root: Path) -> dict[str, Any]:
    mode = str(config["run"]["mode"])
    # A fresh deployment has no output namespace yet.  Create only the new,
    # configured artifacts root before asking psutil for its filesystem; the
    # full-mode probes below still validate all writable output roots.
    artifacts_root = Path(str(config["paths"]["artifacts_root"])).resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    resources = snapshot(artifacts_root)
    require_resources(
        resources,
        require_cuda=bool(get(config, "resources.require_cuda", mode != "smoke")),
        min_free_vram_gib=float(get(config, "resources.min_free_vram_gib", 0.0)),
        min_free_ram_gib=float(get(config, "resources.min_free_ram_gib", 0.0)),
        min_free_disk_gib=float(get(config, "resources.min_free_disk_gib", 0.0)),
    )
    paths: dict[str, dict[str, Any]] = {}
    required = []
    if mode == "full":
        required = [
            "dataset_root",
            "legacy_pairings",
            "legacy_splits",
            "legacy_pet128_manifest",
            "sat3d_code_root",
            "sat3d_checkpoint",
            "model_reference_128",
            "model_mask_128",
        ]
    elif mode == "gpu_smoke_test":
        required = ["sat3d_code_root", "sat3d_checkpoint"]
    directory_keys = {"dataset_root", "sat3d_code_root"}
    for key in required:
        path = Path(str(config["paths"][key])).resolve()
        exists = path.is_dir() if key in directory_keys else path.is_file()
        paths[key] = {
            "path": str(path),
            "exists": exists,
            "expected_kind": "directory" if key in directory_keys else "file",
        }
        if not exists:
            raise FileNotFoundError(f"Required path does not exist: paths.{key}={path}")
    sat3d_source: dict[str, Any] | None = None
    if mode in {"full", "gpu_smoke_test"}:
        python_root, actual_source_sha = sat3d_source_tree_sha256(
            config["paths"]["sat3d_code_root"]
        )
        expected_source_sha = str(config["sat3d"]["expected_source_tree_sha256"])
        if actual_source_sha.lower() != expected_source_sha.lower():
            raise RuntimeError(
                "SAT3D source-tree SHA-256 mismatch: "
                f"expected={expected_source_sha}, actual={actual_source_sha}"
            )
        sat3d_source = {
            "python_root": str(python_root),
            "expected_sha256": expected_source_sha,
            "actual_sha256": actual_source_sha,
            "matched": True,
        }
    writable_roots: dict[str, dict[str, Any]] = {}
    if mode == "full":
        for key in ("artifacts_root", "derivatives_root", "cache_root"):
            root = Path(str(config["paths"][key])).resolve()
            root.mkdir(parents=True, exist_ok=True)
            probe = root / f".mri_pet_geomc_write_probe_{os.getpid()}"
            try:
                probe.write_text("write-probe\n", encoding="utf-8")
                writable = probe.is_file()
            finally:
                probe.unlink(missing_ok=True)
            if not writable:
                raise PermissionError(f"Configured output root is not writable: paths.{key}={root}")
            writable_roots[key] = {"path": str(root), "writable": True}
    commands: dict[str, str | None] = {}
    return {
        "schema_version": 1,
        "passed": True,
        "mode": mode,
        "project_root": str(project_root),
        "resources": resources,
        "paths": paths,
        "sat3d_source_tree": sat3d_source,
        "writable_roots": writable_roots,
        "commands": commands,
    }


def _handlers() -> dict[str, Any]:
    from .stages import HANDLERS

    return HANDLERS


def _full(args: argparse.Namespace, forced_mode: str | None = None) -> int:
    command_started_at = datetime.now(timezone.utc).isoformat()
    project_root = _project_root()
    config = load_config(args.config, project_root=project_root)
    if forced_mode:
        config["run"]["mode"] = forced_mode
        validate_config(config)
    launch_dir = _launch_dir(config, args.launch_dir)
    launch_dir.mkdir(parents=True, exist_ok=True)
    runner = PipelineRunner(
        project_root=project_root,
        launch_dir=launch_dir,
        config=config,
        handlers=_handlers(),
    )
    execution_error: dict[str, str] | None = None
    try:
        # Validate an existing launch's frozen identity before overwriting even
        # auxiliary environment-check/preview files.  New launches are initialized here so
        # an environment-check failure is still represented by an auditable failed state.
        runner.initialize()
        environment_result = environment_check(config, project_root=project_root)
        atomic_write_json(launch_dir / "environment_check.json", environment_result)
        atomic_write_json(launch_dir / "plan.preview.json", plan_payload(build_plan(config)))
        runner.run()
    except Exception as error:  # preserve an auditable failed launch instead of hiding it
        execution_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(launch_dir / "execution_error.json", execution_error)
    verification = verify_launch(launch_dir)
    verification["command_completed_without_fatal_error"] = execution_error is None
    verification["execution_succeeded"] = bool(
        execution_error is None
        and verification.get("all_requested_computations_succeeded", False)
    )
    if execution_error is not None:
        verification["execution_error"] = execution_error
        # A previously completed launch must never turn a refused resume into
        # a successful current command merely because its old stage records verify.
        verification["passed"] = False
    atomic_write_json(launch_dir / "verification.json", verification)
    atomic_write_json(
        launch_dir / "execution_status.json",
        {
            "schema_version": 1,
            "command_started_at": command_started_at,
            "command_finished_at": datetime.now(timezone.utc).isoformat(),
            "command_completed_without_fatal_error": execution_error is None,
            "all_requested_computations_succeeded": bool(
                verification.get("all_requested_computations_succeeded", False)
            ),
            "passed": bool(verification.get("passed", False)),
            "latest_error": execution_error,
            "historical_execution_error_exists": (
                launch_dir / "execution_error.json"
            ).is_file(),
        },
    )
    print(json.dumps(verification, indent=2, ensure_ascii=False))
    print(f"launch_dir={launch_dir}")
    return 0 if verification["passed"] else 2


def _gpu_smoke_test(args: argparse.Namespace) -> int:
    project_root = _project_root()
    config = load_config(args.config, project_root=project_root)
    config["run"]["mode"] = "gpu_smoke_test"
    launch_dir = _launch_dir(config, args.launch_dir)
    launch_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(launch_dir / "environment_check.json", environment_check(config, project_root=project_root))
    from .stages import run_geomc_gpu_smoke_test

    result = run_geomc_gpu_smoke_test(config, launch_dir)
    atomic_write_json(launch_dir / "gpu_smoke_test.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("passed") else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("full", "smoke", "plan", "environment-check"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--launch-dir")
    smoke_test = subparsers.add_parser("gpu-smoke-test")
    smoke_test.add_argument("--config", required=True)
    smoke_test.add_argument("--launch-dir")
    verify = subparsers.add_parser("verify")
    verify.add_argument("launch_dir")
    args = parser.parse_args(argv)
    if args.command == "full":
        return _full(args)
    if args.command == "smoke":
        return _full(args, forced_mode="smoke")
    if args.command == "gpu-smoke-test":
        return _gpu_smoke_test(args)
    if args.command == "verify":
        result = verify_launch(args.launch_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["passed"] else 2
    project_root = _project_root()
    config = load_config(args.config, project_root=project_root)
    if args.command == "environment-check":
        print(json.dumps(environment_check(config, project_root=project_root), indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(plan_payload(build_plan(config)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
