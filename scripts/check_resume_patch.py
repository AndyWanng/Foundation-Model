from __future__ import annotations

import argparse
import json
from pathlib import Path

from mri_pet_geomc.formal.config import load_formal_config
from mri_pet_geomc.formal.workflows import (
    _asset_report,
    _formal_python_source_digest,
    _project_root,
)


def _mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only verification for the pinned in-run runtime corrections"
    )
    parser.add_argument("--config", default="configs/workstation_formal.yaml")
    args = parser.parse_args()

    config = load_formal_config(args.config)
    project_root = _project_root(config)
    assets = _asset_report(config)
    if not assets["all_match"]:
        raise RuntimeError("Bundled asset verification failed")
    manifest = _mapping(project_root / "RESUME_COMPATIBILITY.json")
    current_source = _formal_python_source_digest(project_root)
    if manifest.get("to_python_source_sha256") != current_source:
        raise RuntimeError("Active Python source differs from the pinned patched source")

    run_root = Path(str(config["paths"]["run_root"]))
    contract_path = run_root / "contracts" / "run_contract.json"
    latest_path = run_root / "checkpoints" / "latest.json"
    run_contract = _mapping(contract_path)
    latest = _mapping(latest_path)
    hashes = run_contract.get("hashes")
    latest_hashes = latest.get("hashes")
    if not isinstance(hashes, dict) or latest_hashes != hashes:
        raise RuntimeError("latest checkpoint and frozen run-contract hashes differ")
    if hashes.get("config") != config["_resolved"]["digest"]:
        raise RuntimeError("Current resolved config differs from the frozen run config")

    frozen_source = str(hashes.get("python_source", ""))
    if frozen_source == current_source:
        resume_mode = "native_patched_source"
    elif frozen_source == manifest.get("from_python_source_sha256"):
        resume_mode = "pinned_pre_fix_checkpoint_migration"
    else:
        raise RuntimeError("Frozen run source is not covered by this compatibility patch")

    checkpoint_name = str(latest.get("checkpoint_name", ""))
    if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
        raise RuntimeError("latest.json contains an unsafe checkpoint name")
    checkpoint_path = (latest_path.parent / checkpoint_name).resolve()
    if checkpoint_path.parent != latest_path.parent.resolve() or not checkpoint_path.is_file():
        raise RuntimeError("latest checkpoint is missing or escaped its directory")
    expected_size = int(latest.get("checkpoint_size_bytes", -1))
    if checkpoint_path.stat().st_size != expected_size:
        raise RuntimeError("latest checkpoint size differs from latest.json")

    print(
        json.dumps(
            {
                "resume_compatible": True,
                "resume_mode": resume_mode,
                "migration_id": manifest.get("migration_id"),
                "runtime_behavior": manifest.get("runtime_behavior"),
                "frozen_python_source_sha256": frozen_source,
                "active_python_source_sha256": current_source,
                "config_digest": config["_resolved"]["digest"],
                "epochs": int(config["training"]["epochs"]),
                "cursor": latest.get("cursor"),
                "checkpoint": str(checkpoint_path),
                "checkpoint_size_bytes": expected_size,
                "checkpoint_sha256_check": "deferred_to_strict_training_resume",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
