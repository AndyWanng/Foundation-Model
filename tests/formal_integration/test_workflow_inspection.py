from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mri_pet_geomc.formal.data import FormalDataError
from mri_pet_geomc.formal.runtime import ModelContractError
from mri_pet_geomc.formal.workflows import (
    _COMPANION_FALLBACK_FROM_SOURCE_SHA256,
    _COMPANION_FALLBACK_MIGRATION_ID,
    _RESUME_RUNTIME_BEHAVIOR,
    _formal_python_source_digest,
    _resolve_resume_python_source_contract,
    _inspect_training_cache_contracts,
    _recover_partial_generated_files,
)


def _config(cache_root: Path) -> dict[str, object]:
    return {
        "paths": {"cache_root": str(cache_root)},
        "preprocessing": {
            "target_shape": [128, 128, 128],
            "target_spacing_mm": [2.0, 2.0, 2.0],
        },
        "model": {"geometry": {"modes": 128}},
    }


def test_training_readiness_is_false_when_preprocessing_outputs_are_missing(
    tmp_path: Path,
) -> None:
    report = _inspect_training_cache_contracts(
        _config(tmp_path), assets_match=True
    )

    assert report["deep_validation_performed"] is False
    assert report["all_contracts_match"] is False
    assert "cache_index" in report["missing"]
    assert "Incomplete preprocessing outputs" in report["error"]


def test_training_readiness_fails_closed_for_marker_files_without_valid_content(
    tmp_path: Path,
) -> None:
    paths = (
        "catalog/catalog_contract.json",
        "catalog/observations.jsonl",
        "catalog/subject_splits.jsonl",
        "catalog/relations.jsonl",
        "cache_index.jsonl",
        "preprocessing_summary.json",
        "shard_plan.json",
        "cache_summary.json",
        "reference/target_reference.nii.gz",
        "reference/target_mask.nii.gz",
        "reference/reference_receipt.json",
        "geometry/fem_geometry.pt",
        "geometry/geometry_receipt.json",
    )
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    report = _inspect_training_cache_contracts(
        _config(tmp_path), assets_match=True
    )

    assert report["deep_validation_performed"] is True
    assert report["all_contracts_match"] is False
    assert report["missing"] == []
    assert report["error"]


def test_resume_recovers_only_known_partial_generated_files(tmp_path: Path) -> None:
    first = tmp_path / "observations.jsonl"
    second = tmp_path / "subject_splits.jsonl"
    first.write_text("partial", encoding="utf-8")

    with pytest.raises(FormalDataError, match="--no-resume"):
        _recover_partial_generated_files(
            (first, second), resume=False, description="catalog"
        )
    assert first.is_file()

    removed = _recover_partial_generated_files(
        (first, second), resume=True, description="catalog"
    )
    assert removed == (str(first),)
    assert not first.exists()


def _source_compatibility_fixture(
    tmp_path: Path, *, current_source_digest: str
) -> tuple[Path, dict[str, object]]:
    manifest = {
        "schema_version": 1,
        "migration_id": _COMPANION_FALLBACK_MIGRATION_ID,
        "from_python_source_sha256": _COMPANION_FALLBACK_FROM_SOURCE_SHA256,
        "to_python_source_sha256": current_source_digest,
        "checkpoint_contract_strategy": "retain_original_python_source_hash",
        "runtime_behavior": _RESUME_RUNTIME_BEHAVIOR,
        "scheduler_contract_correction": (
            "warmup_updates=steps_per_coverage*configured_warmup_coverages"
        ),
    }
    manifest_path = tmp_path / "RESUME_COMPATIBILITY.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    run_contract_path = tmp_path / "run_contract.json"
    run_contract_path.write_text(
        json.dumps(
            {"hashes": {"python_source": _COMPANION_FALLBACK_FROM_SOURCE_SHA256}}
        ),
        encoding="utf-8",
    )
    asset_report: dict[str, object] = {
        "checks": {
            "resume_compatibility": {
                "path": str(manifest_path),
                "matches": True,
                "actual_sha256": manifest_sha,
            }
        }
    }
    return run_contract_path, asset_report


def test_known_companion_fix_retains_old_checkpoint_source_contract(
    tmp_path: Path,
) -> None:
    current = "b" * 64
    run_contract_path, asset_report = _source_compatibility_fixture(
        tmp_path, current_source_digest=current
    )

    checkpoint_digest, migration = _resolve_resume_python_source_contract(
        project_root=tmp_path,
        run_contract_path=run_contract_path,
        current_source_digest=current,
        resume=True,
        asset_report=asset_report,
    )

    assert checkpoint_digest == _COMPANION_FALLBACK_FROM_SOURCE_SHA256
    assert migration is not None
    assert migration["active_python_source_sha256"] == current
    assert migration["frozen_checkpoint_python_source_sha256"] == checkpoint_digest
    assert migration["anchor_objective_unchanged"] is True
    assert migration["scheduler_contract_correction"].startswith("warmup_updates=")


def test_source_migration_requires_resume_and_exact_pinned_target(tmp_path: Path) -> None:
    current = "c" * 64
    run_contract_path, asset_report = _source_compatibility_fixture(
        tmp_path, current_source_digest=current
    )
    with pytest.raises(ModelContractError, match="--resume was not enabled"):
        _resolve_resume_python_source_contract(
            project_root=tmp_path,
            run_contract_path=run_contract_path,
            current_source_digest=current,
            resume=False,
            asset_report=asset_report,
        )
    with pytest.raises(ModelContractError, match="manifest mismatch"):
        _resolve_resume_python_source_contract(
            project_root=tmp_path,
            run_contract_path=run_contract_path,
            current_source_digest="d" * 64,
            resume=True,
            asset_report=asset_report,
        )


def test_bundled_legacy_resume_manifest_does_not_authorize_new_model_source(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    bundled = json.loads((project_root / "BUNDLED_ASSETS.json").read_text("utf-8"))
    entry = bundled["resume_compatibility"]
    manifest_path = project_root / entry["path"]
    manifest = json.loads(manifest_path.read_text("utf-8"))

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == entry["sha256"]
    assert manifest["migration_id"] == _COMPANION_FALLBACK_MIGRATION_ID
    assert (
        manifest["from_python_source_sha256"]
        == _COMPANION_FALLBACK_FROM_SOURCE_SHA256
    )
    # This manifest authorizes one historical support/scheduler fix, not every
    # future source revision.  Widening is a new run, not a blanket migration.
    current_digest = _formal_python_source_digest(project_root)
    assert manifest["to_python_source_sha256"] != current_digest
    run_contract_path = tmp_path / "run_contract.json"
    run_contract_path.write_text(
        json.dumps({"hashes": {"python_source": _COMPANION_FALLBACK_FROM_SOURCE_SHA256}}),
        encoding="utf-8",
    )
    asset_report = {
        "checks": {
            "resume_compatibility": {
                "path": str(manifest_path),
                "matches": True,
                "actual_sha256": entry["sha256"],
            }
        }
    }
    with pytest.raises(ModelContractError, match="manifest mismatch"):
        _resolve_resume_python_source_contract(
            project_root=project_root,
            run_contract_path=run_contract_path,
            current_source_digest=current_digest,
            resume=True,
            asset_report=asset_report,
        )
    assert _resolve_resume_python_source_contract(
        project_root=project_root,
        run_contract_path=tmp_path / "new-run" / "run_contract.json",
        current_source_digest=current_digest,
        resume=True,
        asset_report=asset_report,
    ) == (current_digest, None)
