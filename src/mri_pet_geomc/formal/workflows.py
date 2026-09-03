"""End-to-end workstation workflows for the first formal 24k run.

The public CLI imports this module lazily.  Inspection is read-only;
preprocessing and training write only under their configured cache/run roots.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from ..geometry.fem import FEMGeometry, build_fem_geometry
from ..model.trainable_sat3d import sat3d_source_tree_sha256
from ..utils import (
    atomic_write_json,
    digest_object,
    read_json,
    set_deterministic_seed,
    sha256_file,
    utc_now,
)
from .data.cache import load_cache_index
from .data.catalog import CatalogBuildResult, build_catalog, load_catalog
from .data.dataset import FormalVolumeDataset, load_preprocessing_overlay
from .data.inventory import PhysicalInventory
from .data.preprocessing import (
    OfflinePreprocessor,
    PreprocessProgress,
    PreprocessingConfig,
    preprocessing_contract_digest,
)
from .data.reference import ReferenceSpec, reference_contract_digest
from .data.relations import RelationBuildResult, build_relations, load_relations
from .data.schema import FormalDataError, ObservationRelation, validate_relations
from .runtime import (
    AtomicCheckpointManager,
    BatchContract,
    CoverageValidationCadence,
    DefaultResourceProvider,
    FormalTrainingSchedule,
    ModelContractError,
    RelationMixture,
    RunJournal,
    TelemetryRecorder,
    TerminalProgress,
    TrainingRuntimeController,
    ValidationRequest,
    ValidationTier,
)


_COMPANION_FALLBACK_MIGRATION_ID = (
    "companion-fallback-and-configured-horizon-scheduler-v1"
)
_COMPANION_FALLBACK_FROM_SOURCE_SHA256 = (
    "56fb7d23ddbc02e7486bd9861ad78c42f5a0a50e184e1268f9c6e32a66d74060"
)
_RESUME_RUNTIME_BEHAVIOR = (
    "drop_only_optional_companion_when_anchor_coupled_mask_leaves_zero_visible_support;"
    "derive_warmup_updates_from_configured_coverage_horizon"
)


def _path(config: Mapping[str, Any], key: str) -> Path:
    return Path(str(config["paths"][key]))


def _project_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["_resolved"]["project_root"]))


def _metadata_bundle_digest(project_root: Path, metadata_root: Path) -> tuple[int, int, str]:
    files = sorted(path for path in metadata_root.rglob("*") if path.is_file())
    lines = []
    total_bytes = 0
    for path in files:
        total_bytes += path.stat().st_size
        relative = path.relative_to(project_root).as_posix()
        lines.append(f"{relative}\t{sha256_file(path)}\t{path.stat().st_size}")
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return len(files), total_bytes, digest


def _asset_report(config: Mapping[str, Any]) -> dict[str, Any]:
    project_root = _project_root(config)
    manifest_path = project_root / "BUNDLED_ASSETS.json"
    expected = read_json(manifest_path) if manifest_path.is_file() else {}
    report: dict[str, Any] = {
        "manifest": str(manifest_path),
        "manifest_exists": manifest_path.is_file(),
        "checks": {},
    }

    def file_check(name: str, path: Path, expected_sha: str) -> None:
        exists = path.is_file()
        actual = sha256_file(path) if exists else ""
        report["checks"][name] = {
            "path": str(path),
            "exists": exists,
            "expected_sha256": expected_sha,
            "actual_sha256": actual,
            "matches": bool(exists and expected_sha and actual == expected_sha),
        }

    sat_expected = dict(expected.get("sat3d") or {})
    reference_expected = dict(expected.get("reference") or {})
    compatibility_expected = dict(expected.get("resume_compatibility") or {})
    file_check(
        "sat3d_checkpoint",
        _path(config, "sat3d_checkpoint"),
        str(sat_expected.get("checkpoint_sha256", "")),
    )
    file_check(
        "source_reference",
        _path(config, "source_reference"),
        str(reference_expected.get("image_sha256", "")),
    )
    file_check(
        "source_reference_mask",
        _path(config, "source_reference_mask"),
        str(reference_expected.get("mask_sha256", "")),
    )
    compatibility_relative = Path(str(compatibility_expected.get("path", "")))
    compatibility_safe = bool(
        str(compatibility_relative)
        and not compatibility_relative.is_absolute()
        and ".." not in compatibility_relative.parts
    )
    compatibility_path = (
        project_root / compatibility_relative
        if compatibility_safe
        else project_root / ".invalid-resume-compatibility-path"
    )
    file_check(
        "resume_compatibility",
        compatibility_path,
        str(compatibility_expected.get("sha256", "")),
    )
    report["checks"]["resume_compatibility"]["safe_relative_path"] = (
        compatibility_safe
    )
    report["checks"]["resume_compatibility"]["matches"] = bool(
        compatibility_safe
        and report["checks"]["resume_compatibility"]["matches"]
    )
    code_root = _path(config, "sat3d_code_root")
    try:
        resolved_root, tree_digest = sat3d_source_tree_sha256(code_root)
        tree_error = ""
    except Exception as error:  # inspection must report, not hide, a missing tree
        resolved_root, tree_digest, tree_error = code_root, "", str(error)
    expected_tree = str(sat_expected.get("python_source_tree_sha256", ""))
    report["checks"]["sat3d_source_tree"] = {
        "path": str(resolved_root),
        "exists": resolved_root.is_dir(),
        "expected_sha256": expected_tree,
        "actual_sha256": tree_digest,
        "matches": bool(tree_digest and tree_digest == expected_tree),
        "error": tree_error,
    }
    metadata_root = _path(config, "metadata_root")
    if metadata_root.is_dir():
        count, size_bytes, metadata_digest = _metadata_bundle_digest(
            project_root, metadata_root
        )
    else:
        count, size_bytes, metadata_digest = 0, 0, ""
    expected_metadata = dict(expected.get("metadata") or {})
    report["checks"]["metadata_bundle"] = {
        "path": str(metadata_root),
        "exists": metadata_root.is_dir(),
        "file_count": count,
        "bytes": size_bytes,
        "expected_sha256": str(expected_metadata.get("bundle_sha256", "")),
        "actual_sha256": metadata_digest,
        "matches": bool(
            metadata_digest
            and metadata_digest == str(expected_metadata.get("bundle_sha256", ""))
            and count == int(expected_metadata.get("file_count", -1))
        ),
    }
    report["all_match"] = bool(
        report["manifest_exists"]
        and report["checks"]
        and all(bool(item.get("matches")) for item in report["checks"].values())
    )
    return report


def _inspect_training_cache_contracts(
    config: Mapping[str, Any],
    *,
    assets_match: bool,
) -> dict[str, Any]:
    """Deeply validate frozen training inputs without allocating the model.

    This deliberately repeats the real-training preflight.  ``inspect`` is a
    user-facing readiness command, so a few marker files are not enough to
    claim that training can start safely.
    """

    cache_root = _path(config, "cache_root")
    required = {
        "catalog_contract": cache_root / "catalog" / "catalog_contract.json",
        "observations": cache_root / "catalog" / "observations.jsonl",
        "subject_splits": cache_root / "catalog" / "subject_splits.jsonl",
        "relations": cache_root / "catalog" / "relations.jsonl",
        "cache_index": cache_root / "cache_index.jsonl",
        "preprocessing_summary": cache_root / "preprocessing_summary.json",
        "shard_plan": cache_root / "shard_plan.json",
        "cache_summary": cache_root / "cache_summary.json",
        "reference": cache_root / "reference" / "target_reference.nii.gz",
        "reference_mask": cache_root / "reference" / "target_mask.nii.gz",
        "reference_receipt": cache_root / "reference" / "reference_receipt.json",
        "geometry": cache_root / "geometry" / "fem_geometry.pt",
        "geometry_receipt": cache_root / "geometry" / "geometry_receipt.json",
    }
    files = {
        name: {"path": str(path), "exists": path.is_file()}
        for name, path in required.items()
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    report: dict[str, Any] = {
        "deep_validation_performed": False,
        "all_contracts_match": False,
        "files": files,
        "missing": missing,
        "error": "",
        "counts": {},
    }
    if not assets_match:
        report["error"] = "Bundled asset hashes do not match"
        return report
    if missing:
        report["error"] = "Incomplete preprocessing outputs: " + ", ".join(missing)
        return report

    report["deep_validation_performed"] = True
    try:
        catalog = load_catalog(required["observations"], required["subject_splits"])
        relations = load_relations(required["relations"])
        validate_relations(relations, catalog.observations)
        catalog_receipt = read_json(required["catalog_contract"])
        if not isinstance(catalog_receipt, Mapping):
            raise FormalDataError("catalog_contract.json must be an object")
        if catalog_receipt.get("catalog_digest") != catalog.digest:
            raise FormalDataError("Frozen catalog differs from catalog_contract.json")
        if catalog_receipt.get("relations_digest") != _relation_digest(relations):
            raise FormalDataError("Frozen relations differ from catalog_contract.json")

        cache_entries = load_cache_index(required["cache_index"])
        if not cache_entries:
            raise FormalDataError("cache_index is empty; no observation can be trained")
        cached_uids = {entry.observation_uid for entry in cache_entries}
        catalog_uids = {row.observation_uid for row in catalog.observations}
        if len(cached_uids) != len(cache_entries) or not cached_uids <= catalog_uids:
            raise FormalDataError("cache_index UIDs are duplicated or outside the catalog")
        preprocessing_summary = read_json(required["preprocessing_summary"])
        if not isinstance(preprocessing_summary, Mapping):
            raise FormalDataError("preprocessing_summary.json must be an object")
        cache_contract = str(preprocessing_summary.get("contract_digest", ""))
        cache_plan = str(preprocessing_summary.get("plan_digest", ""))
        if not cache_contract or not cache_plan:
            raise FormalDataError("Preprocessing summary lacks cache contracts")
        expected_reference_contract = _expected_reference_contract(config)
        if (
            preprocessing_summary.get("reference_contract_digest")
            != expected_reference_contract
        ):
            raise FormalDataError(
                "Preprocessing summary is not bound to the configured reference"
            )
        expected_cache_contract = preprocessing_contract_digest(
            _formal_preprocessing_config(
                config, cache_root=cache_root, retry_failures=True
            ),
            expected_reference_contract,
        )
        if cache_contract != expected_cache_contract:
            raise FormalDataError(
                "Preprocessing cache contract differs from reference/config"
            )
        if any(entry.contract_digest != cache_contract for entry in cache_entries):
            raise FormalDataError("cache_index contains a different preprocessing contract")
        cache_layout = _validate_cache_layout(
            cache_root=cache_root,
            cache_entries=cache_entries,
            target_shape=config["preprocessing"]["target_shape"],
            expected_contract_digest=cache_contract,
            expected_plan_digest=cache_plan,
        )
        if cache_layout["shard_plan"].get("plan_digest") != cache_plan:
            raise FormalDataError("shard_plan differs from preprocessing_summary")

        reference_receipt = read_json(required["reference_receipt"])
        if not isinstance(reference_receipt, Mapping):
            raise FormalDataError("reference_receipt.json must be an object")
        if reference_receipt.get("contract_digest") != expected_reference_contract:
            raise FormalDataError("Reference receipt contract differs from bundled inputs")
        reference_outputs = reference_receipt.get("outputs") or {}
        if not isinstance(reference_outputs, Mapping):
            raise FormalDataError("Reference receipt outputs must be an object")
        if (
            reference_outputs.get("reference_sha256")
            != sha256_file(required["reference"])
            or reference_outputs.get("mask_sha256")
            != sha256_file(required["reference_mask"])
        ):
            raise FormalDataError("Target reference/mask differs from its receipt")
        reference_image = nib.load(required["reference"])
        if tuple(int(value) for value in reference_image.shape) != tuple(
            int(value) for value in config["preprocessing"]["target_shape"]
        ):
            raise FormalDataError("Target reference shape differs from formal config")
        if not np.allclose(
            nib.affines.voxel_sizes(reference_image.affine),
            np.asarray(config["preprocessing"]["target_spacing_mm"], dtype=float),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise FormalDataError("Target reference spacing differs from formal config")

        geometry_receipt = read_json(required["geometry_receipt"])
        if not isinstance(geometry_receipt, Mapping):
            raise FormalDataError("geometry_receipt.json must be an object")
        if geometry_receipt.get("contract_digest") != _geometry_contract_digest(
            config, expected_reference_contract
        ):
            raise FormalDataError("FEM geometry contract differs from reference/config")
        if geometry_receipt.get("geometry_sha256") != sha256_file(required["geometry"]):
            raise FormalDataError("FEM geometry differs from its receipt")
        if int(geometry_receipt.get("modes", -1)) != int(
            config["model"]["geometry"]["modes"]
        ):
            raise FormalDataError("FEM mode count differs from formal config")

        split_counts = Counter(
            row.split for row in catalog.observations if row.observation_uid in cached_uids
        )
        split_modality_counts = Counter(
            (row.split, str(row.modality).strip().lower())
            for row in catalog.observations
            if row.observation_uid in cached_uids
        )
        for split in ("train", "validation", "test"):
            if split_counts[split] < 1:
                raise FormalDataError(
                    f"No successfully cached observation remains in {split}"
                )
            for modality in ("mri", "pet"):
                if split_modality_counts[(split, modality)] < 1:
                    raise FormalDataError(
                        f"No successfully cached {modality.upper()} remains in {split}"
                    )
        report["counts"] = {
            "catalog_observations": len(catalog.observations),
            "cached_observations": len(cache_entries),
            "relations": len(relations),
            "train": split_counts["train"],
            "validation": split_counts["validation"],
            "test": split_counts["test"],
            "train_mri": split_modality_counts[("train", "mri")],
            "train_pet": split_modality_counts[("train", "pet")],
            "validation_mri": split_modality_counts[("validation", "mri")],
            "validation_pet": split_modality_counts[("validation", "pet")],
            "test_mri": split_modality_counts[("test", "mri")],
            "test_pet": split_modality_counts[("test", "pet")],
            "referenced_shards": int(cache_layout["referenced_shards"]),
        }
        report["receipt_bundle_sha256"] = str(
            cache_layout["receipt_bundle_sha256"]
        )
        report["all_contracts_match"] = True
    except Exception as error:  # report a failed readiness contract without mutation
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def inspect_workstation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only readiness and immutable-contract inspection."""

    fomo_root = _path(config, "fomo_root")
    adni_root = _path(config, "adni_pet_root")
    run_root = _path(config, "run_root")
    raw = {
        "fomo_root": {
            "path": str(fomo_root),
            "placeholder": "CHANGE_ME" in str(fomo_root),
            "exists": fomo_root.is_dir(),
        },
        "adni_pet_root": {
            "path": str(adni_root),
            "placeholder": "CHANGE_ME" in str(adni_root),
            "exists": adni_root.is_dir(),
        },
    }
    inventory_summary: Mapping[str, Any] | None = None
    inventory_error = ""
    if all(bool(item["exists"]) for item in raw.values()):
        try:
            inventory_summary = PhysicalInventory.from_filesystem(
                fomo_root=fomo_root, adni_root=adni_root
            ).summary()
        except Exception as error:
            inventory_error = str(error)
    assets = _asset_report(config)
    training_contracts = _inspect_training_cache_contracts(
        config, assets_match=bool(assets["all_match"])
    )
    checkpoint_pointer = run_root / "checkpoints" / "latest.json"
    ready_for_preprocessing = bool(
        assets["all_match"]
        and all(bool(item["exists"]) for item in raw.values())
        and not inventory_error
        and shutil.which(str(config["preprocessing"].get("dcm2niix", "dcm2niix")))
        and importlib.util.find_spec("SimpleITK") is not None
    )
    ready_for_training = bool(training_contracts["all_contracts_match"])
    return {
        "schema_version": 1,
        "inspection_is_read_only": True,
        "config_path": str(config["_resolved"]["config_path"]),
        "config_digest": str(config["_resolved"]["digest"]),
        "raw_data": raw,
        "inventory": inventory_summary,
        "inventory_error": inventory_error,
        "assets": assets,
        "tools": {
            "dcm2niix": shutil.which(
                str(config["preprocessing"].get("dcm2niix", "dcm2niix"))
            ),
            "SimpleITK_importable": importlib.util.find_spec("SimpleITK") is not None,
            "rich_importable": importlib.util.find_spec("rich") is not None,
        },
        "cache": training_contracts,
        "checkpoint": {
            "latest_pointer": str(checkpoint_pointer),
            "exists": checkpoint_pointer.is_file(),
        },
        "ready_for_preprocessing": ready_for_preprocessing,
        "ready_for_training": ready_for_training,
    }


def _preprocessing_contract(config: Mapping[str, Any]) -> str:
    paths = config["paths"]
    return digest_object(
        {
            "schema": 1,
            "paths": {
                key: paths[key]
                for key in (
                    "metadata_root",
                    "fomo_root",
                    "adni_pet_root",
                    "source_reference",
                    "source_reference_mask",
                )
            },
            "data": config["data"],
            "preprocessing": config["preprocessing"],
        }
    )


def _formal_preprocessing_config(
    config: Mapping[str, Any],
    *,
    cache_root: Path,
    retry_failures: bool,
) -> PreprocessingConfig:
    preprocessing = config["preprocessing"]
    registration = preprocessing["registration"]
    cache_config = preprocessing["cache"]
    executable = str(preprocessing.get("dcm2niix", "dcm2niix"))
    return PreprocessingConfig(
        output_root=cache_root,
        source_reference=_path(config, "source_reference"),
        source_mask=_path(config, "source_reference_mask"),
        target_shape=tuple(int(value) for value in preprocessing["target_shape"]),
        target_spacing=tuple(
            float(value) for value in preprocessing["target_spacing_mm"]
        ),
        shard_size=int(cache_config["shard_size"]),
        workers=int(preprocessing["workers"]),
        registration_mode=str(registration.get("mode", "simpleitk_rigid_affine")),
        mri_final_interpolation=str(registration["interpolation_mri"]),
        pet_final_interpolation=str(registration["interpolation_pet"]),
        registration_sampling_fraction=float(
            registration.get("sampling_fraction", 0.20)
        ),
        rigid_iterations=int(registration.get("rigid_iterations", 100)),
        affine_iterations=int(registration.get("affine_iterations", 100)),
        sitk_threads_per_worker=int(registration.get("threads_per_worker", 1)),
        dcm2niix_command=(
            executable,
            "-z",
            "y",
            "-b",
            "n",
            "-f",
            "converted",
            "-o",
            "{output_dir}",
            "{input_dir}",
        ),
        retry_failures=bool(retry_failures),
    )


def _expected_reference_contract(config: Mapping[str, Any]) -> str:
    return reference_contract_digest(
        ReferenceSpec(
            source_reference=_path(config, "source_reference"),
            source_mask=_path(config, "source_reference_mask"),
            target_shape=tuple(
                int(value) for value in config["preprocessing"]["target_shape"]
            ),
            target_spacing=tuple(
                float(value)
                for value in config["preprocessing"]["target_spacing_mm"]
            ),
        )
    )


def _geometry_contract_digest(
    config: Mapping[str, Any], reference_contract: str
) -> str:
    geometry_config = config["model"]["geometry"]
    return digest_object(
        {
            "schema": 1,
            "reference_contract": str(reference_contract),
            "nodes_target": int(geometry_config["nodes_target"]),
            "modes": int(geometry_config["modes"]),
            "occupancy_threshold": 0.20,
        }
    )


def _relation_digest(relations: Sequence[ObservationRelation]) -> str:
    return digest_object([row.to_dict() for row in relations])


def _recover_partial_generated_files(
    paths: Sequence[Path],
    *,
    resume: bool,
    description: str,
) -> tuple[str, ...]:
    """Remove only a known incomplete generated-file set during resume."""

    existing = tuple(path for path in paths if path.exists())
    if not existing:
        return ()
    if not resume:
        raise FormalDataError(
            f"Partial {description} exists; --no-resume never overwrites it"
        )
    non_files = [str(path) for path in existing if not path.is_file()]
    if non_files:
        raise FormalDataError(
            f"Partial {description} contains non-files: " + ", ".join(non_files)
        )
    removed = tuple(str(path) for path in existing)
    for path in existing:
        path.unlink()
    return removed


def _load_or_build_catalog(
    config: Mapping[str, Any],
    *,
    cache_root: Path,
    resume: bool,
    journal: RunJournal,
    progress: TerminalProgress,
) -> tuple[CatalogBuildResult, tuple[ObservationRelation, ...], Mapping[str, Any]]:
    catalog_dir = cache_root / "catalog"
    receipt_path = catalog_dir / "catalog_contract.json"
    observations_path = catalog_dir / "observations.jsonl"
    splits_path = catalog_dir / "subject_splits.jsonl"
    relations_path = catalog_dir / "relations.jsonl"
    contract = _preprocessing_contract(config)
    existing = all(
        path.is_file()
        for path in (receipt_path, observations_path, splits_path, relations_path)
    )
    if existing:
        if not resume:
            raise FormalDataError(
                f"Catalog already exists under {catalog_dir}; --no-resume never overwrites it"
            )
        receipt = read_json(receipt_path)
        if not isinstance(receipt, Mapping) or receipt.get("contract_digest") != contract:
            raise FormalDataError(
                "Existing catalog preprocessing contract differs; choose a new cache_root"
            )
        progress.begin_stage("catalog", total=1, description="validate frozen catalog")
        catalog = load_catalog(observations_path, splits_path)
        relations = load_relations(relations_path)
        validate_relations(relations, catalog.observations)
        if catalog.digest != receipt.get("catalog_digest"):
            raise FormalDataError("Frozen catalog digest differs from its receipt")
        if _relation_digest(relations) != receipt.get("relations_digest"):
            raise FormalDataError("Frozen relation digest differs from its receipt")
        progress.finish_stage(status="resumed")
        journal.event(
            "catalog_resumed",
            "Validated frozen catalog and relation contracts",
            observation_count=len(catalog.observations),
            subject_count=len(catalog.subject_splits),
            relation_count=len(relations),
            catalog_digest=catalog.digest,
        )
        return catalog, relations, dict(receipt)

    removed_partial = _recover_partial_generated_files(
        (observations_path, splits_path, relations_path, receipt_path),
        resume=resume,
        description=f"catalog under {catalog_dir}",
    )
    if removed_partial:
        journal.warning(
            "partial_catalog_recovered",
            "Removed an incomplete generated catalog set before deterministic rebuild",
            removed_files=list(removed_partial),
        )
    progress.begin_stage("inventory", total=1, description="scan physical raw data")
    inventory = PhysicalInventory.from_filesystem(
        fomo_root=_path(config, "fomo_root"),
        adni_root=_path(config, "adni_pet_root"),
    )
    progress.finish_stage()
    journal.event("inventory_complete", "Physical inventory completed", **inventory.summary())

    dataset_events = 0

    def catalog_progress(event: str, payload: Mapping[str, Any]) -> None:
        nonlocal dataset_events
        if event == "catalog_start":
            progress.begin_stage("catalog", total=9, description="metadata physical inner join")
        elif event == "catalog_dataset":
            dataset_events += 1
            progress.update(
                completed=min(dataset_events, 9),
                status=str(payload.get("dataset_id", "dataset")),
            )
        elif event == "catalog_complete":
            progress.finish_stage()
        journal.event(event, "", **dict(payload))

    catalog = build_catalog(
        _path(config, "metadata_root"),
        inventory,
        split_seed=int(config["data"]["split"]["seed"]),
        progress=catalog_progress,
    )
    catalog.write(catalog_dir)
    progress.begin_stage("relations", total=1, description="build legal same-subject relations")
    relation_result: RelationBuildResult = build_relations(catalog.observations)
    relation_result.write(catalog_dir)
    progress.finish_stage()
    receipt = {
        "schema_version": 1,
        "created_at": utc_now(),
        "contract_digest": contract,
        "catalog_digest": catalog.digest,
        "relations_digest": _relation_digest(relation_result.relations),
        "inventory": inventory.summary(),
        "catalog_summary": dict(catalog.summary),
        "relations_summary": dict(relation_result.summary),
    }
    atomic_write_json(receipt_path, receipt)
    journal.event(
        "catalog_frozen",
        "Wrote immutable catalog, subject split and legal relations",
        catalog_digest=catalog.digest,
        relations_digest=receipt["relations_digest"],
        observation_count=len(catalog.observations),
        subject_count=len(catalog.subject_splits),
        relation_count=len(relation_result.relations),
    )
    return catalog, relation_result.relations, receipt


def _ensure_geometry(
    config: Mapping[str, Any],
    *,
    cache_root: Path,
    reference_mask: Path,
    reference_contract: str,
    resume: bool,
    journal: RunJournal,
    progress: TerminalProgress,
) -> tuple[FEMGeometry, Path, Mapping[str, Any]]:
    geometry_config = config["model"]["geometry"]
    geometry_dir = cache_root / "geometry"
    geometry_path = geometry_dir / "fem_geometry.pt"
    receipt_path = geometry_dir / "geometry_receipt.json"
    contract = _geometry_contract_digest(config, reference_contract)
    progress.begin_stage("geometry", total=1, description="FEM and spectral basis")
    if geometry_path.is_file() and receipt_path.is_file():
        receipt = read_json(receipt_path)
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("contract_digest") != contract
            or receipt.get("geometry_sha256") != sha256_file(geometry_path)
        ):
            raise FormalDataError(
                "Existing FEM geometry contract differs; choose a new cache_root"
            )
        geometry = FEMGeometry.load(geometry_path)
        if geometry.metadata.get("geometry_hash") != receipt.get("geometry_hash"):
            raise FormalDataError("FEM geometry hash differs from its receipt")
        status = "resumed"
    else:
        removed_partial = _recover_partial_generated_files(
            (geometry_path, receipt_path),
            resume=resume,
            description=f"FEM geometry under {geometry_dir}",
        )
        if removed_partial:
            journal.warning(
                "partial_geometry_recovered",
                "Removed incomplete generated FEM files before deterministic rebuild",
                removed_files=list(removed_partial),
            )
        geometry = build_fem_geometry(
            reference_mask,
            nodes=int(geometry_config["nodes_target"]),
            modes=int(geometry_config["modes"]),
            occupancy_threshold=0.20,
            output_path=geometry_path,
        )
        receipt = {
            "schema_version": 1,
            "created_at": utc_now(),
            "contract_digest": contract,
            "geometry_path": str(geometry_path),
            "geometry_sha256": sha256_file(geometry_path),
            "geometry_hash": geometry.metadata.get("geometry_hash"),
            "actual_nodes": geometry.node_count,
            "modes": geometry.mode_count,
            "reference_mask": str(reference_mask),
        }
        atomic_write_json(receipt_path, receipt)
        status = "created"
    progress.finish_stage(status=status)
    journal.event(
        "geometry_ready",
        "Validated fixed FEM geometry",
        status=status,
        geometry_hash=geometry.metadata.get("geometry_hash"),
        nodes=geometry.node_count,
        modes=geometry.mode_count,
    )
    return geometry, geometry_path, dict(receipt)


def run_preprocessing_workflow(
    config: Mapping[str, Any],
    *,
    resume: bool = True,
    force_plain_progress: bool = False,
    limit: int | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Build/resume catalog, model-ready shards and fixed FEM geometry."""

    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer")
    assets = _asset_report(config)
    if not assets["all_match"]:
        raise FormalDataError("Bundled metadata/SAT3D/reference asset hash check failed")
    for key in ("fomo_root", "adni_pet_root"):
        if not _path(config, key).is_dir():
            raise FormalDataError(f"Configured raw-data root is missing: {_path(config, key)}")
    executable = str(config["preprocessing"].get("dcm2niix", "dcm2niix"))
    if shutil.which(executable) is None:
        raise FormalDataError(f"Required DICOM converter is unavailable: {executable}")
    if importlib.util.find_spec("SimpleITK") is None:
        raise FormalDataError(
            "SimpleITK is required by the configured rigid-affine preprocessing"
        )
    configured_cache_root = _path(config, "cache_root")
    cache_root = (
        configured_cache_root
        if limit is None
        else configured_cache_root / f"development-limit-{int(limit)}"
    )
    if not resume and cache_root.exists() and any(cache_root.iterdir()):
        raise FormalDataError(
            f"--no-resume will not overwrite non-empty cache_root: {cache_root}"
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    logs = cache_root / "logs"
    run_id = f"formal-preprocess-{_preprocessing_contract(config)[:16]}"
    with RunJournal(
        logs / "preprocess.log",
        logs / "preprocess.metrics.jsonl",
        run_id=run_id,
        durable=False,
    ) as journal, TerminalProgress(
        journal=journal,
        force_plain=force_plain_progress,
        refresh_per_second=float(config["logging"]["refresh_per_second"]),
    ) as progress:
        journal.event(
            "preprocessing_workflow_started",
            "Formal offline preprocessing started",
            resume=bool(resume),
            development_limit=limit,
            retry_failed_requested=bool(retry_failed),
            cache_root=str(cache_root),
        )
        catalog, relations, catalog_receipt = _load_or_build_catalog(
            config,
            cache_root=cache_root,
            resume=resume,
            journal=journal,
            progress=progress,
        )
        selected = catalog.observations if limit is None else catalog.observations[: int(limit)]
        preprocess_config = _formal_preprocessing_config(
            config,
            cache_root=cache_root,
            retry_failures=bool(
                retry_failed
                or config["preprocessing"]["cache"].get(
                    "retry_failed_on_resume", True
                )
            ),
        )

        def on_case(event: PreprocessProgress) -> None:
            processed = event.completed + event.skipped + event.failed
            metrics = {
                "completed": event.completed,
                "resumed": event.skipped,
                "failed": event.failed,
                "elapsed_seconds": event.elapsed_seconds,
            }
            if event.event == "start":
                progress.begin_stage(
                    "preprocessing",
                    total=max(event.total, 1),
                    description="register and cache observations",
                    unit="cases",
                )
            else:
                progress.update(
                    completed=min(processed, max(event.total, 1)),
                    status=f"ok={event.completed} resume={event.skipped} failed={event.failed}",
                    metrics=metrics,
                )
            journal.metric(
                f"preprocess_{event.event}",
                metrics,
                observation_uid=event.observation_uid,
                message=event.message,
                processed=processed,
                total=event.total,
            )
            if event.event == "complete":
                progress.finish_stage(
                    status=f"attempted all; failed={event.failed}"
                )

        summary = OfflinePreprocessor(
            selected,
            preprocess_config,
            progress=on_case,
        ).run()
        _, geometry_path, geometry_receipt = _ensure_geometry(
            config,
            cache_root=cache_root,
            reference_mask=summary.reference.mask_path,
            reference_contract=summary.reference.contract_digest,
            resume=resume,
            journal=journal,
            progress=progress,
        )
        result = {
            "schema_version": 1,
            "scientific_run": limit is None,
            "cache_root": str(cache_root),
            "catalog_digest": catalog.digest,
            "relations_digest": str(catalog_receipt["relations_digest"]),
            "planned_cases": summary.total,
            "completed_this_invocation": summary.completed,
            "resumed_cases": summary.skipped,
            "failed_cases": summary.failed,
            "pending_cases": summary.pending,
            "frozen_cache_cases": len(summary.cache_entries),
            "cache_contract_digest": summary.contract_digest,
            "cache_plan_digest": summary.plan_digest,
            "reference_contract_digest": summary.reference.contract_digest,
            "geometry_path": str(geometry_path),
            "geometry_hash": geometry_receipt["geometry_hash"],
            "logs": {
                "text": str(logs / "preprocess.log"),
                "jsonl": str(logs / "preprocess.metrics.jsonl"),
                "failures": str(cache_root / "failures.jsonl"),
            },
        }
        atomic_write_json(cache_root / "formal_preprocessing_result.json", result)
        journal.event(
            "preprocessing_workflow_completed",
            "All planned cases were attempted and artifacts were frozen",
            **result,
        )
        journal.sync()
        return result


def run_training_workflow(
    config: Mapping[str, Any],
    *,
    resume: bool = True,
    force_plain_progress: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Dispatch the bounded synthetic smoke or the real formal training path."""

    from .integration import run_synthetic_smoke

    if smoke:
        smoke_root = _path(config, "run_root") / "synthetic-smoke"
        return dict(run_synthetic_smoke(work_dir=smoke_root, device="cpu"))
    return _run_real_training_workflow(
        config,
        resume=resume,
        force_plain_progress=force_plain_progress,
    )


def _formal_python_source_digest(project_root: Path) -> str:
    source_root = project_root / "src" / "mri_pet_geomc"
    files = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    if not files:
        raise FormalDataError(f"No Python source found under {source_root}")
    return digest_object(
        [
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    )


def _resolve_resume_python_source_contract(
    *,
    project_root: Path,
    run_contract_path: Path,
    current_source_digest: str,
    resume: bool,
    asset_report: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any] | None]:
    """Retain one explicitly mapped pre-fix source hash for strict resume.

    The checkpoint remains byte-for-byte immutable.  Only the known
    companion-visible-support fallback and configured-horizon scheduler
    correction may resume a run whose frozen source hash is the exact
    vulnerable package digest.  The active fixed source is itself pinned by
    the bundled compatibility manifest and recorded beside the original run
    contract.
    """

    if not run_contract_path.is_file():
        return current_source_digest, None
    existing = read_json(run_contract_path)
    if not isinstance(existing, Mapping):
        raise ModelContractError("Frozen run_contract.json must be an object")
    existing_hashes = existing.get("hashes") or {}
    if not isinstance(existing_hashes, Mapping):
        raise ModelContractError("Frozen run contract hashes must be an object")
    frozen_source_digest = str(existing_hashes.get("python_source", ""))
    if not frozen_source_digest:
        raise ModelContractError("Frozen run contract lacks python_source hash")
    if frozen_source_digest == current_source_digest:
        return current_source_digest, None
    if not resume:
        raise ModelContractError(
            "Python source differs from the frozen run and --resume was not enabled"
        )

    checks = asset_report.get("checks") or {}
    compatibility_check = (
        checks.get("resume_compatibility") if isinstance(checks, Mapping) else None
    )
    if not isinstance(compatibility_check, Mapping) or not bool(
        compatibility_check.get("matches")
    ):
        raise ModelContractError(
            "Python source changed and the bundled resume-compatibility asset is invalid"
        )
    manifest_path = Path(str(compatibility_check.get("path", "")))
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ModelContractError("Resume-compatibility manifest must be an object")
    required = {
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
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ModelContractError(
                f"Resume-compatibility manifest mismatch for {key}"
            )
    if frozen_source_digest != _COMPANION_FALLBACK_FROM_SOURCE_SHA256:
        raise ModelContractError(
            "Frozen source hash is not the explicitly supported pre-fix package"
        )
    resolved_manifest = manifest_path.resolve()
    if resolved_manifest.parent != project_root.resolve():
        raise ModelContractError(
            "Resume-compatibility manifest must remain in the project root"
        )
    migration_record = {
        "schema_version": 1,
        "migration_id": _COMPANION_FALLBACK_MIGRATION_ID,
        "frozen_checkpoint_python_source_sha256": frozen_source_digest,
        "active_python_source_sha256": current_source_digest,
        "compatibility_manifest": manifest_path.relative_to(project_root).as_posix(),
        "compatibility_manifest_sha256": str(
            compatibility_check.get("actual_sha256", "")
        ),
        "checkpoint_contract_strategy": "retain_original_python_source_hash",
        "runtime_behavior": required["runtime_behavior"],
        "normal_companion_behavior_unchanged": True,
        "anchor_objective_unchanged": True,
        "scheduler_contract_correction": required[
            "scheduler_contract_correction"
        ],
    }
    return frozen_source_digest, migration_record


def _freeze_exact_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create an immutable run contract or verify its exact prior value."""

    payload = dict(value)
    if path.is_file():
        existing = read_json(path)
        if existing != payload:
            raise ModelContractError(
                f"Frozen run contract differs from the current invocation: {path}"
            )
        return
    if path.exists():
        raise ModelContractError(f"Frozen run contract path is not a file: {path}")
    atomic_write_json(path, payload)


def _validate_cache_layout(
    *,
    cache_root: Path,
    cache_entries: Sequence[Any],
    target_shape: Sequence[int],
    expected_contract_digest: str,
    expected_plan_digest: str,
) -> Mapping[str, Any]:
    """Validate mmap headers and every conditioning receipt without image reads."""

    shard_plan_path = cache_root / "shard_plan.json"
    cache_summary_path = cache_root / "cache_summary.json"
    if not shard_plan_path.is_file() or not cache_summary_path.is_file():
        raise FormalDataError("Frozen cache is missing shard_plan.json or cache_summary.json")
    shard_plan = read_json(shard_plan_path)
    cache_summary = read_json(cache_summary_path)
    if not isinstance(shard_plan, Mapping) or not isinstance(cache_summary, Mapping):
        raise FormalDataError("Frozen cache plan/summary must be JSON objects")
    if int(cache_summary.get("complete_count", -1)) != len(cache_entries):
        raise FormalDataError("cache_summary complete_count differs from cache_index")
    if (
        str(shard_plan.get("contract_digest", "")) != expected_contract_digest
        or str(cache_summary.get("contract_digest", "")) != expected_contract_digest
    ):
        raise FormalDataError("Frozen shard/cache contract digest differs from preprocessing")
    if (
        str(shard_plan.get("plan_digest", "")) != expected_plan_digest
        or str(cache_summary.get("plan_digest", "")) != expected_plan_digest
    ):
        raise FormalDataError("Frozen shard/cache plan digest differs from preprocessing")
    shape = tuple(int(value) for value in target_shape)
    if tuple(int(value) for value in shard_plan.get("shape", ())) != shape:
        raise FormalDataError("Frozen shard shape differs from the formal target shape")
    shard_counts = {
        int(key): int(value)
        for key, value in dict(shard_plan.get("shard_counts") or {}).items()
    }
    planned_count = int(shard_plan.get("observation_count", -1))
    if planned_count < 1 or sum(shard_counts.values()) != planned_count:
        raise FormalDataError("shard_plan observation_count differs from shard_counts")
    if int(cache_summary.get("planned_count", -1)) != planned_count:
        raise FormalDataError("cache_summary planned_count differs from shard_plan")
    if (
        int(cache_summary.get("complete_count", -1))
        + int(cache_summary.get("failed_or_pending_count", -1))
        != planned_count
    ):
        raise FormalDataError("cache_summary completion accounting is inconsistent")
    locations = [(int(entry.shard_id), int(entry.slot)) for entry in cache_entries]
    if len(set(locations)) != len(locations):
        raise FormalDataError("cache_index contains duplicate shard/slot locations")
    packed_voxels = (int(np.prod(shape)) + 7) // 8
    for entry in cache_entries:
        expected_image = f"shards/shard-{int(entry.shard_id):05d}.images.npy"
        expected_support = f"shards/shard-{int(entry.shard_id):05d}.support.npy"
        if entry.image_path != expected_image or entry.support_path != expected_support:
            raise FormalDataError(
                f"Unsafe/noncanonical cache paths for {entry.observation_uid}"
            )
        if tuple(int(value) for value in entry.shape) != shape or entry.dtype != "float16":
            raise FormalDataError(
                f"Cache entry shape/dtype mismatch for {entry.observation_uid}"
            )
        planned_count = shard_counts.get(int(entry.shard_id), -1)
        if not 0 <= int(entry.slot) < planned_count:
            raise FormalDataError(f"Cache slot is outside its shard for {entry.observation_uid}")
    referenced_shards = sorted({int(entry.shard_id) for entry in cache_entries})
    for shard_id in referenced_shards:
        if shard_id not in shard_counts:
            raise FormalDataError(f"cache_index references unplanned shard {shard_id}")
        image_path = cache_root / f"shards/shard-{shard_id:05d}.images.npy"
        support_path = cache_root / f"shards/shard-{shard_id:05d}.support.npy"
        if not image_path.is_file() or not support_path.is_file():
            raise FormalDataError(f"Frozen mmap shard {shard_id} is missing")
        images = np.load(image_path, mmap_mode="r", allow_pickle=False)
        supports = np.load(support_path, mmap_mode="r", allow_pickle=False)
        expected_count = shard_counts[shard_id]
        if images.shape != (expected_count, 1, *shape) or images.dtype != np.float16:
            raise FormalDataError(f"Image mmap header mismatch for shard {shard_id}")
        if supports.shape != (expected_count, packed_voxels) or supports.dtype != np.uint8:
            raise FormalDataError(f"Support mmap header mismatch for shard {shard_id}")
        del images, supports

    receipt_dir = cache_root / "receipts"
    receipt_lines: list[str] = []
    for entry in sorted(cache_entries, key=lambda item: item.observation_uid):
        receipt_path = receipt_dir / f"{entry.observation_uid}.json"
        receipt = read_json(receipt_path)
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("status") != "complete"
            or receipt.get("observation_uid") != entry.observation_uid
            or receipt.get("contract_digest") != expected_contract_digest
            or receipt.get("plan_digest") != expected_plan_digest
            or int(receipt.get("shard_id", -1)) != int(entry.shard_id)
            or int(receipt.get("slot", -1)) != int(entry.slot)
        ):
            raise FormalDataError(
                f"Cache receipt contract mismatch for {entry.observation_uid}"
            )
        # Parse the structural overlay now so malformed metadata fails before
        # the expensive SAT3D model is allocated on the GPU.
        load_preprocessing_overlay(receipt_dir, entry.observation_uid)
        receipt_lines.append(
            f"{entry.observation_uid}\t{sha256_file(receipt_path)}"
        )
    receipt_digest = hashlib.sha256(
        "\n".join(receipt_lines).encode("utf-8")
    ).hexdigest()
    return {
        "shard_plan": dict(shard_plan),
        "cache_summary": dict(cache_summary),
        "receipt_bundle_sha256": receipt_digest,
        "referenced_shards": len(referenced_shards),
    }


def _metadata_records_for_split(
    observations: Sequence[Any],
    *,
    cached_uids: set[str],
    split: str,
    receipt_dir: Path,
) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for observation in observations:
        if observation.split != split or observation.observation_uid not in cached_uids:
            continue
        metadata = observation.to_dict()
        metadata["preprocessing"] = dict(
            load_preprocessing_overlay(receipt_dir, observation.observation_uid)
        )
        records.append(metadata)
    if not records:
        raise FormalDataError(f"No successfully cached observations for split {split!r}")
    return records


def _run_real_training_workflow(
    config: Mapping[str, Any],
    *,
    resume: bool,
    force_plain_progress: bool,
) -> dict[str, Any]:
    """Validate every frozen contract, then run one fixed formal experiment."""

    import torch

    from .integration import (
        FormalCoverageDataSource,
        FormalModelBuildSpec,
        TorchFormalTrainingBackend,
        build_acquisition_vocabulary,
        build_real_training_components,
        run_final_test_once,
        run_full_training,
    )

    run_root = _path(config, "run_root")
    formal_targets = (
        run_root / "logs",
        run_root / "checkpoints",
        run_root / "contracts",
        run_root / "formal_training_result.json",
    )
    if not resume and any(path.exists() for path in formal_targets):
        raise ModelContractError(
            f"--no-resume will not overwrite formal run state under {run_root}"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    logs = run_root / "logs"
    contracts_dir = run_root / "contracts"
    config_digest = str(config["_resolved"]["digest"])
    run_id = f"formal-24k-{config_digest[:16]}"

    with RunJournal(
        logs / "train.log",
        logs / "train.metrics.jsonl",
        run_id=run_id,
        durable=False,
    ) as journal, TerminalProgress(
        journal=journal,
        force_plain=force_plain_progress,
        refresh_per_second=float(config["logging"]["refresh_per_second"]),
    ) as progress:
        try:
            journal.event(
                "formal_training_workflow_started",
                "Validating frozen data/model contracts before GPU allocation",
                resume=bool(resume),
                run_root=str(run_root),
                fixed_horizon_coverages=int(config["training"]["epochs"]),
                performance_gate=None,
                early_stopping=False,
            )
            progress.begin_stage(
                "contracts",
                total=6,
                description="validate cache, receipts and immutable assets",
                unit="checks",
            )
            assets = _asset_report(config)
            if not assets["all_match"]:
                raise FormalDataError(
                    "Bundled metadata/SAT3D/reference asset hash check failed"
                )
            project_root = _project_root(config)
            current_python_source_digest = _formal_python_source_digest(project_root)
            checkpoint_python_source_digest, source_migration = (
                _resolve_resume_python_source_contract(
                    project_root=project_root,
                    run_contract_path=contracts_dir / "run_contract.json",
                    current_source_digest=current_python_source_digest,
                    resume=resume,
                    asset_report=assets,
                )
            )
            progress.update(completed=1, status="bundled assets")

            cache_root = _path(config, "cache_root")
            catalog_dir = cache_root / "catalog"
            required = {
                "catalog_contract": catalog_dir / "catalog_contract.json",
                "observations": catalog_dir / "observations.jsonl",
                "subject_splits": catalog_dir / "subject_splits.jsonl",
                "relations": catalog_dir / "relations.jsonl",
                "cache_index": cache_root / "cache_index.jsonl",
                "preprocessing_summary": cache_root / "preprocessing_summary.json",
                "reference": cache_root / "reference" / "target_reference.nii.gz",
                "reference_mask": cache_root / "reference" / "target_mask.nii.gz",
                "reference_receipt": cache_root / "reference" / "reference_receipt.json",
                "geometry": cache_root / "geometry" / "fem_geometry.pt",
                "geometry_receipt": cache_root / "geometry" / "geometry_receipt.json",
            }
            missing = [str(path) for path in required.values() if not path.is_file()]
            if missing:
                raise FormalDataError(
                    "Formal preprocessing is incomplete; missing: " + ", ".join(missing)
                )

            catalog = load_catalog(
                required["observations"], required["subject_splits"]
            )
            relations = load_relations(required["relations"])
            validate_relations(relations, catalog.observations)
            catalog_receipt = read_json(required["catalog_contract"])
            if not isinstance(catalog_receipt, Mapping):
                raise FormalDataError("catalog_contract.json must be an object")
            relation_digest = _relation_digest(relations)
            if catalog_receipt.get("catalog_digest") != catalog.digest:
                raise FormalDataError("Frozen catalog differs from catalog_contract.json")
            if catalog_receipt.get("relations_digest") != relation_digest:
                raise FormalDataError("Frozen relations differ from catalog_contract.json")
            progress.update(completed=2, status="catalog and subject split")

            cache_entries = load_cache_index(required["cache_index"])
            if not cache_entries:
                raise FormalDataError("cache_index is empty; no observation can be trained")
            cached_uids = {entry.observation_uid for entry in cache_entries}
            catalog_uids = {row.observation_uid for row in catalog.observations}
            if len(cached_uids) != len(cache_entries) or not cached_uids <= catalog_uids:
                raise FormalDataError("cache_index UIDs are duplicated or outside the catalog")
            preprocessing_summary = read_json(required["preprocessing_summary"])
            if not isinstance(preprocessing_summary, Mapping):
                raise FormalDataError("preprocessing_summary.json must be an object")
            cache_contract = str(preprocessing_summary.get("contract_digest", ""))
            cache_plan = str(preprocessing_summary.get("plan_digest", ""))
            if not cache_contract or not cache_plan:
                raise FormalDataError("Preprocessing summary lacks cache contracts")
            expected_reference_contract = _expected_reference_contract(config)
            if (
                preprocessing_summary.get("reference_contract_digest")
                != expected_reference_contract
            ):
                raise FormalDataError(
                    "Preprocessing summary is not bound to the configured reference"
                )
            expected_cache_contract = preprocessing_contract_digest(
                _formal_preprocessing_config(
                    config, cache_root=cache_root, retry_failures=True
                ),
                expected_reference_contract,
            )
            if cache_contract != expected_cache_contract:
                raise FormalDataError(
                    "Preprocessing cache contract differs from reference/config"
                )
            if any(entry.contract_digest != cache_contract for entry in cache_entries):
                raise FormalDataError("cache_index contains a different preprocessing contract")
            cache_layout = _validate_cache_layout(
                cache_root=cache_root,
                cache_entries=cache_entries,
                target_shape=config["preprocessing"]["target_shape"],
                expected_contract_digest=cache_contract,
                expected_plan_digest=cache_plan,
            )
            if cache_layout["shard_plan"].get("plan_digest") != cache_plan:
                raise FormalDataError("shard_plan differs from preprocessing_summary")
            progress.update(completed=3, status="mmap headers and all receipts")

            reference_receipt = read_json(required["reference_receipt"])
            if not isinstance(reference_receipt, Mapping):
                raise FormalDataError("reference_receipt.json must be an object")
            if reference_receipt.get("contract_digest") != expected_reference_contract:
                raise FormalDataError(
                    "Reference receipt contract differs from bundled inputs"
                )
            reference_outputs = reference_receipt.get("outputs") or {}
            if not isinstance(reference_outputs, Mapping):
                raise FormalDataError("Reference receipt outputs must be an object")
            if (
                reference_outputs.get("reference_sha256")
                != sha256_file(required["reference"])
                or reference_outputs.get("mask_sha256")
                != sha256_file(required["reference_mask"])
            ):
                raise FormalDataError("Target reference/mask differs from its receipt")
            reference_image = nib.load(required["reference"])
            if tuple(int(value) for value in reference_image.shape) != tuple(
                int(value) for value in config["preprocessing"]["target_shape"]
            ):
                raise FormalDataError("Target reference shape differs from formal config")
            if not np.allclose(
                nib.affines.voxel_sizes(reference_image.affine),
                np.asarray(config["preprocessing"]["target_spacing_mm"], dtype=float),
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise FormalDataError("Target reference spacing differs from formal config")
            progress.update(completed=4, status="reference lattice")

            geometry_receipt = read_json(required["geometry_receipt"])
            if not isinstance(geometry_receipt, Mapping):
                raise FormalDataError("geometry_receipt.json must be an object")
            if geometry_receipt.get("contract_digest") != _geometry_contract_digest(
                config, expected_reference_contract
            ):
                raise FormalDataError(
                    "FEM geometry contract differs from reference/config"
                )
            if geometry_receipt.get("geometry_sha256") != sha256_file(
                required["geometry"]
            ):
                raise FormalDataError("FEM geometry differs from its receipt")
            if int(geometry_receipt.get("modes", -1)) != int(
                config["model"]["geometry"]["modes"]
            ):
                raise FormalDataError("FEM mode count differs from formal config")
            progress.update(completed=5, status="FEM geometry")

            split_counts = Counter(
                row.split for row in catalog.observations if row.observation_uid in cached_uids
            )
            split_modality_counts = Counter(
                (row.split, str(row.modality).strip().lower())
                for row in catalog.observations
                if row.observation_uid in cached_uids
            )
            for split in ("train", "validation", "test"):
                if split_counts[split] < 1:
                    raise FormalDataError(
                        f"No successfully cached observation remains in {split}"
                    )
                for modality in ("mri", "pet"):
                    if split_modality_counts[(split, modality)] < 1:
                        raise FormalDataError(
                            f"No successfully cached {modality.upper()} remains in {split}"
                        )
            progress.finish_stage(
                status=(
                    f"cache={len(cache_entries)} "
                    f"train/val/test={split_counts['train']}/"
                    f"{split_counts['validation']}/{split_counts['test']}"
                )
            )

            seed = int(config["project"]["seed"])
            set_deterministic_seed(seed)
            device = torch.device(str(config["training"]["device"]))
            if device.type != "cuda" or device.index not in (None, 0):
                raise ModelContractError(
                    "The first formal run is intentionally fixed to one GPU at cuda:0"
                )
            if not torch.cuda.is_available():
                raise ModelContractError("CUDA is unavailable; formal training cannot start")
            torch.cuda.set_device(0)
            if str(config["training"]["precision"]) != "bf16":
                raise ModelContractError("The formal workstation precision is fixed to BF16")
            if not torch.cuda.is_bf16_supported():
                raise ModelContractError("cuda:0 does not report BF16 support")
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu = torch.cuda.get_device_properties(0)
            gpu_total_gib = float(gpu.total_memory) / float(1024**3)
            journal.event(
                "cuda_contract_validated",
                "Using one BF16 CUDA device; the second GPU remains unused",
                torch_version=torch.__version__,
                cuda_runtime=torch.version.cuda,
                device_name=gpu.name,
                total_memory_gib=gpu_total_gib,
                device_count=torch.cuda.device_count(),
                tf32_enabled=True,
            )
            if gpu_total_gib < float(config["resources"]["vram_soft_limit_gib"]):
                journal.warning(
                    "gpu_memory_below_soft_planning_value",
                    "GPU memory is below the configured planning value; this is warning-only",
                    total_memory_gib=gpu_total_gib,
                    planning_value_gib=float(
                        config["resources"]["vram_soft_limit_gib"]
                    ),
                )

            receipt_dir = cache_root / "receipts"
            train_records = _metadata_records_for_split(
                catalog.observations,
                cached_uids=cached_uids,
                split="train",
                receipt_dir=receipt_dir,
            )
            vocabulary = build_acquisition_vocabulary(train_records)
            vocabulary_payload = vocabulary.as_dict()
            vocabulary_digest = digest_object(vocabulary_payload)
            contracts_dir.mkdir(parents=True, exist_ok=True)
            _freeze_exact_json(
                contracts_dir / "config.resolved.json", dict(config)
            )
            _freeze_exact_json(
                contracts_dir / "metadata_vocabulary.json", vocabulary_payload
            )

            loader = config["loader"]
            dataset_kwargs = {
                "observations": catalog.observations,
                "relations": relations,
                "cache_root": cache_root,
                "cache_index": cache_entries,
                "relation_seed": seed,
                "max_open_shards": int(loader["max_open_shards_per_worker"]),
                "receipt_dir": receipt_dir,
            }
            train_dataset = FormalVolumeDataset(split="train", **dataset_kwargs)
            validation_dataset = FormalVolumeDataset(
                split="validation", **dataset_kwargs
            )
            test_dataset = FormalVolumeDataset(split="test", **dataset_kwargs)
            if len(train_dataset) != len(train_records):
                raise FormalDataError("Training vocabulary and dataset counts differ")

            source_kwargs = {
                "seed": seed,
                "num_workers": int(loader["num_workers"]),
                "prefetch_factor": int(loader["prefetch_factor"]),
                "persistent_workers": bool(loader["persistent_workers"]),
                "pin_memory": bool(loader["pin_memory"]),
            }
            train_source = FormalCoverageDataSource(train_dataset, **source_kwargs)
            validation_source = FormalCoverageDataSource(
                validation_dataset, **source_kwargs
            )
            test_source = FormalCoverageDataSource(test_dataset, **source_kwargs)

            progress.begin_stage(
                "model",
                total=2,
                description="load SAT3D, EMA, GeoMC and JEPA predictor",
                unit="contracts",
            )
            model_spec = FormalModelBuildSpec.from_config(
                config,
                sat3d_code_root=_path(config, "sat3d_code_root"),
                sat3d_checkpoint=_path(config, "sat3d_checkpoint"),
                fem_geometry_path=required["geometry"],
                reference_affine=np.asarray(reference_image.affine).tolist(),
            )
            progress.update(completed=1, status="YAML model contract")
            training = config["training"]
            steps_per_coverage = math.ceil(
                len(train_dataset) / int(training["global_batch_size"])
            )
            total_optimizer_steps = steps_per_coverage * int(training["epochs"])

            def evaluation_progress(
                event: str,
                completed: int,
                total: int,
                label: str,
            ) -> None:
                stage = "held-out-test" if "TEST" in label else "validation"
                if event == "start":
                    progress.begin_stage(
                        stage,
                        total=max(int(total), 1),
                        description=label,
                        unit="anchor-views",
                    )
                elif event == "update":
                    progress.update(completed=min(int(completed), int(total)))
                elif event == "complete":
                    progress.finish_stage(status="observational only")
                elif event == "error":
                    progress.update(
                        completed=min(int(completed), int(total)),
                        status="failed",
                    )

            def companion_fallback_callback(
                fallbacks: Sequence[Mapping[str, Any]],
            ) -> None:
                journal.warning(
                    "optional_companion_visible_support_empty",
                    "Dropped unusable optional companion and kept the anchor self-JEPA objective",
                    fallback_count=len(fallbacks),
                    fallbacks=[dict(value) for value in fallbacks],
                    anchor_optimizer_step_will_continue=True,
                    affects_anchor_target=False,
                )

            components = build_real_training_components(
                spec=model_spec,
                vocabulary=vocabulary,
                device=device,
                total_optimizer_steps=total_optimizer_steps,
                total_coverages=int(training["epochs"]),
                validation_source=validation_source,
                evaluation_progress=evaluation_progress,
                validation_modality_weights=dict(
                    config["validation"]["modality_macro_weights"]
                ),
                precision=str(training["precision"]),
                metadata_dropout_probability=float(
                    config["metadata_conditioning"]["dropout_probability"]
                ),
                mask_seed=seed,
                companion_fallback_callback=companion_fallback_callback,
            )
            progress.finish_stage(
                status=(
                    f"safe checkpoint stages="
                    f"{components.model_result.activation_checkpointing.safe_wrapper_count}"
                )
            )

            model_payload = {
                "schema_version": 1,
                "build": dict(components.model_result.specification),
                "activation_checkpointing": (
                    components.model_result.activation_checkpointing.as_dict()
                ),
                "valid_for_scientific_results": bool(
                    components.model_result.valid_for_scientific_results
                ),
            }
            model_digest = digest_object(model_payload)
            _freeze_exact_json(
                contracts_dir / "model_specification.json", model_payload
            )
            cache_index_digest = digest_object(
                [entry.to_dict() for entry in cache_entries]
            )
            hashes = {
                "config": config_digest,
                "python_source": checkpoint_python_source_digest,
                "catalog": catalog.digest,
                "relations": relation_digest,
                "catalog_contract": str(catalog_receipt["contract_digest"]),
                "cache_contract": cache_contract,
                "cache_plan": cache_plan,
                "cache_index": cache_index_digest,
                "preprocessing_receipts": str(
                    cache_layout["receipt_bundle_sha256"]
                ),
                "reference_contract": str(reference_receipt["contract_digest"]),
                "reference": str(reference_outputs["reference_sha256"]),
                "reference_mask": str(reference_outputs["mask_sha256"]),
                "fem_geometry": str(geometry_receipt["geometry_sha256"]),
                "fem_geometry_contract": str(geometry_receipt["contract_digest"]),
                "sat3d_checkpoint": str(
                    assets["checks"]["sat3d_checkpoint"]["actual_sha256"]
                ),
                "sat3d_source_tree": str(
                    assets["checks"]["sat3d_source_tree"]["actual_sha256"]
                ),
                "metadata_vocabulary": vocabulary_digest,
                "model_specification": model_digest,
            }
            run_contract = {
                "schema_version": 1,
                "run_id": run_id,
                "hashes": hashes,
                "observations": {
                    "catalog": len(catalog.observations),
                    "cached": len(cache_entries),
                    "train": len(train_dataset),
                    "validation": len(validation_dataset),
                    "test": len(test_dataset),
                },
                "subjects": dict(Counter(row.split for row in catalog.subject_splits)),
                "relations": len(relations),
                "steps_per_coverage": steps_per_coverage,
                "total_optimizer_steps": total_optimizer_steps,
                "fixed_coverages": int(training["epochs"]),
                "global_batch_size": int(training["global_batch_size"]),
                "companion_objective": "context_only_anchor_self_jepa",
                "performance_gate": None,
                "early_stopping": False,
            }
            _freeze_exact_json(contracts_dir / "run_contract.json", run_contract)
            if source_migration is not None:
                _freeze_exact_json(
                    contracts_dir / "resume_source_compatibility.json",
                    source_migration,
                )
                journal.warning(
                    "resume_source_compatibility_activated",
                    "Resuming the immutable pre-fix checkpoint with the pinned runtime corrections",
                    **dict(source_migration),
                )
            journal.event(
                "formal_run_contract_frozen",
                "Bound data, cache, source, SAT3D, geometry and model contracts",
                **run_contract,
            )
            journal.sync()

            mixture_config = training["relation_schedule"]["steady_mixture"]
            schedule = FormalTrainingSchedule(
                total_coverages=int(training["epochs"]),
                batch=BatchContract(
                    micro_batch_anchors=int(training["microbatch_size"]),
                    gradient_accumulation_steps=int(
                        training["gradient_accumulation"]
                    ),
                    world_size=1,
                ),
                fixed_mixture=RelationMixture(
                    self_only=float(mixture_config["self_only"]),
                    same_session_cross_sequence=float(
                        mixture_config["same_session_cross_sequence"]
                    ),
                    longitudinal_same_acquisition=float(
                        mixture_config["longitudinal_same_acquisition"]
                    ),
                    same_session_repeat=float(
                        mixture_config["same_session_repeat"]
                    ),
                ),
            )
            validation = config["validation"]
            cadence = CoverageValidationCadence(
                total_coverages=int(training["epochs"]),
                monitor_every_coverages=1,
                full_every_coverages=int(validation["full_every_coverages"]),
                monitor_sample_limit=int(validation["health_anchor_count"]),
                monitor_masks_per_anchor=int(validation["masks_per_anchor"]),
                full_masks_per_anchor=1,
            )
            telemetry = TelemetryRecorder(
                journal=journal,
                progress=progress,
                resource_provider=DefaultResourceProvider(gpu_device=0),
                gpu_reserved_soft_limit_gib=float(
                    config["resources"]["vram_soft_limit_gib"]
                ),
            )
            checkpoints = AtomicCheckpointManager(
                run_root / "checkpoints", journal=journal
            )
            controller = TrainingRuntimeController(
                run_id=run_id,
                hashes=hashes,
                journal=journal,
                progress=progress,
                telemetry=telemetry,
                checkpoints=checkpoints,
                schedule=schedule,
                validation_cadence=cadence,
                checkpoint_every_optimizer_steps=int(
                    training["checkpoint"]["every_updates"]
                ),
            )
            loop_result = run_full_training(
                controller=controller,
                data_source=train_source,
                backend=components.backend,
                resume=resume,
                keep_last=int(training["checkpoint"]["keep_last"]),
                keep_every_coverages=int(
                    training["checkpoint"]["keep_every_coverages"]
                ),
            )

            def evaluate_held_out_test(live_backend: Any) -> Mapping[str, Any]:
                if not isinstance(live_backend, TorchFormalTrainingBackend):
                    raise ModelContractError("Held-out TEST received a different backend")
                original_source = live_backend.validation_source
                live_backend.validation_source = test_source
                try:
                    return live_backend.validate(
                        ValidationRequest(
                            coverage=int(training["epochs"]),
                            optimizer_step=loop_result.cursor.optimizer_step,
                            tier=ValidationTier.FULL_VALIDATION,
                            reason="held-out TEST after fixed training horizon",
                            sample_limit=None,
                            masks_per_anchor=1,
                            observational_only=True,
                        )
                    )
                finally:
                    live_backend.validation_source = original_source

            test_metrics = run_final_test_once(
                controller=controller,
                backend=components.backend,
                callback=evaluate_held_out_test,
            )
            result = {
                "schema_version": 1,
                "completed": bool(loop_result.completed),
                "already_complete_on_entry": bool(loop_result.already_complete),
                "resumed": bool(loop_result.resumed),
                "fixed_horizon_coverages": int(training["epochs"]),
                "cursor": loop_result.cursor.as_dict(),
                "trained_anchors_this_invocation": loop_result.trained_anchors,
                "trained_optimizer_steps_this_invocation": (
                    loop_result.trained_optimizer_steps
                ),
                "held_out_test": dict(test_metrics),
                "performance_gate": None,
                "early_stopping": False,
                "run_contract": str(contracts_dir / "run_contract.json"),
                "final_checkpoint": str(loop_result.final_checkpoint),
                "completion_marker": str(loop_result.completion_marker),
                "logs": {
                    "text": str(logs / "train.log"),
                    "jsonl": str(logs / "train.metrics.jsonl"),
                },
                "gpu": {
                    "name": gpu.name,
                    "total_memory_gib": gpu_total_gib,
                    "device": "cuda:0",
                    "precision": "bf16",
                },
            }
            atomic_write_json(run_root / "formal_training_result.json", result)
            journal.event(
                "formal_training_workflow_completed",
                "Training and one-time held-out TEST are complete",
                **result,
            )
            journal.sync()
            return result
        except Exception as error:
            journal.fatal(
                "formal_training_workflow_fatal",
                str(error),
                exception_type=type(error).__name__,
            )
            journal.sync()
            raise


__all__ = [
    "inspect_workstation",
    "run_preprocessing_workflow",
    "run_training_workflow",
]
