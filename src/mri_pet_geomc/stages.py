"""Auditable stages for the main single-model GeoMC pipeline."""

from __future__ import annotations

import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .config import METHOD_NAME, get
from .data.automatic_qc import evaluate_registered_pair
from .data.observations import (
    CrossModalLink,
    Observation,
    ObservationTables,
    ObservationVolumeCache,
    build_observation_tables,
    import_legacy_pair_pet128,
    load_observation_tables,
    stable_observation_uid,
    write_observation_tables,
)
from .field.geometry_embedding import GeometryEmbedding
from .geometry.fem import FEMGeometry, build_fem_geometry
from .geometry.frame import ParsevalResolventFrame
from .model.geomc import GeoMCCore
from .pipeline import StageContext, StageResult
from .training.engine import (
    _autocast,
    _forward_prepared,
    _hash_seed,
    _make_sampler,
    _seed_everything,
    build_encoder,
    build_fixed_reference_targets,
    build_model_geometry,
    build_foundation_model,
    evaluate_checkpoint,
    run_small_sample_overfit_check,
    prepare_example,
    run_training,
)
from .training.jepa import representation_diagnostics
from .utils import (
    atomic_write_json,
    atomic_write_text,
    digest_object,
    read_json,
    sha256_file,
    source_tree_digest,
    utc_now,
)


def _mode(ctx: StageContext) -> str:
    return str(get(ctx.config, "run.mode"))


def _observation_paths(launch_dir: Path) -> tuple[Path, Path, Path]:
    root = launch_dir / "stages" / "01.observations" / "tables"
    return (
        root / "observations.jsonl",
        root / "verified_links.jsonl",
        root / "subject_splits.jsonl",
    )


def _load_tables(launch_dir: Path) -> ObservationTables:
    return load_observation_tables(*_observation_paths(launch_dir))


def _geometry_path(launch_dir: Path) -> Path:
    return launch_dir / "stages" / "04.geometry" / "fem_geometry.pt"


def _load_geometry(launch_dir: Path) -> FEMGeometry:
    return FEMGeometry.load(_geometry_path(launch_dir))


def _reference_path(ctx: StageContext, tables: ObservationTables) -> Path:
    if _mode(ctx) == "full":
        return Path(str(get(ctx.config, "paths.model_reference_128"))).resolve()
    return tables.observations[0].volume_path.resolve()


def _model_geometry(ctx: StageContext, tables: ObservationTables):
    geometry = _load_geometry(ctx.launch_dir)
    reference_path = _reference_path(ctx, tables)
    reference = nib.load(str(reference_path), mmap=True)
    return build_model_geometry(
        geometry,
        reference_shape=reference.shape[:3],
        reference_affine=np.asarray(reference.affine),
        grid_shape=get(ctx.config, "geometry.token_grid"),
        neighbors=int(get(ctx.config, "geometry.interpolation_neighbors", 8)),
        sigma_mm=float(get(ctx.config, "geometry.interpolation_sigma_mm", 12.0)),
    )


def _all_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _verify_pet128_preprocessing_record(
    observation: Observation,
    *,
    expected_relative_normalization: str,
) -> dict[str, Any]:
    """Verify the reused PET tensor's one-resample/relative-intensity preprocessing record."""

    preprocessing_record_path = observation.preprocessing_record_path
    if preprocessing_record_path is None or not preprocessing_record_path.is_file():
        raise RuntimeError(
            f"PET observation {observation.observation_uid} has no readable PET128 preprocessing record"
        )
    preprocessing_record = read_json(preprocessing_record_path)
    if str(preprocessing_record.get("status")) != "completed":
        raise RuntimeError(f"PET128 preprocessing record is not completed: {preprocessing_record_path}")
    count_fields = {
        "preprocessing_record": preprocessing_record.get("spatial_resample_count"),
        "plan": (preprocessing_record.get("plan") or {}).get("spatial_resample_count"),
        "execution": (preprocessing_record.get("execution") or {}).get("spatial_resample_count"),
    }
    if any(int(value or -1) != 1 for value in count_fields.values()):
        raise RuntimeError(
            f"PET128 preprocessing record does not prove exactly one spatial resample: "
            f"{preprocessing_record_path}; counts={count_fields}"
        )
    relative = preprocessing_record.get("relative_normalization") or {}
    sat3d = preprocessing_record.get("sat3d_normalization") or {}
    if int(relative.get("spatial_resample_count_added", -1)) != 0 or int(
        sat3d.get("spatial_resample_count_added", -1)
    ) != 0:
        raise RuntimeError(f"PET normalization added an unexpected resample: {preprocessing_record_path}")
    if str(relative.get("normalization")) != expected_relative_normalization:
        raise RuntimeError(
            f"PET relative-normalization contract mismatch: {preprocessing_record_path}"
        )
    if str(relative.get("quantification_contract")) != "relative_fdg_only":
        raise RuntimeError(f"PET preprocessing record does not prove relative FDG semantics: {preprocessing_record_path}")
    if any(
        value is not False
        for value in (
            preprocessing_record.get("suv_claim_allowed"),
            relative.get("suv_claim_allowed"),
            sat3d.get("suv_claim_allowed"),
        )
    ):
        raise RuntimeError(f"PET preprocessing record does not explicitly forbid an SUV claim: {preprocessing_record_path}")

    volume_path = observation.volume_path.resolve()
    volume_sha = sha256_file(volume_path)
    output_records = list(preprocessing_record.get("outputs") or [])
    matching_outputs = [
        record
        for record in output_records
        if Path(str(record.get("path", ""))).resolve() == volume_path
    ]
    if len(matching_outputs) != 1 or matching_outputs[0].get("sha256") != volume_sha:
        raise RuntimeError(
            f"PET model input is not hash-bound to its PET128 preprocessing record: {volume_path}"
        )
    if str(sat3d.get("output_sha256")) != volume_sha:
        raise RuntimeError(
            f"PET SAT3D normalization hash does not match the reused tensor: {volume_path}"
        )
    return {
        "preprocessing_record_path": str(preprocessing_record_path.resolve()),
        "preprocessing_record_sha256": sha256_file(preprocessing_record_path),
        "spatial_resample_count": 1,
        "relative_normalization": expected_relative_normalization,
        "quantification_contract": "relative_fdg_only",
        "suv_claim_allowed": False,
        "model_input_sha256": volume_sha,
    }


def geomc_preflight(ctx: StageContext) -> StageResult:
    method_name = str(get(ctx.config, "project.method"))
    if method_name != METHOD_NAME:
        raise RuntimeError("GeoMC stage received the wrong method name")
    mode = _mode(ctx)
    backend = str(get(ctx.config, "resources.encoder_backend", "tiny"))
    inputs: list[Path] = []
    path_status: dict[str, Any] = {}
    if mode == "full":
        required_files = (
            "legacy_pairings",
            "legacy_splits",
            "legacy_pet128_manifest",
            "sat3d_checkpoint",
            "model_reference_128",
            "model_mask_128",
        )
        for key in required_files:
            path = Path(str(get(ctx.config, f"paths.{key}"))).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"paths.{key}={path}")
            inputs.append(path)
            path_status[key] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        dataset = Path(str(get(ctx.config, "paths.dataset_root"))).resolve()
        code_root = Path(str(get(ctx.config, "paths.sat3d_code_root"))).resolve()
        if not dataset.is_dir() or not code_root.is_dir():
            raise FileNotFoundError(
                f"dataset/code root missing: dataset={dataset}, sat3d={code_root}"
            )
        expected = str(get(ctx.config, "sat3d.expected_checkpoint_sha256"))
        if path_status["sat3d_checkpoint"]["sha256"] != expected:
            raise RuntimeError("Pinned SAT3D checkpoint SHA-256 does not match")
        from .model.trainable_sat3d import sat3d_source_tree_sha256

        python_root, actual_source_sha = sat3d_source_tree_sha256(code_root)
        expected_source_sha = str(get(ctx.config, "sat3d.expected_source_tree_sha256"))
        if actual_source_sha.lower() != expected_source_sha.lower():
            raise RuntimeError(
                "Pinned SAT3D source-tree SHA-256 does not match: "
                f"expected={expected_source_sha}, actual={actual_source_sha}"
            )
        path_status["sat3d_source_tree"] = {
            "path": str(python_root),
            "sha256": actual_source_sha,
            "expected_sha256": expected_source_sha,
        }
        source_files = sorted(
            path for path in python_root.rglob("*.py") if path.is_file()
        )
        if not source_files:
            raise RuntimeError(f"Pinned SAT3D source tree has no Python files: {python_root}")
        inputs.extend(source_files)
        path_status["sat3d_source_tree"]["python_file_count"] = len(source_files)
        if backend != "sat3d":
            raise RuntimeError("Full run must use the audited SAT3D backend")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Full server run requires CUDA with BF16 support")
    payload = {
        "schema_version": 1,
        "passed": True,
        "method": method_name,
        "mode": mode,
        "encoder_backend": backend,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": bool(
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ),
        "source_digest": source_tree_digest(ctx.project_root),
        "path_status": path_status,
        "execution_policy": "exhaustive_nonblocking_v1",
        "manual_review_required": False,
        "reuse_boundary": {
            "read_only": [
                "legacy pair/split manifests",
                "one-resample PET128 derivatives",
                "reference and mask",
                "SAT3D source and checkpoint",
            ],
            "not_model_inputs": [
                "legacy frozen SAT3D feature cache",
                "legacy PCA projection",
                "task-specific checkpoints from the archived feasibility repository",
            ],
        },
        "created_at": utc_now(),
    }
    output = atomic_write_json(ctx.stage_dir / "preflight.json", payload)
    return StageResult(outputs=[output], inputs=inputs, details=payload)


def _synthetic_tables(ctx: StageContext) -> tuple[ObservationTables, list[Path]]:
    count = int(get(ctx.config, "data.expected_subjects", 8))
    split_counts = dict(get(ctx.config, "data.split_counts", {}))
    size = int(get(ctx.config, "data.synthetic_input_size", 16))
    root = ctx.stage_dir / "synthetic_data"
    root.mkdir(parents=True, exist_ok=True)
    coordinate = np.stack(
        np.meshgrid(
            np.linspace(-1.0, 1.0, size),
            np.linspace(-1.0, 1.0, size),
            np.linspace(-1.0, 1.0, size),
            indexing="ij",
        ),
        axis=0,
    )
    radius = np.sqrt(np.square(coordinate).sum(axis=0))
    mask = (radius < 0.88).astype(np.uint8)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    mask_path = root / "reference_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), str(mask_path))
    files = [mask_path]
    split_sequence = [
        split
        for split in ("train", "validation", "test")
        for _ in range(int(split_counts[split]))
    ]
    if len(split_sequence) != count:
        raise RuntimeError("Synthetic split counts do not match subject count")
    observations: list[Observation] = []
    links: list[CrossModalLink] = []
    subject_splits: dict[str, str] = {}
    for index in range(count):
        subject = f"SYN{index:03d}"
        split = split_sequence[index]
        subject_splits[subject] = split
        generator = np.random.default_rng(20260806 + index)
        center = generator.uniform(-0.35, 0.35, size=3)
        blob = np.exp(
            -sum((coordinate[axis] - center[axis]) ** 2 for axis in range(3))
            / generator.uniform(0.10, 0.24)
        )
        anatomy = (
            0.8 * (1.0 - radius)
            + 0.7 * blob
            + 0.12 * coordinate[0]
            - 0.08 * coordinate[2]
        ) * mask
        mri = (anatomy + 0.025 * generator.standard_normal(anatomy.shape)) * mask
        pet = (
            0.65 * gaussian_filter(anatomy, sigma=1.0)
            + 0.25 * blob
            + 0.025 * generator.standard_normal(anatomy.shape)
        ) * mask
        per_modality: dict[str, Observation] = {}
        for modality, array in (("mri", mri), ("pet", pet)):
            path = root / f"{subject}_{modality}.nii.gz"
            nib.save(nib.Nifti1Image(array.astype(np.float32), affine), str(path))
            files.append(path)
            observation = Observation(
                observation_uid=stable_observation_uid("synthetic", subject, modality),
                acquisition_uid=f"{subject}-{modality}",
                dataset_id="synthetic_implementation_check",
                subject_id=subject,
                session_id="S001",
                split=split,
                modality=modality,
                volume_path=path,
                brain_mask_path=mask_path,
                reliability=1.0,
                intensity_preprocessing="synthetic_no_scientific_claim",
                metadata={
                    "synthetic": True,
                    "site": "SYNTHETIC_SITE",
                    "acquisition_metadata": {},
                    "acquisition_metadata_available": False,
                },
            )
            observations.append(observation)
            per_modality[modality] = observation
        links.append(
            CrossModalLink(
                link_uid=stable_observation_uid("synthetic-link", subject),
                subject_id=subject,
                split=split,
                mri_observation_uid=per_modality["mri"].observation_uid,
                pet_observation_uid=per_modality["pet"].observation_uid,
                verified=True,
                verification_method="synthetic_generator_identity",
                verification_scope="implementation_check_only",
                metadata={"synthetic": True},
            )
        )
    return build_observation_tables(observations, links, subject_splits), files


def geomc_observations(ctx: StageContext) -> StageResult:
    if _mode(ctx) == "smoke":
        tables, generated = _synthetic_tables(ctx)
        inputs: list[Path] = []
    else:
        pair_path = Path(str(get(ctx.config, "paths.legacy_pairings"))).resolve()
        split_path = Path(str(get(ctx.config, "paths.legacy_splits"))).resolve()
        pet_path = Path(str(get(ctx.config, "paths.legacy_pet128_manifest"))).resolve()
        tables = import_legacy_pair_pet128(
            pair_path, pet_path, split_manifest=split_path
        )
        generated = []
        inputs = [pair_path, split_path, pet_path]
    expected = int(get(ctx.config, "data.expected_subjects"))
    if len(tables.subject_splits) != expected:
        raise RuntimeError(
            f"Observation subject count {len(tables.subject_splits)} != expected {expected}"
        )
    expected_split_counts = {
        str(split): int(count)
        for split, count in dict(get(ctx.config, "data.split_counts", {})).items()
    }
    actual_split_counts = {
        split: sum(value == split for value in tables.subject_splits.values())
        for split in expected_split_counts
    }
    if actual_split_counts != expected_split_counts:
        raise RuntimeError(
            "Imported subject splits do not match the frozen config contract: "
            f"actual={actual_split_counts}, expected={expected_split_counts}"
        )
    if _mode(ctx) == "full":
        modality_counts = {
            modality: sum(item.modality == modality for item in tables.observations)
            for modality in ("mri", "pet")
        }
        verified_links = sum(item.verified for item in tables.links)
        if modality_counts != {"mri": expected, "pet": expected}:
            raise RuntimeError(
                "The frozen 30-pair feasibility cohort must contain one MRI and "
                f"one PET observation per subject: {modality_counts}"
            )
        if verified_links != expected:
            raise RuntimeError(
                "The frozen paired feasibility cohort must contain one verified "
                f"MRI/PET link per subject: {verified_links} != {expected}"
            )
    files = write_observation_tables(tables, ctx.stage_dir / "tables")
    summary = read_json(files.summary_path)
    summary.update(
        {
            "fmri_observations": 0,
            "pseudo_pairs_allowed": False,
            "subject_split_precedes_relation_construction": True,
        }
    )
    atomic_write_json(files.summary_path, summary)
    return StageResult(
        outputs=[*files.outputs, *generated],
        inputs=inputs,
        metrics={
            "subjects": float(len(tables.subject_splits)),
            "observations": float(len(tables.observations)),
            "verified_links": float(sum(item.verified for item in tables.links)),
        },
        details=summary,
    )


def geomc_asset_snapshot(ctx: StageContext) -> StageResult:
    tables = _load_tables(ctx.launch_dir)
    cache = ObservationVolumeCache(
        tables.observations, max_items=max(16, len(tables.observations))
    )
    inputs = list(_observation_paths(ctx.launch_dir))
    rows: list[dict[str, Any]] = []
    reference_shape: tuple[int, ...] | None = None
    reference_affine: np.ndarray | None = None
    if _mode(ctx) == "full":
        reference_path = Path(str(get(ctx.config, "paths.model_reference_128"))).resolve()
        mask_path = Path(str(get(ctx.config, "paths.model_mask_128"))).resolve()
        reference = nib.load(str(reference_path), mmap=True)
        reference_shape = tuple(int(value) for value in reference.shape[:3])
        reference_affine = np.asarray(reference.affine)
        inputs.extend([reference_path, mask_path])
    for observation in tables.observations:
        observation_volume = cache.get(observation.observation_uid)
        image = nib.load(str(observation.volume_path), mmap=True)
        shape = tuple(int(value) for value in image.shape[:3])
        affine_match = bool(
            _mode(ctx) != "full"
            or (
                shape == reference_shape
                and np.allclose(image.affine, reference_affine, atol=1.0e-5, rtol=0.0)
            )
        )
        pet128_provenance: dict[str, Any] | None = None
        if _mode(ctx) == "full" and observation.modality == "pet":
            pet128_provenance = _verify_pet128_preprocessing_record(
                observation,
                expected_relative_normalization=str(
                    get(ctx.config, "preprocessing.pet128.relative_normalization")
                ),
            )
            assert observation.preprocessing_record_path is not None
            inputs.append(observation.preprocessing_record_path)
        elif _mode(ctx) == "full" and observation.preprocessing_record_path is not None:
            if not observation.preprocessing_record_path.is_file():
                raise RuntimeError(
                    f"Observation preprocessing record is missing: {observation.preprocessing_record_path}"
                )
            inputs.append(observation.preprocessing_record_path)
        row = {
            "observation_uid": observation.observation_uid,
            "subject_id": observation.subject_id,
            "modality": observation.modality,
            "volume_path": str(observation.volume_path),
            "volume_sha256": sha256_file(observation.volume_path),
            "mask_path": str(observation.brain_mask_path),
            "mask_sha256": sha256_file(observation.brain_mask_path),
            "shape": list(shape),
            "reference_affine_matches": affine_match,
            "normalization_mean": observation_volume.normalization_mean,
            "normalization_std": observation_volume.normalization_std,
            "relative_pet_only": observation.modality == "pet",
            "suv_claim_allowed": False if observation.modality == "pet" else None,
            "pet128_provenance": pet128_provenance,
        }
        rows.append(row)
        inputs.extend([observation.volume_path, observation.brain_mask_path])
    findings = [
        row["observation_uid"]
        for row in rows
        if not bool(row["reference_affine_matches"])
    ]
    payload = {
        "schema_version": 1,
        "status": "completed_with_findings" if findings else "completed",
        "asset_count": len(rows),
        "pet_count": sum(row["modality"] == "pet" for row in rows),
        "one_resample_pet_contract_applicable": _mode(ctx) == "full",
        "one_resample_pet_contract_verified": bool(
            _mode(ctx) == "full"
            and all(
                row["pet128_provenance"] is not None
                for row in rows
                if row["modality"] == "pet"
            )
        ),
        "reference_affine_findings": findings,
        "findings_block_execution": False,
        "assets": rows,
        "asset_identity": digest_object(rows),
    }
    output = atomic_write_json(ctx.stage_dir / "asset_snapshot.json", payload)
    return StageResult(
        outputs=[output],
        inputs=sorted(set(inputs)),
        metrics={"assets": float(len(rows)), "findings": float(len(findings))},
        details={key: value for key, value in payload.items() if key != "assets"},
    )


def geomc_registration_qc(ctx: StageContext) -> StageResult:
    """Automatic diagnostics only; no signature and no training gate."""

    tables = _load_tables(ctx.launch_dir)
    by_uid = tables.observations_by_uid
    rows: list[dict[str, Any]] = []
    for link in sorted(tables.links, key=lambda value: value.subject_id):
        if not link.verified:
            continue
        mri = by_uid[link.mri_observation_uid]
        pet = by_uid[link.pet_observation_uid]
        try:
            result = evaluate_registered_pair(
                mri_volume_path=mri.volume_path,
                mri_mask_path=mri.brain_mask_path,
                pet_volume_path=pet.volume_path,
                pet_mask_path=pet.brain_mask_path,
                affine_tolerance=float(get(ctx.config, "registration_qc.affine_tolerance")),
                minimum_support_voxels=int(
                    get(ctx.config, "registration_qc.minimum_support_voxels")
                ),
                diagnostic_maximum_axis=int(
                    get(ctx.config, "registration_qc.diagnostic_maximum_axis")
                ),
                histogram_bins=int(get(ctx.config, "registration_qc.histogram_bins")),
                mask_dice_flag_below=float(
                    get(ctx.config, "registration_qc.mask_dice_flag_below")
                ),
                centroid_distance_mm_flag_above=float(
                    get(ctx.config, "registration_qc.centroid_distance_mm_flag_above")
                ),
                nmi_shift_margin_flag_below=float(
                    get(ctx.config, "registration_qc.nmi_shift_margin_flag_below")
                ),
            )
            rows.append(
                {
                    "subject_id": link.subject_id,
                    "link_uid": link.link_uid,
                    "input_validation_passed": bool(result["input_validation_passed"]),
                    "input_validation_failures": list(result["input_validation_failures"]),
                    "statistical_flags": list(result["statistical_flags"]),
                    "checks": result["checks"],
                    "diagnostics": result["diagnostics"],
                }
            )
        except Exception as error:
            rows.append(
                {
                    "subject_id": link.subject_id,
                    "link_uid": link.link_uid,
                    "input_validation_passed": False,
                    "input_validation_failures": [f"automatic_qc_exception:{type(error).__name__}:{error}"],
                    "statistical_flags": [],
                    "checks": {},
                    "diagnostics": {},
                }
            )
    validation_failure_count = sum(len(row["input_validation_failures"]) for row in rows)
    flags = sum(len(row["statistical_flags"]) for row in rows)
    payload = {
        "schema_version": 1,
        "status": "completed_with_findings" if validation_failure_count or flags else "completed",
        "policy": "automatic_nonblocking_v1",
        "subject_count": len(rows),
        "input_validation_failure_count": validation_failure_count,
        "statistical_flag_count": flags,
        "manual_review_required": False,
        "manual_visual_review_performed": False,
        "statistical_flags_block_training": False,
        "findings_block_execution": False,
        "rows": rows,
        "limitations": [
            "Automatic diagnostics cannot prove absence of subtle registration error.",
            "These uncalibrated flags are not labels and do not suppress experiments.",
        ],
    }
    output = atomic_write_json(ctx.stage_dir / "automatic_registration_qc.json", payload)
    return StageResult(
        outputs=[output],
        inputs=list(_observation_paths(ctx.launch_dir)),
        metrics={"subjects": float(len(rows)), "input_validation_findings": float(validation_failure_count), "flags": float(flags)},
        details={key: value for key, value in payload.items() if key != "rows"},
    )


def geomc_geometry(ctx: StageContext) -> StageResult:
    tables = _load_tables(ctx.launch_dir)
    mask_path = (
        Path(str(get(ctx.config, "paths.model_mask_128"))).resolve()
        if _mode(ctx) == "full"
        else tables.observations[0].brain_mask_path.resolve()
    )
    geometry_path = ctx.stage_dir / "fem_geometry.pt"
    geometry = build_fem_geometry(
        mask_path,
        nodes=int(get(ctx.config, "geometry.nodes_target")),
        modes=int(get(ctx.config, "geometry.modes")),
        output_path=geometry_path,
    )
    model_geometry = _model_geometry(ctx, tables)
    generator = torch.Generator(device="cpu").manual_seed(
        int(get(ctx.config, "project.seed"))
    )
    probe = torch.randn(2, geometry.node_count, 5, generator=generator)
    frame = ParsevalResolventFrame(
        geometry.eigenvalues,
        geometry.eigenvectors,
        geometry.mass,
        radii=get(ctx.config, "geometry.resolvent_radii_mm"),
        normalize_spectrum=False,
        factorization="parseval_sqrt",
    )
    frame_diagnostics = frame.diagnostics(probe).as_dict()
    summary = {
        "schema_version": 2,
        "domain": str(get(ctx.config, "geometry.domain")),
        "domain_claim": "volumetric FEM feasibility proxy, not cortical-surface proof",
        "nodes": geometry.node_count,
        "modes": geometry.mode_count,
        "tokens": int(model_geometry.token_xyz_mm.shape[0]),
        "spectral_scales_including_complement": len(
            frame_diagnostics["scale_energy_fractions"]
        ),
        "resolvent_radii_mm": [
            float(value)
            for value in get(ctx.config, "geometry.resolvent_radii_mm")
        ],
        "physical_spectrum_normalized": False,
        "eigenvalue_units": "mm^-2",
        "exact_complement": True,
        "model_geometry_digest": model_geometry.contract_digest,
        "geometry_hash": geometry.metadata.get("geometry_hash"),
        "primary_frame_factorization": "parseval_sqrt",
        "frame_diagnostics": frame_diagnostics,
        "features_are_interpolated_to_fem_nodes": True,
        "geomc_runs_on_fem_nodes": True,
    }
    summary_path = atomic_write_json(ctx.stage_dir / "geometry_summary.json", summary)
    return StageResult(
        outputs=[geometry_path, summary_path],
        inputs=[mask_path, *_observation_paths(ctx.launch_dir)],
        metrics={
            "nodes": float(geometry.node_count),
            "modes": float(geometry.mode_count),
            "frame_reconstruction_relative_l2": float(frame_diagnostics["reconstruction_relative_l2_error"]),
            "parseval_energy_closure_relative_error": float(
                frame_diagnostics["additive_energy_closure_relative_error"]
            ),
        },
        details=summary,
    )


def geomc_model_check(ctx: StageContext) -> StageResult:
    """Run finite-output and gradient checks for the configured GeoMC model core."""

    seed = int(get(ctx.config, "project.seed"))
    _seed_everything(seed)
    geometry = _load_geometry(ctx.launch_dir)
    experiment = _main_experiment(ctx.config)
    latent_dim = min(int(get(ctx.config, "model.latent_dim")), 32)
    frame = ParsevalResolventFrame(
        geometry.eigenvalues,
        geometry.eigenvectors,
        geometry.mass,
        radii=get(ctx.config, "geometry.resolvent_radii_mm"),
        normalize_spectrum=False,
        factorization="parseval_sqrt",
    )
    core = GeoMCCore(
        frame,
        latent_dim,
        num_blocks=int(get(ctx.config, "model.geomc_blocks")),
        hidden_dim=max(64, 2 * latent_dim),
        scale_embedding_dim=int(get(ctx.config, "model.geomc_scale_embedding_dim")),
        residual_scale=float(get(ctx.config, "model.geomc_residual_scale")),
    )
    geometry_embedding = GeometryEmbedding(
        frame,
        latent_dim,
        node_xyz_mm=geometry.node_xyz_mm,
        feature_type=str(get(ctx.config, "geometry_embedding.type")),
        hidden_dim=int(get(ctx.config, "geometry_embedding.hidden_dim")),
        initial_scale=float(get(ctx.config, "geometry_embedding.initial_scale")),
        trainable_scale=bool(get(ctx.config, "geometry_embedding.trainable_scale")),
    )
    input_field = torch.randn(
        2, geometry.node_count, latent_dim, requires_grad=True
    )
    output, diagnostics = core(
        input_field,
        geometry_embedding(2, reference=input_field),
        return_diagnostics=True,
    )
    objective = output.float().square().mean()
    named_parameters = [
        *((f"core.{name}", parameter) for name, parameter in core.named_parameters()),
        *((f"geometry_embedding.{name}", parameter) for name, parameter in geometry_embedding.named_parameters()),
    ]
    named_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        objective,
        [parameter for _, parameter in named_parameters],
        allow_unused=True,
    )
    gradient_rows = [
        (name, gradient)
        for (name, _), gradient in zip(named_parameters, gradients)
        if gradient is not None
    ]
    routing_gradients = [
        gradient for name, gradient in gradient_rows if ".routing_mlp." in name
    ]
    scale_update_gradients = [
        gradient for name, gradient in gradient_rows if ".scale_update." in name
    ]
    first_routing = diagnostics.blocks[0].routing_matrix_mean
    identity_routing_at_initialization = all(
        abs(float(value) - (1.0 if row == column else 0.0)) < 1.0e-7
        for row, row_values in enumerate(first_routing)
        for column, value in enumerate(row_values)
    )
    routing_gradient_l1 = float(
        sum(value.detach().float().abs().sum() for value in routing_gradients).cpu()
        if routing_gradients
        else 0.0
    )
    scale_update_gradient_l1 = float(
        sum(
            value.detach().float().abs().sum()
            for value in scale_update_gradients
        ).cpu()
        if scale_update_gradients
        else 0.0
    )
    frame_diagnostics = frame.diagnostics(input_field.detach()).as_dict()
    implementation_checks = {
        "output_finite": bool(torch.isfinite(output).all()),
        "output_shape_matches_field": list(output.shape)
        == [2, geometry.node_count, latent_dim],
        "minimum_three_blocks": len(core.blocks) >= 3,
        "all_present_gradients_finite": bool(
            gradient_rows
            and all(torch.isfinite(value).all() for _, value in gradient_rows)
        ),
        "routing_parameters_trainable": bool(
            routing_gradients
            and all(torch.isfinite(value).all() for value in routing_gradients)
            and routing_gradient_l1 > 0.0
        ),
        "scale_update_trainable": bool(
            scale_update_gradients
            and all(torch.isfinite(value).all() for value in scale_update_gradients)
            and scale_update_gradient_l1 > 0.0
        ),
        "identity_routing_initialization": identity_routing_at_initialization,
        "frame_reconstruction_finite": math.isfinite(
            float(frame_diagnostics["reconstruction_relative_l2_error"])
        ),
        "frame_energy_closure_finite": math.isfinite(
            float(frame_diagnostics["additive_energy_closure_relative_error"])
        ),
    }
    if not all(implementation_checks.values()):
        raise RuntimeError(f"GeoMC model check failed: {implementation_checks}")

    row = {
        "experiment_id": str(experiment["id"]),
        "method": METHOD_NAME,
        "geometry_embedding_type": str(get(ctx.config, "geometry_embedding.type")),
        "routing_mode": core.routing_mode,
        "frame_factorization": frame.factorization,
        "output_shape": list(output.shape),
        "block_count": len(core.blocks),
        "trainable_parameters": sum(
            parameter.numel() for _, parameter in named_parameters
        ),
        "routing_gradient_l1": routing_gradient_l1,
        "scale_update_gradient_l1": scale_update_gradient_l1,
        "identity_routing_at_initialization": identity_routing_at_initialization,
        "core_diagnostics": diagnostics.as_dict(),
        "frame_diagnostics": frame_diagnostics,
        "geometry_embedding_diagnostics": geometry_embedding.diagnostics().as_dict(),
        "implementation_checks": implementation_checks,
    }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "execution_succeeded": True,
        "supports_scientific_claims": False,
        "method": METHOD_NAME,
        "experiment_count": 1,
        "rows": [row],
        "model_check_scope": (
            "one aggregated observation field, one Parseval resolvent frame, "
            "and one node-wise scale-routing core"
        ),
        "routing_domain": "fem_nodes",
        "parseval_frame_includes_orthogonal_complement": True,
        "decoder_reads_core_output_only": True,
        "target_metadata_used_only_by_decoder": True,
        "findings_block_training": False,
    }
    output_path = atomic_write_json(ctx.stage_dir / "model_check.json", payload)
    return StageResult(
        outputs=[output_path],
        inputs=[_geometry_path(ctx.launch_dir)],
        metrics={
            "experiment_count": 1.0,
            "execution_succeeded": 1.0,
            "routing_gradient_l1": routing_gradient_l1,
            "scale_update_gradient_l1": scale_update_gradient_l1,
        },
        details={key: value for key, value in payload.items() if key != "rows"},
    )


def _main_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    primary_identifier = str(
        get(config, "evaluation.primary_experiment_id", "geomc")
    )
    for item in get(config, "experiments", []):
        if str(item.get("id")) == primary_identifier:
            return dict(item)
    raise RuntimeError(
        f"Configuration has no primary experiment {primary_identifier!r}"
    )


def _nonblocking_failure(
    path: Path,
    *,
    kind: str,
    error: Exception,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "execution_failed_nonblocking",
        "kind": kind,
        "execution_succeeded": False,
        "findings_block_other_experiments": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        **dict(extra or {}),
    }
    atomic_write_json(path, payload)
    return payload


def geomc_smoke_test(ctx: StageContext) -> StageResult:
    """Run one real forward/backward/update through the main model."""

    output_path = ctx.stage_dir / "smoke_test.json"
    try:
        tables = _load_tables(ctx.launch_dir)
        model_geometry = _model_geometry(ctx, tables)
        experiment = _main_experiment(ctx.config)
        device = torch.device(str(get(ctx.config, "resources.device", "cpu")))
        precision = str(get(ctx.config, "training.precision", "fp32"))
        seed = int(get(ctx.config, "project.seed"))
        _seed_everything(seed)
        model = build_foundation_model(
            ctx.config, experiment, model_geometry, seed=seed
        ).to(device).train()
        optimizer = torch.optim.AdamW(
            model.optimizer_groups(
                encoder_learning_rate=float(
                    get(ctx.config, "training.encoder_learning_rate")
                ),
                model_learning_rate=float(
                    get(ctx.config, "training.model_learning_rate")
                ),
                weight_decay=float(get(ctx.config, "training.weight_decay")),
            )
        )
        sampler = _make_sampler(
            tables, ctx.config, experiment, split="train", seed=seed
        )
        example = sampler.groups.get("mri->pet", sampler.supported_examples)[0]
        cache = ObservationVolumeCache(tables.observations, max_items=4)
        prepared = prepare_example(
            cache,
            example,
            grid=model_geometry.grid_shape,
            hidden_fraction=float(get(ctx.config, "masking.hidden_fraction")),
            query_tokens=int(get(ctx.config, "masking.query_tokens")),
            include_target_observation_probability=float(
                get(ctx.config, "inputs.include_target_observation_probability")
            ),
            metadata_dim=int(get(ctx.config, "model.metadata_dim")),
            seed=seed,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output, objective = _forward_prepared(
            model,
            prepared,
            device=device,
            precision=precision,
            collect_diagnostics=True,
        )
        objective.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        diagnostics = representation_diagnostics(
            output.predicted_latents,
            output.target_latents,
            output.query_mask,
            query_weight=prepared.query_weight.to(device),
        )
        payload = {
            "schema_version": 1,
            "status": "completed",
            "execution_succeeded": True,
            "method": METHOD_NAME,
            "experiment_id": str(experiment["id"]),
            "loss": float(objective.loss.detach().cpu()),

            "gradient_norm": float(gradient_norm.detach().cpu()),
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": (
                float(torch.cuda.max_memory_allocated(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
            "prediction_shape": list(output.predicted_latents.shape),
            "field_shape": list(output.latent_field.shape),
            "input_observations": len(prepared.observations),
            "model_specification": model.model_specification(),
            "core_diagnostics": (
                output.core_diagnostics.as_dict()
                if output.core_diagnostics is not None
                else None
            ),
            "representation": diagnostics,
            "supports_scientific_claims": False,
            "findings_block_execution": False,
        }
        atomic_write_json(output_path, payload)
    except Exception as error:
        payload = _nonblocking_failure(
            output_path, kind="end_to_end_smoke_test", error=error
        )
    return StageResult(
        outputs=[output_path],
        inputs=[_geometry_path(ctx.launch_dir), *_observation_paths(ctx.launch_dir)],
        metrics={
            "execution_succeeded": float(bool(payload.get("execution_succeeded")))
        },
        details={
            key: value for key, value in payload.items() if key != "core_diagnostics"
        },
    )

def geomc_small_sample_overfit_check(ctx: StageContext) -> StageResult:
    output_path = ctx.stage_dir / "small_sample_overfit_check.json"
    try:
        tables = _load_tables(ctx.launch_dir)
        model_geometry = _model_geometry(ctx, tables)
        payload = run_small_sample_overfit_check(
            tables, model_geometry, ctx.config, output_dir=ctx.stage_dir / "training"
        )
        payload = {**payload, "execution_succeeded": True}
        atomic_write_json(output_path, payload)
    except Exception as error:
        payload = _nonblocking_failure(
            output_path, kind="small_sample_overfit_check", error=error
        )
    return StageResult(
        outputs=[output_path],
        inputs=[_geometry_path(ctx.launch_dir), *_observation_paths(ctx.launch_dir)],
        metrics={"execution_succeeded": float(bool(payload.get("execution_succeeded")))},
        details={key: value for key, value in payload.items() if key != "error"},
    )


def geomc_train(ctx: StageContext) -> StageResult:
    experiment = dict(ctx.spec.params["experiment"])
    seed = int(ctx.spec.params["seed"])
    result_path = ctx.stage_dir / "experiment_result.json"
    outputs = [result_path]
    try:
        tables = _load_tables(ctx.launch_dir)
        model_geometry = _model_geometry(ctx, tables)
        result = run_training(
            tables,
            model_geometry,
            ctx.config,
            experiment,
            seed=seed,
            output_dir=ctx.stage_dir / "training",
            event_logger=ctx.event_logger,
        )
        payload = {
            "schema_version": 2,
            "status": "completed",
            "execution_succeeded": True,
            "experiment": experiment,
            "seed": seed,
            "training_summary": result["summary_path"],
            "best_checkpoint": result["best_checkpoint"],
            "best_checkpoint_sha256": result["best_checkpoint_sha256"],
            "final_checkpoint": result["final_checkpoint"],
            "final_checkpoint_sha256": result["final_checkpoint_sha256"],
            "final_step": result["final_identity"]["step"],
            "best_validation_loss": result["best_validation_loss"],
            "model_specification": result["model_specification"],
            "training_contract_digest": result["training_contract_digest"],
            "encoder_provenance": result["encoder_provenance"],
            "findings_block_other_experiments": False,
        }
        atomic_write_json(result_path, payload)
        outputs.append(Path(result["summary_path"]))
        outputs.append(Path(result["best_checkpoint"]))
        outputs.append(Path(result["final_checkpoint"]))
    except Exception as error:
        payload = _nonblocking_failure(
            result_path,
            kind="experiment",
            error=error,
            extra={"experiment": experiment, "seed": seed},
        )
    stage_metrics = {
        "execution_succeeded": float(bool(payload.get("execution_succeeded")))
    }
    if "best_validation_loss" in payload:
        stage_metrics["best_validation_loss"] = float(payload["best_validation_loss"])
    return StageResult(
        outputs=outputs,
        inputs=[_geometry_path(ctx.launch_dir), *_observation_paths(ctx.launch_dir)],
        metrics=stage_metrics,
        details={key: value for key, value in payload.items() if key != "error"},
    )


def _experiment_stage_id(experiment_id: str) -> str:
    return f"08.experiment.{experiment_id}.seed0"


def geomc_evaluate(ctx: StageContext) -> StageResult:
    """Evaluate the one main checkpoint in its two valid metric spaces."""

    experiment = _main_experiment(ctx.config)
    identifier = str(experiment["id"])
    if identifier != "geomc":
        raise RuntimeError("The main evaluation accepts only experiment 'geomc'")
    if list(get(ctx.config, "evaluation.comparisons", [])):
        raise RuntimeError("Single-model GeoMC evaluation requires comparisons=[]")

    inputs: list[Path] = [
        _geometry_path(ctx.launch_dir),
        *_observation_paths(ctx.launch_dir),
    ]
    shared_errors: list[dict[str, Any]] = []
    result: dict[str, Any]
    try:
        tables = _load_tables(ctx.launch_dir)
        model_geometry = _model_geometry(ctx, tables)
    except Exception as error:
        shared_errors.append(
            {
                "component": "data_or_geometry",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        result = {
            "status": "evaluation_unavailable_shared_infrastructure",
            "error": shared_errors[-1],
        }
    else:
        fixed_reference_targets = None
        try:
            fixed_reference_targets = build_fixed_reference_targets(
                tables,
                model_geometry,
                ctx.config,
                seed=0,
                splits=("train", "test"),
            )
        except Exception as error:
            shared_errors.append(
                {
                    "component": "fixed_reference",
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )

        result_path = (
            ctx.launch_dir
            / "stages"
            / _experiment_stage_id(identifier)
            / "experiment_result.json"
        )
        if result_path.is_file():
            inputs.append(result_path)
        try:
            training_result = read_json(result_path)
            if not bool(training_result.get("execution_succeeded")):
                result = {
                    "status": "not_evaluated_training_failed",
                    "training_error": training_result.get("error"),
                }
            else:
                checkpoint = Path(
                    str(training_result["final_checkpoint"])
                ).resolve()
                inputs.append(checkpoint)
                actual_step = int(training_result.get("final_step", -1))
                planned_step = int(experiment["max_updates"])
                evaluation = evaluate_checkpoint(
                    tables,
                    model_geometry,
                    ctx.config,
                    experiment,
                    seed=0,
                    checkpoint_path=checkpoint,
                    fixed_reference_targets=fixed_reference_targets,
                    split="test",
                    maximum_examples=100000,
                )
                result = {
                    "status": "evaluated",
                    "checkpoint_contract": {
                        "policy": "fixed_final_checkpoint",
                        "actual_step": actual_step,
                        "planned_step": planned_step,
                        "matches_planned_updates": actual_step == planned_step,
                    },
                    "model_specification": training_result.get("model_specification"),
                    **evaluation,
                }
        except Exception as error:
            result = {
                "status": "evaluation_failed_nonblocking",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }

    ema_target_metrics = dict(result.get("ema_target_metrics") or {})
    fixed_reference = dict(result.get("fixed_reference") or {})
    absolute_metrics = {
        "ema_target_huber_loss": ema_target_metrics.get(
            "loss"
        ),
        "ema_target_mean_absolute_error": ema_target_metrics.get("mae"),
        "fixed_reference_aligned_huber_loss": fixed_reference.get(
            "fixed_reference_aligned_huber_loss"
        ),
        "fixed_reference_unaligned_huber_loss": fixed_reference.get(
            "fixed_reference_unaligned_huber_loss"
        ),
    }
    evaluated = result.get("status") == "evaluated"
    payload = {
        "schema_version": 1,
        "status": "complete" if evaluated else "evaluation_failed",
        "execution_succeeded": bool(evaluated),
        "method": METHOD_NAME,
        "primary_experiment_id": identifier,
        "requested_experiments": 1,
        "evaluated_experiments": int(evaluated),
        "single_model": True,
        "single_seed": True,
        "aggregate_by_subject": True,
        "results": {identifier: result},
        "absolute_metrics": absolute_metrics,
        "comparisons": [],
        "metric_spaces": {
            "ema_target": {
                "artifact_path": "results.geomc.ema_target_metrics",
                "purpose": "within-model optimization diagnostic",
                "cross_model_comparable": False,
            },
            "fixed_reference": {
                "artifact_path": "results.geomc.fixed_reference",
                "purpose": "absolute held-out score in a fixed target space",
                "alignment_fit_split": "train",
                "score_split": "test",
                "available": bool(fixed_reference),
            },
        },
        "shared_infrastructure_errors": shared_errors,
        "interpretation_limits": [
            "These are absolute scores for one trained model, not improvement estimates.",
            "EMA-target losses are optimization diagnostics and have no cross-model interpretation.",
            "A fixed-reference score is comparable only when another model is evaluated under the identical fixed-reference protocol.",
        ],
        "thresholds_block_execution": False,
    }
    output_path = atomic_write_json(ctx.stage_dir / "evaluation.json", payload)
    return StageResult(
        outputs=[output_path],
        inputs=[path for path in inputs if path.is_file()],
        metrics={
            "evaluated_experiments": float(evaluated),
            "execution_succeeded": float(evaluated),
            "fixed_reference_available": float(bool(fixed_reference)),
        },
        details={
            "status": payload["status"],
            "evaluated_experiments": int(evaluated),
            "requested_experiments": 1,
            "shared_infrastructure_errors": shared_errors,
        },
    )

def geomc_report(ctx: StageContext) -> StageResult:
    """Summarize one GeoMC run without manufacturing comparison claims."""

    report_inputs: list[Path] = []

    def optional_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {"status": "unavailable", "path": str(path)}
        report_inputs.append(path)
        try:
            return {
                "status": "available",
                "path": str(path),
                "payload": read_json(path),
            }
        except Exception as error:
            return {
                "status": "unreadable",
                "path": str(path),
                "error": f"{type(error).__name__}: {error}",
            }

    experiment = _main_experiment(ctx.config)
    identifier = str(experiment["id"])
    artifacts = {
        "observations": optional_json(
            ctx.launch_dir
            / "stages"
            / "01.observations"
            / "tables"
            / "observation_summary.json"
        ),
        "registration_qc": optional_json(
            ctx.launch_dir
            / "stages"
            / "03.registration_qc"
            / "automatic_registration_qc.json"
        ),
        "geometry": optional_json(
            ctx.launch_dir / "stages" / "04.geometry" / "geometry_summary.json"
        ),
        "model_check": optional_json(
            ctx.launch_dir
            / "stages"
            / "05.model_check"
            / "model_check.json"
        ),
        "smoke_test": optional_json(
            ctx.launch_dir
            / "stages"
            / "06.end_to_end_smoke_test"
            / "smoke_test.json"
        ),
        "small_sample_overfit_check": optional_json(
            ctx.launch_dir
            / "stages"
            / "07.small_sample_overfit_check"
            / "small_sample_overfit_check.json"
        ),
        "training": optional_json(
            ctx.launch_dir
            / "stages"
            / _experiment_stage_id(identifier)
            / "experiment_result.json"
        ),
        "evaluation": optional_json(
            ctx.launch_dir / "stages" / "09.evaluate" / "evaluation.json"
        ),
    }

    def payload(name: str) -> dict[str, Any]:
        return dict(artifacts[name].get("payload") or {})

    observation = payload("observations")
    model_check = payload("model_check")
    smoke_test = payload("smoke_test")
    small_sample_check = payload("small_sample_overfit_check")
    training = payload("training")
    evaluation = payload("evaluation")
    result = dict((evaluation.get("results") or {}).get(identifier) or {})
    absolute_metrics = dict(evaluation.get("absolute_metrics") or {})
    if not absolute_metrics:
        ema_target_metrics = dict(result.get("ema_target_metrics") or {})
        fixed_reference = dict(result.get("fixed_reference") or {})
        absolute_metrics = {
            "ema_target_huber_loss": ema_target_metrics.get(
                "loss"
            ),
            "ema_target_mean_absolute_error": ema_target_metrics.get("mae"),
            "fixed_reference_aligned_huber_loss": fixed_reference.get(
                "fixed_reference_aligned_huber_loss"
            ),
            "fixed_reference_unaligned_huber_loss": fixed_reference.get(
                "fixed_reference_unaligned_huber_loss"
            ),
        }

    execution = {
        "observation_table_available": artifacts["observations"]["status"]
        == "available",
        "geometry_available": artifacts["geometry"]["status"] == "available",
        "model_check_executed": bool(
            model_check.get("execution_succeeded")
        ),
        "smoke_test_executed": bool(smoke_test.get("execution_succeeded")),
        "small_sample_overfit_check_executed": bool(small_sample_check.get("execution_succeeded")),
        "training_executed": bool(training.get("execution_succeeded")),
        "evaluation_executed": result.get("status") == "evaluated",
        "fixed_reference_available": bool(result.get("fixed_reference")),
    }
    if execution["training_executed"] and execution["evaluation_executed"]:
        decision = "SINGLE_MODEL_EVALUATION_COMPLETE"
    elif execution["training_executed"]:
        decision = "TRAINED_EVALUATION_INCOMPLETE"
    else:
        decision = "EXECUTION_INCOMPLETE"

    interpretation_limits = [
        {
            "claim": "implementation",
            "status": "supported_only_if_recorded_stage_executed",
            "meaning": (
                "Model-check, smoke-test, and training artifacts establish the implementation path "
                "only for this run."
            ),
        },
        {
            "claim": "absolute_held_out_performance",
            "status": (
                "measured" if execution["evaluation_executed"] else "unavailable"
            ),
            "meaning": (
                "Reported EMA-target and fixed-reference values are absolute scores "
                "for the main model."
            ),
        },
        {
            "claim": "relative_improvement",
            "status": "not_identified",
            "meaning": (
                "There is no control model in this repository and comparisons "
                "are intentionally empty."
            ),
        },
        {
            "claim": "foundation_model_generality",
            "status": "not_identified",
            "meaning": (
                "A single cohort/run does not establish external transfer, "
                "scale, unpaired-data benefit or downstream generality."
            ),
        },
        {
            "claim": "biological_mechanism",
            "status": "not_identified",
            "meaning": (
                "Geometry-conditioned computation does not by itself prove a "
                "biological causal mechanism."
            ),
        },
    ]
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "decision": decision,
        "method": METHOD_NAME,
        "experiment_id": identifier,
        "single_model": True,
        "comparisons": [],
        "model_specification": (
            result.get("model_specification")
            or training.get("model_specification")
            or smoke_test.get("model_specification")
        ),
        "data_scope": {
            "subject_count": observation.get("subject_count"),
            "observation_count": observation.get("observation_count"),
            "verified_pair_count": observation.get("verified_pair_count"),
            "scientific_use": bool(
                get(ctx.config, "project.scientific_use", False)
            ),
        },
        "execution": execution,
        "absolute_metrics": absolute_metrics,
        "metric_interpretation": {
            "ema_target_huber_loss": (
                "Within-model optimization diagnostic in the model's own EMA "
                "target space; not a cross-model metric."
            ),
            "fixed_reference_aligned_huber_loss": (
                "Held-out absolute loss after train-split alignment to the "
                "frozen fixed reference, when available."
            ),
        },
        "model_check": {
            "status": artifacts["model_check"]["status"],
            "execution_succeeded": model_check.get("execution_succeeded"),
            "supports_scientific_claims": False,
            "model_check_scope": model_check.get("model_check_scope"),
        },
        "smoke_test": {
            "status": artifacts["smoke_test"]["status"],
            "execution_succeeded": smoke_test.get("execution_succeeded"),
            "loss": smoke_test.get("loss"),
        },
        "small_sample_overfit_check": {
            "status": artifacts["small_sample_overfit_check"]["status"],
            "execution_succeeded": small_sample_check.get("execution_succeeded"),
            "relative_drop": small_sample_check.get("relative_drop"),
            "supports_scientific_claims": False,
        },
        "training": {
            "status": artifacts["training"]["status"],
            "execution_succeeded": training.get("execution_succeeded"),
            "final_step": training.get("final_step"),
            "best_validation_loss": training.get("best_validation_loss"),
            "checkpoint": training.get("final_checkpoint"),
        },
        "evaluation": {
            "status": evaluation.get("status", "unavailable"),
            "execution_succeeded": evaluation.get("execution_succeeded", False),
            "result_status": result.get("status", "unavailable"),
            "shared_infrastructure_errors": evaluation.get(
                "shared_infrastructure_errors", []
            ),
        },
        "interpretation_limits": interpretation_limits,
        "claims_allowed": [
            "Whether each recorded stage executed.",
            "The absolute metrics explicitly present in this artifact.",
        ],
        "claims_not_allowed": [
            "Performance improvement over another architecture.",
            "Benefits caused by geometry, pairing, or any removed model version.",
            "External, downstream, unpaired-data, or foundation-model generality.",
            "A causal biological interpretation of learned routing.",
        ],
        "thresholds_block_execution": False,
        "artifact_status": {
            name: {
                key: value
                for key, value in artifact.items()
                if key != "payload"
            }
            for name, artifact in artifacts.items()
        },
    }

    def format_metric(value: Any) -> str:
        return f"{float(value):.6g}" if isinstance(value, (int, float)) else "unavailable"

    markdown = "\n".join(
        [
            "# Main GeoMC single-model report",
            "",
            f"- Decision: **{decision}**",
            f"- Method: `{METHOD_NAME}`",
            f"- Experiment: `{identifier}`",
            f"- Training executed: {execution['training_executed']}",
            f"- Evaluation executed: {execution['evaluation_executed']}",
            "",
            "## Absolute metrics",
            "",
            "| Metric | Value | Interpretation |",
            "|---|---:|---|",
            (
                "| EMA-target raw Huber | "
                + format_metric(absolute_metrics.get("ema_target_huber_loss"))
                + " | within-model optimization diagnostic |"
            ),
            (
                "| fixed-reference aligned Huber | "
                + format_metric(
                    absolute_metrics.get("fixed_reference_aligned_huber_loss")
                )
                + " | held-out absolute score when available |"
            ),
            "",
            "## Evidence boundary",
            "",
            (
                "This cleaned repository contains one model and no comparison "
                "arms. It therefore reports execution and absolute scores only; "
                "it does not establish relative improvement, external "
                "generality, or a causal biological mechanism."
            ),
            "",
        ]
    )
    json_path = atomic_write_json(ctx.stage_dir / "feasibility_report.json", report)
    markdown_path = atomic_write_text(
        ctx.stage_dir / "feasibility_report.md", markdown
    )
    return StageResult(
        outputs=[json_path, markdown_path],
        inputs=report_inputs,
        metrics={
            "training_executed": float(execution["training_executed"]),
            "evaluation_executed": float(execution["evaluation_executed"]),
            "fixed_reference_available": float(
                execution["fixed_reference_available"]
            ),
        },
        details={
            "decision": decision,
            "single_model": True,
            "experiment_id": identifier,
        },
    )

def run_geomc_gpu_smoke_test(config: Mapping[str, Any], launch_dir: Path) -> dict[str, Any]:
    """Standalone real-SAT3D forward/backward memory probe."""

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    experiment = _main_experiment(config)
    seed = int(get(config, "project.seed"))
    _seed_everything(seed)
    encoder = build_encoder(config, experiment, seed=seed).to(device).train()
    shape = tuple(int(value) for value in encoder.input_shape)
    volume = torch.randn(1, *shape, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with _autocast(device, str(get(config, "training.precision", "bf16"))):
        output = encoder(volume)
        loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gradients = [
        parameter.grad
        for parameter in encoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return {
        "schema_version": 1,
        "passed": bool(gradients and all(torch.isfinite(value).all() for value in gradients)),
        "input_shape": [1, *shape],
        "output_shape": list(output.shape),
        "trainability": encoder.trainability,
        "trainable_parameters": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3),
        "encoder_provenance": encoder.provenance,
        "supports_scientific_claims": False,
        "launch_dir": str(launch_dir.resolve()),
    }


HANDLERS = {
    "geomc_preflight": geomc_preflight,
    "geomc_observations": geomc_observations,
    "geomc_asset_snapshot": geomc_asset_snapshot,
    "geomc_registration_qc": geomc_registration_qc,
    "geomc_geometry": geomc_geometry,
    "geomc_model_check": geomc_model_check,
    "geomc_smoke_test": geomc_smoke_test,
    "geomc_small_sample_overfit_check": geomc_small_sample_overfit_check,
    "geomc_train": geomc_train,
    "geomc_evaluate": geomc_evaluate,
    "geomc_report": geomc_report,
}


__all__ = ["HANDLERS", "run_geomc_gpu_smoke_test"]
