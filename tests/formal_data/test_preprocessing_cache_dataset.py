from __future__ import annotations

import json
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import mri_pet_geomc.formal.data.preprocessing as preprocessing_module
from mri_pet_geomc.formal.data.cache import MMapCache
from mri_pet_geomc.formal.data.collate import packed_collate
from mri_pet_geomc.formal.data.dataset import (
    FormalSample,
    FormalVolumeDataset,
    load_preprocessing_overlay,
    robust_normalize_visible,
)
from mri_pet_geomc.formal.data.preprocessing import (
    OfflinePreprocessor,
    PreprocessingConfig,
    preprocessing_contract_digest,
)
from mri_pet_geomc.formal.data.reference import (
    ReferenceSpec,
    reference_contract_digest,
)
from mri_pet_geomc.formal.data.relations import build_relations
from mri_pet_geomc.formal.data.sampler import CoverageSampler
from mri_pet_geomc.formal.data.schema import (
    CatalogObservation,
    FormalDataError,
    SourceLocator,
)
from mri_pet_geomc.formal.workflows import _validate_cache_layout


def _save(path: Path, values: np.ndarray, affine: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(values.astype(np.float32), affine), path)
    return path


def _reference(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    affine[:3, 3] = -3.75
    reference = _save(tmp_path / "source_reference.nii.gz", np.ones((6, 6, 6)), affine)
    mask_values = np.zeros((6, 6, 6), dtype=np.uint8)
    mask_values[1:-1, 1:-1, 1:-1] = 1
    mask = _save(tmp_path / "source_mask.nii.gz", mask_values, affine)
    return reference, mask, affine


def _observation(uid: str, source: SourceLocator) -> CatalogObservation:
    return CatalogObservation(
        observation_uid=uid,
        dataset_id="PT001_ClevelandCCF",
        canonical_subject_id=f"FOMO:PT001_ClevelandCCF:{uid}",
        participant_id=uid,
        modality="mri",
        session_id="ses-01",
        source_session_id="source-session-01",
        acquisition_uid=f"acq-{uid}",
        source=source,
        split="train",
        sequence="T1w",
        product="T1w",
    )


def test_preprocessing_continues_failures_resumes_and_keeps_float16_finite(
    tmp_path: Path,
) -> None:
    source_reference, source_mask, affine = _reference(tmp_path)
    normal_path = _save(tmp_path / "normal.nii.gz", np.arange(216).reshape(6, 6, 6), affine)
    high_path = _save(tmp_path / "high.nii.gz", np.full((6, 6, 6), 1.0e8), affine)
    archive = tmp_path / "case.zip"
    member = "sub-a/ses-01/anat/a.nii.gz"
    with zipfile.ZipFile(archive, "w") as handle:
        # FOMO session archives may store the exact mapping path or strip the
        # already-implied subject/session prefix.
        handle.write(normal_path, "anat/a.nii.gz")
    observations = (
        _observation(
            "zip-ok",
            SourceLocator(
                "zip_member", str(archive), "FOMO/a.zip", member, signature="zip-sig"
            ),
        ),
        _observation(
            "high-ok",
            SourceLocator("nifti", str(high_path), "FOMO/high.nii.gz", signature="high-sig"),
        ),
        _observation(
            "missing",
            SourceLocator(
                "nifti", str(tmp_path / "missing.nii.gz"), "FOMO/missing.nii.gz", signature="missing-sig"
            ),
        ),
    )
    config = PreprocessingConfig(
        output_root=tmp_path / "cache",
        source_reference=source_reference,
        source_mask=source_mask,
        target_shape=(4, 4, 4),
        target_spacing=(2.0, 2.0, 2.0),
        shard_size=2,
        workers=1,
        registration_mode="direct",
    )
    first = OfflinePreprocessor(observations, config).run()
    assert (first.completed, first.skipped, first.failed) == (2, 0, 1)
    assert len(first.cache_entries) == 2
    assert first.reference.resumed is False
    source_centre = nib.affines.apply_affine(affine, np.array([2.5, 2.5, 2.5]))
    target_centre = nib.affines.apply_affine(
        first.reference.affine, np.array([1.5, 1.5, 1.5])
    )
    np.testing.assert_allclose(target_centre, source_centre)
    assert first.reference.spacing == (2.0, 2.0, 2.0)
    second = OfflinePreprocessor(observations, config).run()
    assert (second.completed, second.skipped, second.failed) == (0, 2, 1)
    assert second.reference.resumed is True
    expected_reference_contract = reference_contract_digest(
        ReferenceSpec(
            source_reference=source_reference,
            source_mask=source_mask,
            target_shape=config.target_shape,
            target_spacing=config.target_spacing,
        )
    )
    assert second.reference.contract_digest == expected_reference_contract
    assert (
        preprocessing_contract_digest(config, expected_reference_contract)
        == second.contract_digest
    )
    preprocessing_summary = json.loads(
        (config.output_root / "preprocessing_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        preprocessing_summary["reference_contract_digest"]
        == expected_reference_contract
    )
    cache = MMapCache(config.output_root, config.output_root / "cache_index.jsonl")
    assert cache.get("zip-ok").image.shape == (1, 4, 4, 4)
    high = cache.get("high-ok").image
    assert np.isfinite(high).all()
    receipt = json.loads(
        (config.output_root / "receipts" / "high-ok.json").read_text(encoding="utf-8")
    )
    assert receipt["intensity_storage_scale"] > 1.0
    assert receipt["stored_float16_finite"] is True
    assert receipt["geometry"]["native_shape"] == [6, 6, 6]
    assert receipt["geometry"]["native_spacing"] == [1.5, 1.5, 1.5]
    assert receipt["geometry"]["effective_spacing"] == [2.0, 2.0, 2.0]
    overlay = load_preprocessing_overlay(config.output_root / "receipts", "high-ok")
    assert overlay["geometry"]["native_orientation"] == "RAS"
    failures = (config.output_root / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(failures) == 2

    relations = build_relations(observations).relations
    dataset = FormalVolumeDataset(
        observations,
        relations,
        cache_root=config.output_root,
        cache_index=config.output_root / "cache_index.jsonl",
        split="train",
        relation_seed=5,
        receipt_dir=config.output_root / "receipts",
    )
    sampler = CoverageSampler(dataset.observation_uids, coverage_id=1, seed=7)
    samples = [dataset[key] for key in sampler]
    batch = packed_collate(samples)
    assert batch["anchor_image"].shape == (2, 1, 4, 4, 4)
    assert batch["companion_image"].shape[0] == 0
    assert all("preprocessing" in metadata for metadata in batch["anchor_metadata"])

    layout = _validate_cache_layout(
        cache_root=config.output_root,
        cache_entries=second.cache_entries,
        target_shape=config.target_shape,
        expected_contract_digest=second.contract_digest,
        expected_plan_digest=second.plan_digest,
    )
    assert layout["referenced_shards"] == 2
    summary_path = config.output_root / "cache_summary.json"
    cache_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache_summary["plan_digest"] = "tampered"
    summary_path.write_text(json.dumps(cache_summary), encoding="utf-8")
    with pytest.raises(FormalDataError, match="plan digest"):
        _validate_cache_layout(
            cache_root=config.output_root,
            cache_entries=second.cache_entries,
            target_shape=config.target_shape,
            expected_contract_digest=second.contract_digest,
            expected_plan_digest=second.plan_digest,
        )


def test_simpleitk_registration_branch_is_default_and_receipted(
    tmp_path: Path, monkeypatch
) -> None:
    source_reference, source_mask, affine = _reference(tmp_path)
    input_path = _save(tmp_path / "input.nii.gz", np.ones((6, 6, 6)), affine)
    observation = _observation(
        "registered",
        SourceLocator("nifti", str(input_path), "input.nii.gz", signature="sig"),
    )
    calls = []

    def fake_registration(moving_path, item, reference, config):
        calls.append((moving_path, item.observation_uid, config.registration_mode))
        return (
            np.ones(reference.shape, dtype=np.float32),
            np.ones(reference.shape, dtype=bool),
            {"type": "synthetic-affine", "parameters": [1.0], "sha256": "test"},
        )

    monkeypatch.setattr(
        preprocessing_module, "_simpleitk_rigid_affine", fake_registration
    )
    config = PreprocessingConfig(
        output_root=tmp_path / "cache",
        source_reference=source_reference,
        source_mask=source_mask,
        target_shape=(4, 4, 4),
        target_spacing=(2.0, 2.0, 2.0),
        workers=1,
    )
    result = OfflinePreprocessor([observation], config).run()
    assert result.completed == 1
    assert calls and calls[0][2] == "simpleitk_rigid_affine"
    receipt = json.loads(
        (config.output_root / "receipts" / "registered.json").read_text(encoding="utf-8")
    )
    assert receipt["registration_mode"] == "simpleitk_rigid_affine"
    assert receipt["final_interpolation"] == "bspline"
    assert receipt["geometry"]["registration_transform"]["sha256"] == "test"


def test_visible_normalization_does_not_read_hidden_outlier() -> None:
    image = torch.tensor([[[[1.0, 2.0], [3.0, 1000.0]]]])
    support = torch.ones((1, 2, 2), dtype=torch.bool)
    visible = torch.tensor([[[True, True], [True, False]]])
    normalized, stats = robust_normalize_visible(image, support, visible)
    assert stats["median"] == 2.0
    assert normalized.shape == image.shape


def test_nonzero_rigid_transform_is_copied_into_affine_stage() -> None:
    sitk = pytest.importorskip("SimpleITK")
    rigid = sitk.Euler3DTransform()
    rigid.SetCenter((8.0, -3.0, 2.0))
    rigid.SetRotation(0.11, -0.07, 0.19)
    rigid.SetTranslation((4.0, -2.5, 1.25))
    affine = preprocessing_module._rigid_to_affine(sitk, rigid)
    point = (13.5, 4.0, -7.0)
    np.testing.assert_allclose(
        affine.TransformPoint(point), rigid.TransformPoint(point), atol=1e-10, rtol=0.0
    )
    assert any(abs(value) > 0 for value in affine.GetTranslation())
