"""Materialize the fixed formal lattice from the bundled 1.5 mm reference.

The audited source reference and its non-boundary mask are immutable inputs.
The formal default is a 128 cubed, 2 mm isotropic lattice with the same world
centre and axis directions.  Materialization is hash-bound and resumable.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from ...utils import atomic_write_json, digest_object, read_json, sha256_file, utc_now
from .schema import FormalDataError


def _tuple3(values: Sequence[int | float], cast: type) -> tuple[Any, Any, Any]:
    result = tuple(cast(value) for value in values)
    if len(result) != 3:
        raise FormalDataError(f"Expected three spatial values, got {values!r}")
    return result  # type: ignore[return-value]


def _atomic_save_nifti(image: nib.spatialimages.SpatialImage, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".nii.gz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_text)
    try:
        nib.save(image, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ReferenceSpec:
    source_reference: Path
    source_mask: Path
    target_shape: tuple[int, int, int] = (128, 128, 128)
    target_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_reference", Path(self.source_reference))
        object.__setattr__(self, "source_mask", Path(self.source_mask))
        object.__setattr__(self, "target_shape", _tuple3(self.target_shape, int))
        object.__setattr__(self, "target_spacing", _tuple3(self.target_spacing, float))
        if any(value <= 0 for value in self.target_shape):
            raise FormalDataError("Reference target_shape values must be positive")
        if any(not np.isfinite(value) or value <= 0 for value in self.target_spacing):
            raise FormalDataError("Reference target_spacing values must be finite and positive")


@dataclass(frozen=True)
class MaterializedReference:
    reference_path: Path
    mask_path: Path
    receipt_path: Path
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    affine: np.ndarray
    contract_digest: str
    resumed: bool


def reference_contract_digest(spec: ReferenceSpec) -> str:
    """Recompute the immutable source-to-target lattice contract."""

    if not spec.source_reference.is_file() or not spec.source_mask.is_file():
        raise FormalDataError(
            "Bundled source reference and mask must both exist: "
            f"{spec.source_reference}, {spec.source_mask}"
        )
    return digest_object(
        {
            "schema": 1,
            "source_hashes": {
                "reference_sha256": sha256_file(spec.source_reference),
                "mask_sha256": sha256_file(spec.source_mask),
            },
            "target_shape": list(spec.target_shape),
            "target_spacing": list(spec.target_spacing),
            "world_centre_policy": "preserve_source_world_centre_and_axis_directions",
        }
    )


def _target_affine(
    source_affine: np.ndarray,
    source_shape: Sequence[int],
    target_shape: Sequence[int],
    target_spacing: Sequence[float],
) -> np.ndarray:
    source_linear = np.asarray(source_affine[:3, :3], dtype=np.float64)
    source_spacing = np.asarray(nib.affines.voxel_sizes(source_affine), dtype=np.float64)
    if np.any(~np.isfinite(source_spacing)) or np.any(source_spacing <= 0):
        raise FormalDataError("Source reference has invalid voxel spacing")
    directions = source_linear / source_spacing[np.newaxis, :]
    centre_index = (np.asarray(source_shape[:3], dtype=np.float64) - 1.0) / 2.0
    world_centre = nib.affines.apply_affine(source_affine, centre_index)
    target_linear = directions * np.asarray(target_spacing, dtype=np.float64)[np.newaxis, :]
    target_centre = (np.asarray(target_shape, dtype=np.float64) - 1.0) / 2.0
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = target_linear
    affine[:3, 3] = world_centre - target_linear @ target_centre
    return affine


def _receipt_matches(
    receipt: Mapping[str, Any],
    *,
    expected_digest: str,
    reference_path: Path,
    mask_path: Path,
) -> bool:
    if receipt.get("contract_digest") != expected_digest:
        return False
    if not reference_path.is_file() or not mask_path.is_file():
        return False
    outputs = receipt.get("outputs") or {}
    return (
        outputs.get("reference_sha256") == sha256_file(reference_path)
        and outputs.get("mask_sha256") == sha256_file(mask_path)
    )


def materialize_reference(spec: ReferenceSpec, output_dir: str | Path) -> MaterializedReference:
    """Create or resume an immutable target reference and structural support mask."""

    if not spec.source_reference.is_file() or not spec.source_mask.is_file():
        raise FormalDataError(
            "Bundled source reference and mask must both exist: "
            f"{spec.source_reference}, {spec.source_mask}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_reference = output / "target_reference.nii.gz"
    target_mask = output / "target_mask.nii.gz"
    receipt_path = output / "reference_receipt.json"
    source_hashes = {
        "reference_sha256": sha256_file(spec.source_reference),
        "mask_sha256": sha256_file(spec.source_mask),
    }
    contract_digest = reference_contract_digest(spec)
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if isinstance(receipt, Mapping) and _receipt_matches(
            receipt,
            expected_digest=contract_digest,
            reference_path=target_reference,
            mask_path=target_mask,
        ):
            reference_image = nib.load(target_reference)
            return MaterializedReference(
                target_reference,
                target_mask,
                receipt_path,
                tuple(int(value) for value in reference_image.shape[:3]),
                tuple(float(value) for value in nib.affines.voxel_sizes(reference_image.affine)),
                np.asarray(reference_image.affine, dtype=np.float64),
                contract_digest,
                True,
            )
    source_reference = nib.as_closest_canonical(nib.load(spec.source_reference))
    source_mask = nib.as_closest_canonical(nib.load(spec.source_mask))
    if len(source_reference.shape) != 3 or len(source_mask.shape) != 3:
        raise FormalDataError("Source reference and mask must be three-dimensional")
    target_affine = _target_affine(
        source_reference.affine,
        source_reference.shape,
        spec.target_shape,
        spec.target_spacing,
    )
    target_contract = (spec.target_shape, target_affine)
    reference_resampled = resample_from_to(source_reference, target_contract, order=1, cval=0.0)
    mask_resampled = resample_from_to(source_mask, target_contract, order=0, cval=0.0)
    reference_values = np.asarray(reference_resampled.dataobj, dtype=np.float32)
    mask_values = np.asarray(mask_resampled.dataobj) > 0
    if not np.isfinite(reference_values).all():
        raise FormalDataError("Derived reference contains non-finite voxels")
    if not mask_values.any():
        raise FormalDataError("Derived reference mask is empty")
    reference_image = nib.Nifti1Image(reference_values, target_affine)
    reference_image.header.set_zooms(spec.target_spacing)
    mask_image = nib.Nifti1Image(mask_values.astype(np.uint8), target_affine)
    mask_image.header.set_zooms(spec.target_spacing)
    _atomic_save_nifti(reference_image, target_reference)
    _atomic_save_nifti(mask_image, target_mask)
    receipt = {
        "schema_version": 1,
        "created_at": utc_now(),
        "contract_digest": contract_digest,
        "source": {
            "reference": str(spec.source_reference),
            "mask": str(spec.source_mask),
            **source_hashes,
        },
        "target": {
            "shape": list(spec.target_shape),
            "spacing": list(spec.target_spacing),
            "affine": target_affine.tolist(),
            "world_centre_policy": "preserve_source_world_centre_and_axis_directions",
        },
        "outputs": {
            "reference": str(target_reference),
            "mask": str(target_mask),
            "reference_sha256": sha256_file(target_reference),
            "mask_sha256": sha256_file(target_mask),
        },
    }
    atomic_write_json(receipt_path, receipt)
    return MaterializedReference(
        target_reference,
        target_mask,
        receipt_path,
        spec.target_shape,
        spec.target_spacing,
        target_affine,
        contract_digest,
        False,
    )
