"""Scalable, non-interactive QC for model-space MRI/PET observations.

The checks in this module deliberately separate two validation levels:

* required data and spatial checks are deterministic and may reject an invalid item;
* cross-modal image statistics are uncalibrated diagnostics and only raise flags.

The latter must not be represented as proof that subtle misregistration is absent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def _single_channel(image: nib.spatialimages.SpatialImage, *, name: str) -> np.ndarray:
    array = np.asarray(image.dataobj, dtype=np.float32)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"{name} must be a static 3-D volume, got {array.shape}")
    return array


def _world_centroid(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    total = float(mask.sum())
    if total <= 0.0:
        raise ValueError("Cannot calculate a centroid for an empty support")
    coordinates = []
    for axis, size in enumerate(mask.shape):
        reduce_axes = tuple(index for index in range(3) if index != axis)
        marginal = mask.sum(axis=reduce_axes, dtype=np.float64)
        coordinates.append(
            float(np.dot(np.arange(size, dtype=np.float64), marginal) / total)
        )
    voxel = np.asarray(coordinates, dtype=np.float64)
    return np.asarray(nib.affines.apply_affine(affine, voxel), dtype=np.float64)


def _zscore(array: np.ndarray, support: np.ndarray) -> np.ndarray:
    values = np.asarray(array[support], dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    normalized = np.zeros(array.shape, dtype=np.float32)
    normalized[support] = ((values - mean) / max(std, 1.0e-12)).astype(np.float32)
    return normalized


def _nmi(first: np.ndarray, second: np.ndarray, support: np.ndarray, *, bins: int) -> float | None:
    if int(support.sum()) < max(16, bins):
        return None
    x = np.asarray(first[support], dtype=np.float64)
    y = np.asarray(second[support], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if float(x.std()) <= 1.0e-12 or float(y.std()) <= 1.0e-12:
        return None
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    total = float(joint.sum())
    if total <= 0.0:
        return None
    probability = joint / total
    px = probability.sum(axis=1)
    py = probability.sum(axis=0)

    def entropy(values: np.ndarray) -> float:
        nonzero = values[values > 0.0]
        return float(-(nonzero * np.log(nonzero)).sum())

    h_x = entropy(px)
    h_y = entropy(py)
    h_xy = entropy(probability)
    if h_xy <= 1.0e-12:
        return None
    return float((h_x + h_y) / h_xy)


def _correlation(first: np.ndarray, second: np.ndarray, support: np.ndarray) -> float | None:
    if int(support.sum()) < 16:
        return None
    x = np.asarray(first[support], dtype=np.float64)
    y = np.asarray(second[support], dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1.0e-12:
        return None
    return float(np.dot(x, y) / denominator)


def _shift_without_wrap(array: np.ndarray, shift: int, axis: int) -> np.ndarray:
    shifted = np.zeros_like(array)
    source = [slice(None)] * array.ndim
    target = [slice(None)] * array.ndim
    if shift > 0:
        source[axis] = slice(0, -shift)
        target[axis] = slice(shift, None)
    elif shift < 0:
        source[axis] = slice(-shift, None)
        target[axis] = slice(0, shift)
    else:
        return array.copy()
    shifted[tuple(target)] = array[tuple(source)]
    return shifted


def _downsample(
    array: np.ndarray, support: np.ndarray, *, maximum_axis: int
) -> tuple[np.ndarray, np.ndarray]:
    strides = tuple(
        max(1, int(math.ceil(size / max(1, maximum_axis)))) for size in array.shape
    )
    selection = tuple(slice(None, None, stride) for stride in strides)
    return array[selection], support[selection]


def evaluate_registered_pair(
    *,
    mri_volume_path: Path,
    mri_mask_path: Path,
    pet_volume_path: Path,
    pet_mask_path: Path,
    affine_tolerance: float,
    minimum_support_voxels: int,
    diagnostic_maximum_axis: int,
    histogram_bins: int,
    mask_dice_flag_below: float,
    centroid_distance_mm_flag_above: float,
    nmi_shift_margin_flag_below: float,
) -> dict[str, Any]:
    """Evaluate one registered pair without requiring any human action."""

    input_validation_failures: list[str] = []
    statistical_flags: list[str] = []
    checks: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    loaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for modality, volume_path, mask_path in (
        ("mri", Path(mri_volume_path), Path(mri_mask_path)),
        ("pet", Path(pet_volume_path), Path(pet_mask_path)),
    ):
        try:
            image = nib.load(str(volume_path), mmap=True)
            mask_image = nib.load(str(mask_path), mmap=True)
            array = _single_channel(image, name=f"{modality} volume")
            mask_array = _single_channel(mask_image, name=f"{modality} mask")
        except Exception as error:
            input_validation_failures.append(f"{modality}_load_error:{type(error).__name__}:{error}")
            continue

        support = mask_array > 0.5
        support_voxels = int(support.sum())
        shape_matches_mask = array.shape == mask_array.shape
        affine_matches_mask = bool(
            np.allclose(
                np.asarray(image.affine),
                np.asarray(mask_image.affine),
                atol=affine_tolerance,
                rtol=0.0,
            )
        )
        mask_is_finite = bool(np.isfinite(mask_array).all())
        supported_values_are_finite = bool(
            shape_matches_mask
            and support_voxels > 0
            and np.isfinite(array[support]).all()
        )
        supported_std = (
            float(np.asarray(array[support], dtype=np.float64).std(ddof=0))
            if supported_values_are_finite
            else None
        )
        determinant = float(np.linalg.det(np.asarray(image.affine)[:3, :3]))
        valid_grid = bool(math.isfinite(determinant) and abs(determinant) > 1.0e-12)
        checks[modality] = {
            "shape": list(array.shape),
            "shape_matches_mask": shape_matches_mask,
            "affine_matches_mask": affine_matches_mask,
            "mask_is_finite": mask_is_finite,
            "support_voxels": support_voxels,
            "supported_values_are_finite": supported_values_are_finite,
            "supported_intensity_std": supported_std,
            "valid_physical_grid": valid_grid,
            "outside_support_is_zero": bool(
                shape_matches_mask
                and np.isfinite(array[~support]).all()
                and np.count_nonzero(array[~support]) == 0
            ),
        }
        if not shape_matches_mask:
            input_validation_failures.append(f"{modality}_volume_mask_shape_mismatch")
        if not affine_matches_mask:
            input_validation_failures.append(f"{modality}_volume_mask_affine_mismatch")
        if not mask_is_finite:
            input_validation_failures.append(f"{modality}_mask_nonfinite")
        if support_voxels < minimum_support_voxels:
            input_validation_failures.append(f"{modality}_support_too_small")
        if not supported_values_are_finite:
            input_validation_failures.append(f"{modality}_supported_values_nonfinite")
        if supported_std is None or supported_std <= 1.0e-6:
            input_validation_failures.append(f"{modality}_supported_intensity_degenerate")
        if not valid_grid:
            input_validation_failures.append(f"{modality}_invalid_physical_grid")
        if not input_validation_failures or not any(item.startswith(f"{modality}_") for item in input_validation_failures):
            loaded[modality] = (
                array,
                support,
                np.asarray(image.affine, dtype=np.float64),
            )

    if set(loaded) == {"mri", "pet"}:
        mri, mri_support, mri_affine = loaded["mri"]
        pet, pet_support, pet_affine = loaded["pet"]
        paired_shape_equal = mri.shape == pet.shape
        paired_affine_equal = bool(
            np.allclose(mri_affine, pet_affine, atol=affine_tolerance, rtol=0.0)
        )
        checks["paired_shape_equal"] = paired_shape_equal
        checks["paired_affine_equal"] = paired_affine_equal
        if not paired_shape_equal:
            input_validation_failures.append("paired_volume_shape_mismatch")
        if not paired_affine_equal:
            input_validation_failures.append("paired_volume_affine_mismatch")

        if paired_shape_equal and paired_affine_equal:
            intersection = mri_support & pet_support
            intersection_voxels = int(intersection.sum())
            support_sum = int(mri_support.sum()) + int(pet_support.sum())
            mask_dice = (
                float(2.0 * intersection_voxels / support_sum)
                if support_sum > 0
                else 0.0
            )
            centroid_distance = float(
                np.linalg.norm(
                    _world_centroid(mri_support, mri_affine)
                    - _world_centroid(pet_support, pet_affine)
                )
            )
            diagnostics.update(
                {
                    "support_intersection_voxels": intersection_voxels,
                    "mask_dice": mask_dice,
                    "support_centroid_distance_mm": centroid_distance,
                }
            )
            if intersection_voxels < minimum_support_voxels:
                input_validation_failures.append("paired_support_intersection_too_small")
            else:
                mri_ds, mri_support_ds = _downsample(
                    mri, mri_support, maximum_axis=diagnostic_maximum_axis
                )
                pet_ds, pet_support_ds = _downsample(
                    pet, pet_support, maximum_axis=diagnostic_maximum_axis
                )
                common_ds = mri_support_ds & pet_support_ds
                mri_z = _zscore(mri_ds, mri_support_ds)
                pet_z = _zscore(pet_ds, pet_support_ds)
                observed_nmi = _nmi(mri_z, pet_z, common_ds, bins=histogram_bins)
                smooth_mri = gaussian_filter(mri_z, sigma=0.75)
                smooth_pet = gaussian_filter(pet_z, sigma=0.75)
                mri_gradient = np.sqrt(
                    sum(component * component for component in np.gradient(smooth_mri))
                )
                pet_gradient = np.sqrt(
                    sum(component * component for component in np.gradient(smooth_pet))
                )
                gradient_correlation = _correlation(
                    mri_gradient, pet_gradient, common_ds
                )
                null_nmi: list[float] = []
                for axis, size in enumerate(pet_z.shape):
                    shift = max(2, size // 4)
                    shifted_pet = _shift_without_wrap(pet_z, shift, axis)
                    shifted_support = _shift_without_wrap(
                        pet_support_ds.astype(np.uint8), shift, axis
                    ).astype(bool)
                    value = _nmi(
                        mri_z,
                        shifted_pet,
                        mri_support_ds & shifted_support,
                        bins=histogram_bins,
                    )
                    if value is not None:
                        null_nmi.append(value)
                maximum_shifted_nmi = max(null_nmi) if null_nmi else None
                nmi_shift_margin = (
                    float(observed_nmi - maximum_shifted_nmi)
                    if observed_nmi is not None and maximum_shifted_nmi is not None
                    else None
                )
                diagnostics.update(
                    {
                        "diagnostic_grid_shape": list(mri_ds.shape),
                        "normalized_mutual_information": observed_nmi,
                        "maximum_large_shift_nmi": maximum_shifted_nmi,
                        "nmi_large_shift_margin": nmi_shift_margin,
                        "gradient_magnitude_correlation": gradient_correlation,
                    }
                )

            if mask_dice < mask_dice_flag_below:
                statistical_flags.append("low_mask_dice")
            if centroid_distance > centroid_distance_mm_flag_above:
                statistical_flags.append("large_support_centroid_distance")
            margin = diagnostics.get("nmi_large_shift_margin")
            if margin is None:
                statistical_flags.append("nmi_shift_margin_unavailable")
            elif float(margin) < nmi_shift_margin_flag_below:
                statistical_flags.append("nonpositive_nmi_shift_margin")

    return {
        "input_validation_passed": not input_validation_failures,
        "input_validation_failures": input_validation_failures,
        "statistical_flags": statistical_flags,
        "checks": checks,
        "diagnostics": diagnostics,
    }
