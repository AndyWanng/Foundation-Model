"""Fixed physical-coordinate interpolation between spatial point sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn


def token_centers_from_reference(
    shape: Sequence[int],
    affine: np.ndarray | Tensor,
    *,
    grid_shape: Sequence[int] = (8, 8, 8),
) -> Tensor:
    """Return the world-coordinate centers of regular SAT3D output cells."""

    spatial = tuple(int(value) for value in shape)
    grid = tuple(int(value) for value in grid_shape)
    if len(spatial) != 3 or len(grid) != 3 or any(value < 1 for value in grid):
        raise ValueError("shape and grid_shape must contain three positive integers.")
    if any(size % cells for size, cells in zip(spatial, grid, strict=True)):
        raise ValueError(f"Reference shape {spatial} is not divisible by grid {grid}.")
    transform = np.asarray(affine, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("affine must be a finite 4x4 matrix.")
    axes = []
    for size, cells in zip(spatial, grid, strict=True):
        step = size / cells
        axes.append((np.arange(cells, dtype=np.float64) + 0.5) * step - 0.5)
    mesh = np.meshgrid(*axes, indexing="ij")
    voxel = np.stack(mesh, axis=-1).reshape(-1, 3)
    homogeneous = np.concatenate((voxel, np.ones((len(voxel), 1))), axis=1)
    return torch.from_numpy((homogeneous @ transform.T)[:, :3]).double()


@dataclass(frozen=True)
class InterpolationPlan:
    """Precomputed local interpolation weights from source to target points."""

    source_indices: Tensor
    kernel_weights: Tensor
    source_cell_volume: Tensor
    full_weight_sum: Tensor

    def validate(self) -> None:
        if self.source_indices.ndim != 2:
            raise ValueError("source_indices must be [N,k].")
        if self.kernel_weights.shape != self.source_indices.shape:
            raise ValueError("kernel_weights must match source_indices.")
        if self.source_cell_volume.ndim != 1:
            raise ValueError("source_cell_volume must be [S].")
        if self.full_weight_sum.shape != (self.source_indices.shape[0],):
            raise ValueError("full_weight_sum must be [N].")
        if torch.any(self.source_indices < 0) or torch.any(
            self.source_indices >= self.source_cell_volume.numel()
        ):
            raise ValueError("source_indices references an invalid source point.")
        if torch.any(self.kernel_weights <= 0) or torch.any(self.source_cell_volume <= 0):
            raise ValueError("Kernel weights and source volumes must be positive.")
        if torch.any(self.full_weight_sum <= 0):
            raise ValueError("Every target point must have interpolation support.")

    @classmethod
    def from_coordinates(
        cls,
        source_xyz_mm: Tensor,
        target_xyz_mm: Tensor,
        *,
        neighbors: int = 8,
        sigma_mm: float = 30.0,
        source_cell_volume: Tensor | None = None,
    ) -> "InterpolationPlan":
        source_points = torch.as_tensor(source_xyz_mm, dtype=torch.float64, device="cpu")
        target_points = torch.as_tensor(target_xyz_mm, dtype=torch.float64, device="cpu")
        if (
            source_points.ndim != 2
            or source_points.shape[1] != 3
            or not torch.isfinite(source_points).all()
        ):
            raise ValueError("source_xyz_mm must be finite [S,3].")
        if (
            target_points.ndim != 2
            or target_points.shape[1] != 3
            or not torch.isfinite(target_points).all()
        ):
            raise ValueError("target_xyz_mm must be finite [N,3].")
        if source_points.shape[0] < 1 or target_points.shape[0] < 1:
            raise ValueError("Interpolation requires at least one source and target point.")
        if float(sigma_mm) <= 0:
            raise ValueError("sigma_mm must be positive.")
        count = min(max(1, int(neighbors)), int(source_points.shape[0]))
        distance = torch.cdist(target_points, source_points)
        nearest_distance, source_indices = torch.topk(distance, k=count, largest=False)
        kernel_weights = torch.exp(
            -0.5 * (nearest_distance / float(sigma_mm)).square()
        )
        volume = (
            torch.ones(source_points.shape[0], dtype=torch.float64)
            if source_cell_volume is None
            else torch.as_tensor(source_cell_volume, dtype=torch.float64, device="cpu")
        )
        if volume.shape != (source_points.shape[0],) or torch.any(volume <= 0):
            raise ValueError("source_cell_volume must be positive [S].")
        full_weight_sum = torch.sum(
            kernel_weights * volume[source_indices], dim=-1
        )
        plan = cls(source_indices.long(), kernel_weights, volume, full_weight_sum)
        plan.validate()
        return plan


class FixedSpatialInterpolator(nn.Module):
    """Coverage-aware local interpolation with exact constant preservation."""

    def __init__(self, plan: InterpolationPlan, *, eps: float = 1.0e-8) -> None:
        super().__init__()
        plan.validate()
        self.register_buffer("source_indices", plan.source_indices.long(), persistent=True)
        self.register_buffer("kernel_weights", plan.kernel_weights.float(), persistent=True)
        self.register_buffer("source_cell_volume", plan.source_cell_volume.float(), persistent=True)
        self.register_buffer(
            "full_weight_sum", plan.full_weight_sum.float(), persistent=True
        )
        self.eps = float(eps)

    @property
    def target_count(self) -> int:
        return int(self.source_indices.shape[0])

    @property
    def source_count(self) -> int:
        return int(self.source_cell_volume.numel())

    def forward(
        self,
        source_features: Tensor,
        source_coverage: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if source_features.ndim != 3 or source_features.shape[1] != self.source_count:
            raise ValueError(
                f"source_features must be [B,{self.source_count},D], "
                f"got {tuple(source_features.shape)}."
            )
        batch = source_features.shape[0]
        if source_coverage is None:
            source_coverage = torch.ones(
                batch,
                self.source_count,
                device=source_features.device,
                dtype=source_features.dtype,
            )
        if source_coverage.shape != (batch, self.source_count):
            raise ValueError("source_coverage must be [B,S].")
        if (
            not torch.isfinite(source_coverage).all()
            or torch.any(source_coverage < 0)
            or torch.any(source_coverage > 1)
        ):
            raise ValueError("source_coverage must be finite in [0,1].")
        indices = self.source_indices.to(source_features.device)
        gathered = source_features[:, indices, :]
        # Interpolation support must not depend on the encoder autocast dtype.
        # Compute fixed weights in FP32, then cast the interpolated features back.
        coverage = source_coverage[:, indices].float()
        volume = self.source_cell_volume.to(source_features.device, torch.float32)[indices]
        kernel = self.kernel_weights.to(source_features.device, torch.float32)
        weights = kernel.unsqueeze(0) * volume.unsqueeze(0) * coverage
        available_weight_sum = weights.sum(dim=-1)
        normalized_weights = weights / available_weight_sum.clamp_min(
            self.eps
        ).unsqueeze(-1)
        gathered_work = (
            gathered.double()
            if source_features.dtype == torch.float64
            else gathered.float()
        )
        target_features = torch.sum(
            normalized_weights.to(gathered_work.dtype).unsqueeze(-1) * gathered_work,
            dim=2,
        ).to(source_features.dtype)
        supported = available_weight_sum > self.eps
        target_features = torch.where(
            supported.unsqueeze(-1),
            target_features,
            torch.zeros_like(target_features),
        )
        full_weight_sum = self.full_weight_sum.to(
            source_features.device, torch.float32
        )
        target_coverage = torch.clamp(
            available_weight_sum / full_weight_sum.clamp_min(self.eps),
            min=0.0,
            max=1.0,
        )
        target_coverage = torch.where(
            supported, target_coverage, torch.zeros_like(target_coverage)
        )
        if (
            not torch.isfinite(target_features).all()
            or not torch.isfinite(target_coverage).all()
        ):
            raise FloatingPointError("Interpolation produced non-finite values.")
        return target_features, target_coverage


def build_interpolator(
    source_xyz_mm: Tensor,
    target_xyz_mm: Tensor,
    **kwargs: object,
) -> FixedSpatialInterpolator:
    return FixedSpatialInterpolator(
        InterpolationPlan.from_coordinates(
            source_xyz_mm, target_xyz_mm, **kwargs  # type: ignore[arg-type]
        )
    )


__all__ = [
    "FixedSpatialInterpolator",
    "InterpolationPlan",
    "build_interpolator",
    "token_centers_from_reference",
]
