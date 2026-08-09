"""Training-split alignment to a fixed latent reference.

Separately trained self-supervised models may learn different latent coordinate
systems.  A similarity transform removes only a global coordinate difference --
translation, one positive scale and an orthogonal rotation/reflection -- before
predictions are scored against one fixed reference.  It deliberately cannot learn
an unconstrained linear decoder or restore rank that is absent from the source.

This module contains no model, data-loader or split-selection logic.  Callers
must pass training rows to :meth:`WeightedSimilarityTransform.fit` and must
apply the resulting immutable transform to held-out rows themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import torch
from torch import Tensor


def _as_finite_matrix(value: Tensor, *, name: str) -> Tensor:
    matrix = torch.as_tensor(value).detach()
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"{name} must be a non-empty [rows,channels] matrix")
    if not matrix.is_floating_point():
        matrix = matrix.float()
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix.to(dtype=torch.float64)


def _as_finite_weight(value: Tensor, *, rows: int) -> Tensor:
    weight = torch.as_tensor(value).detach()
    if weight.shape != (int(rows),):
        raise ValueError(f"weight must have shape [{rows}]")
    if not weight.is_floating_point():
        weight = weight.float()
    if not torch.isfinite(weight).all() or torch.any(weight < 0):
        raise ValueError("weight must be finite and non-negative")
    positive = int(torch.count_nonzero(weight > 0))
    if positive < 2 or float(weight.sum()) <= 0:
        raise ValueError("alignment requires at least two positive-weight rows")
    return weight.to(dtype=torch.float64)


def _entropy_rank(weighted_centered: Tensor, *, epsilon: float) -> float:
    singular_energy = torch.linalg.svdvals(weighted_centered).square()
    total = singular_energy.sum()
    if float(total) <= float(epsilon):
        return 0.0
    probability = singular_energy / total
    entropy = -(probability * probability.clamp_min(float(epsilon)).log()).sum()
    return float(torch.exp(entropy).cpu())


def subject_balanced_weights(
    raw_weight: Tensor,
    subject_ids: Sequence[Hashable],
    group_ids: Sequence[Hashable] | None = None,
) -> Tensor:
    """Normalize token reliability without subject/view pseudoreplication.

    The returned weights sum to one.  Every subject receives equal total mass;
    every group within a subject receives equal mass; raw reliability controls
    only the distribution within one group.  ``group_ids`` should identify a
    deterministic view/relation instance.  If omitted, each subject is one
    group.
    """

    weight = torch.as_tensor(raw_weight).detach()
    if weight.ndim != 1:
        raise ValueError("raw_weight must be one-dimensional")
    rows = int(weight.numel())
    if len(subject_ids) != rows:
        raise ValueError("subject_ids length must match raw_weight")
    if group_ids is not None and len(group_ids) != rows:
        raise ValueError("group_ids length must match raw_weight")
    if rows < 2:
        raise ValueError("at least two rows are required")
    if not weight.is_floating_point():
        weight = weight.float()
    if not torch.isfinite(weight).all() or torch.any(weight < 0):
        raise ValueError("raw_weight must be finite and non-negative")

    subjects = tuple(subject_ids)
    if any(subject is None for subject in subjects):
        raise ValueError("subject_ids cannot contain None")
    unique_subjects = tuple(dict.fromkeys(subjects))
    if not unique_subjects:
        raise ValueError("at least one subject is required")
    groups = tuple(group_ids) if group_ids is not None else subjects
    if any(group is None for group in groups):
        raise ValueError("group_ids cannot contain None")

    result = torch.zeros_like(weight, dtype=torch.float64)
    subject_share = 1.0 / float(len(unique_subjects))
    for subject in unique_subjects:
        subject_rows = [index for index, value in enumerate(subjects) if value == subject]
        subject_groups = tuple(dict.fromkeys(groups[index] for index in subject_rows))
        if not subject_groups:
            raise RuntimeError("subject grouping produced no rows")
        group_share = subject_share / float(len(subject_groups))
        for group in subject_groups:
            indices = [
                index
                for index in subject_rows
                if groups[index] == group
            ]
            index_tensor = torch.tensor(indices, device=weight.device, dtype=torch.long)
            selected = weight[index_tensor].to(dtype=torch.float64)
            total = selected.sum()
            if float(total) <= 0:
                raise ValueError(
                    f"subject/group has zero positive reliability: {subject!r}/{group!r}"
                )
            result[index_tensor] = group_share * selected / total

    if not torch.isfinite(result).all() or torch.any(result < 0):
        raise RuntimeError("subject-balanced weight construction became invalid")
    if not torch.allclose(
        result.sum(),
        torch.ones((), device=result.device, dtype=result.dtype),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("subject-balanced weights do not sum to one")
    return result


@dataclass(frozen=True)
class WeightedSimilarityTransform:
    """Immutable positive-scale orthogonal alignment fitted on training rows."""

    source_mean: Tensor
    target_mean: Tensor
    rotation: Tensor
    scale: float
    fit_row_count: int
    fit_positive_weight_rows: int
    source_centered_energy: float
    target_centered_energy: float
    source_entropy_effective_rank: float
    target_entropy_effective_rank: float
    orthogonality_max_abs_error: float
    weighted_fit_rmse: float

    @classmethod
    def fit(
        cls,
        source: Tensor,
        target: Tensor,
        weight: Tensor,
        *,
        absolute_energy_floor: float = 1.0e-12,
        source_to_target_energy_floor: float = 1.0e-8,
    ) -> "WeightedSimilarityTransform":
        """Fit ``scale * (source - mean) @ rotation + target_mean``.

        The transform is the closed-form weighted similarity-Procrustes
        solution.  A collapsed source or target fails closed; no epsilon clamp
        is allowed to turn numerical noise into a high-gain representation.
        """

        if float(absolute_energy_floor) <= 0:
            raise ValueError("absolute_energy_floor must be positive")
        if not 0.0 < float(source_to_target_energy_floor) < 1.0:
            raise ValueError("source_to_target_energy_floor must lie in (0,1)")
        left = _as_finite_matrix(source, name="source")
        right = _as_finite_matrix(target, name="target")
        if left.shape != right.shape:
            raise ValueError("source and target must have identical shapes")
        if left.device != right.device:
            raise ValueError("source and target must use the same device")
        row_weight = _as_finite_weight(weight, rows=left.shape[0]).to(left.device)
        probability = row_weight / row_weight.sum()
        left_mean = torch.sum(probability[:, None] * left, dim=0)
        right_mean = torch.sum(probability[:, None] * right, dim=0)
        left_centered = left - left_mean
        right_centered = right - right_mean
        root_weight = probability.sqrt()[:, None]
        weighted_left = root_weight * left_centered
        weighted_right = root_weight * right_centered
        left_energy = weighted_left.square().sum()
        right_energy = weighted_right.square().sum()
        absolute_floor = float(absolute_energy_floor)
        if float(right_energy) <= absolute_floor:
            raise ValueError("target reference is collapsed in centered energy")
        relative_floor = float(source_to_target_energy_floor) * float(right_energy)
        if float(left_energy) <= max(absolute_floor, relative_floor):
            raise ValueError(
                "source representation is collapsed relative to the target reference"
            )

        cross_covariance = weighted_left.T @ weighted_right
        u, singular, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
        rotation = u @ vh
        scale = singular.sum() / left_energy
        if not torch.isfinite(scale) or float(scale) <= 0:
            raise ValueError("source and target have no finite positive aligned scale")
        identity = torch.eye(rotation.shape[0], device=rotation.device, dtype=rotation.dtype)
        orthogonality_error = (rotation.T @ rotation - identity).abs().max()
        aligned = float(scale) * (left_centered @ rotation) + right_mean
        fit_rmse = torch.sqrt(
            torch.sum(probability * (aligned - right).square().mean(dim=-1))
        )
        values = (
            rotation,
            orthogonality_error,
            fit_rmse,
        )
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("similarity-Procrustes fit produced non-finite values")

        return cls(
            source_mean=left_mean.detach().cpu(),
            target_mean=right_mean.detach().cpu(),
            rotation=rotation.detach().cpu(),
            scale=float(scale.detach().cpu()),
            fit_row_count=int(left.shape[0]),
            fit_positive_weight_rows=int(torch.count_nonzero(row_weight > 0)),
            source_centered_energy=float(left_energy.detach().cpu()),
            target_centered_energy=float(right_energy.detach().cpu()),
            source_entropy_effective_rank=_entropy_rank(
                weighted_left, epsilon=absolute_floor
            ),
            target_entropy_effective_rank=_entropy_rank(
                weighted_right, epsilon=absolute_floor
            ),
            orthogonality_max_abs_error=float(orthogonality_error.detach().cpu()),
            weighted_fit_rmse=float(fit_rmse.detach().cpu()),
        )

    @property
    def dimension(self) -> int:
        return int(self.rotation.shape[0])

    def apply(self, source: Tensor) -> Tensor:
        """Apply the frozen transform without changing the input shape/dtype."""

        value = torch.as_tensor(source)
        if value.ndim < 2 or value.shape[-1] != self.dimension:
            raise ValueError(
                f"source must end in the fitted channel dimension {self.dimension}"
            )
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("source must be a finite floating-point tensor")
        original_dtype = value.dtype
        compute = value.to(dtype=torch.float64)
        mean = self.source_mean.to(device=value.device, dtype=torch.float64)
        target_mean = self.target_mean.to(device=value.device, dtype=torch.float64)
        rotation = self.rotation.to(device=value.device, dtype=torch.float64)
        aligned = float(self.scale) * ((compute - mean) @ rotation) + target_mean
        if not torch.isfinite(aligned).all():
            raise FloatingPointError("similarity transform produced non-finite values")
        return aligned.to(dtype=original_dtype)

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "method": "global_subject_weighted_similarity_procrustes",
            "dimension": self.dimension,
            "fit_row_count": self.fit_row_count,
            "fit_positive_weight_rows": self.fit_positive_weight_rows,
            "scale": self.scale,
            "source_centered_energy": self.source_centered_energy,
            "target_centered_energy": self.target_centered_energy,
            "source_to_target_energy_ratio": (
                self.source_centered_energy / self.target_centered_energy
            ),
            "source_entropy_effective_rank": self.source_entropy_effective_rank,
            "target_entropy_effective_rank": self.target_entropy_effective_rank,
            "orthogonality_max_abs_error": self.orthogonality_max_abs_error,
            "weighted_fit_rmse": self.weighted_fit_rmse,
        }

    def state_dict(self) -> dict[str, object]:
        """Return a CPU-only serialization payload with explicit schema."""

        return {
            "schema_version": 1,
            "source_mean": self.source_mean.detach().cpu().clone(),
            "target_mean": self.target_mean.detach().cpu().clone(),
            "rotation": self.rotation.detach().cpu().clone(),
            "scale": self.scale,
            "fit_row_count": self.fit_row_count,
            "fit_positive_weight_rows": self.fit_positive_weight_rows,
            "source_centered_energy": self.source_centered_energy,
            "target_centered_energy": self.target_centered_energy,
            "source_entropy_effective_rank": self.source_entropy_effective_rank,
            "target_entropy_effective_rank": self.target_entropy_effective_rank,
            "orthogonality_max_abs_error": self.orthogonality_max_abs_error,
            "weighted_fit_rmse": self.weighted_fit_rmse,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "WeightedSimilarityTransform":
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("unsupported similarity-transform schema")
        source_mean = _as_finite_matrix(
            torch.as_tensor(payload["source_mean"]).reshape(1, -1),
            name="source_mean",
        ).squeeze(0)
        target_mean = _as_finite_matrix(
            torch.as_tensor(payload["target_mean"]).reshape(1, -1),
            name="target_mean",
        ).squeeze(0)
        rotation = _as_finite_matrix(
            torch.as_tensor(payload["rotation"]), name="rotation"
        )
        if rotation.shape[0] != rotation.shape[1]:
            raise ValueError("rotation must be square")
        if source_mean.shape != target_mean.shape or source_mean.numel() != rotation.shape[0]:
            raise ValueError("serialized transform dimensions disagree")
        transform = cls(
            source_mean=source_mean.cpu(),
            target_mean=target_mean.cpu(),
            rotation=rotation.cpu(),
            scale=float(payload["scale"]),
            fit_row_count=int(payload["fit_row_count"]),
            fit_positive_weight_rows=int(payload["fit_positive_weight_rows"]),
            source_centered_energy=float(payload["source_centered_energy"]),
            target_centered_energy=float(payload["target_centered_energy"]),
            source_entropy_effective_rank=float(payload["source_entropy_effective_rank"]),
            target_entropy_effective_rank=float(payload["target_entropy_effective_rank"]),
            orthogonality_max_abs_error=float(payload["orthogonality_max_abs_error"]),
            weighted_fit_rmse=float(payload["weighted_fit_rmse"]),
        )
        scalars = (
            transform.scale,
            transform.source_centered_energy,
            transform.target_centered_energy,
            transform.source_entropy_effective_rank,
            transform.target_entropy_effective_rank,
            transform.orthogonality_max_abs_error,
            transform.weighted_fit_rmse,
        )
        if (
            transform.scale <= 0
            or transform.fit_row_count < 2
            or transform.fit_positive_weight_rows < 2
            or not all(torch.isfinite(torch.tensor(value)) for value in scalars)
        ):
            raise ValueError("serialized similarity transform is invalid")
        identity = torch.eye(transform.dimension, dtype=torch.float64)
        error = (transform.rotation.T @ transform.rotation - identity).abs().max()
        if float(error) > 1.0e-7:
            raise ValueError("serialized rotation is not orthogonal")
        return transform


__all__ = ["WeightedSimilarityTransform", "subject_balanced_weights"]
