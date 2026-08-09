"""Single masked-Huber JEPA objective and representation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class JEPAObjectiveResult:
    loss: Tensor

    mean_selected_token_count: Tensor
    mean_absolute_error: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "loss": self.loss,

            "mean_selected_token_count": self.mean_selected_token_count,
            "mean_absolute_error": self.mean_absolute_error,
        }


def masked_huber_objective(
    prediction: Tensor,
    target: Tensor,
    query_mask: Tensor,
    *,
    query_weight: Tensor | None = None,
    beta: float = 1.0,
) -> JEPAObjectiveResult:
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical [B,T,D] shapes")
    if float(beta) <= 0:
        raise ValueError("Huber beta must be positive")
    truth = target.detach()
    mask = torch.as_tensor(query_mask, device=prediction.device, dtype=torch.bool)
    if mask.shape != prediction.shape[:2] or torch.any(mask.sum(dim=1) < 1):
        raise ValueError("query_mask must be [B,T] with at least one token per item")
    if query_weight is None:
        weight = mask.to(prediction.dtype)
    else:
        confidence = torch.as_tensor(
            query_weight, device=prediction.device, dtype=prediction.dtype
        )
        if confidence.shape != mask.shape or torch.any(confidence < 0):
            raise ValueError("query_weight must be non-negative [B,T]")
        weight = confidence * mask
    if torch.any(weight.sum(dim=1) <= 0):
        raise ValueError("Every item needs positive total query weight")
    element_loss = F.smooth_l1_loss(
        prediction.float(), truth.float(), beta=float(beta), reduction="none"
    ).mean(dim=-1)
    token_absolute_error = (prediction.float() - truth.float()).abs().mean(dim=-1)
    # Equal example weighting regardless of the number of selected tokens.
    example_loss = (element_loss * weight).sum(dim=1) / weight.sum(dim=1)
    example_absolute_error = (token_absolute_error * weight).sum(dim=1) / weight.sum(dim=1)
    loss = example_loss.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("JEPA objective is non-finite")
    return JEPAObjectiveResult(
        loss=loss,

        mean_selected_token_count=mask.sum(dim=1).float().mean(),
        mean_absolute_error=example_absolute_error.mean(),
    )


class MaskedJEPAObjective(nn.Module):
    def __init__(
        self,
        *,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = float(beta)

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        query_mask: Tensor,
        query_weight: Tensor | None = None,
    ) -> JEPAObjectiveResult:
        return masked_huber_objective(
            prediction,
            target,
            query_mask,
            query_weight=query_weight,
            beta=self.beta,
        )


def _entropy_rank_from_centered(value: Tensor) -> Tensor:
    singular_energy = torch.linalg.svdvals(value.float()).square()
    total = singular_energy.sum()
    if float(total) <= 1.0e-12:
        return torch.zeros((), device=value.device)
    probability = singular_energy / total
    return torch.exp(-(probability * probability.clamp_min(1.0e-12).log()).sum())


def _effective_rank(value: Tensor) -> Tensor:
    centered = value.float() - value.float().mean(dim=0, keepdim=True)
    return _entropy_rank_from_centered(centered)


def entropy_effective_rank(value: Tensor) -> Tensor:
    """Entropy effective rank after flattening all non-channel axes."""

    if value.ndim < 2:
        raise ValueError("effective rank requires at least [tokens,channels]")
    return _effective_rank(value.reshape(-1, value.shape[-1]))


@torch.no_grad()
def field_representation_diagnostics(
    value: Tensor,
    *,
    point_weight: Tensor | None = None,
    variance_epsilon: float = 1.0e-6,
) -> dict[str, float]:
    """Describe one field without conflating a spatially constant offset with shape.

    ``value`` is ``[B,N,D]``.  ``point_weight`` may be ``[N]`` or ``[B,N]``
    and is normalized independently for every item.  This makes the same
    implementation usable for FEM-mass-weighted node fields and
    reliability-weighted query tokens.  Entropy rank is computed from the
    centered, square-root-weighted field, so it uses the same measure as the
    reported centered variance and spatial-mean energy fraction.
    """

    if value.ndim != 3 or not torch.isfinite(value).all():
        raise ValueError("value must be a finite [B,N,D] tensor")
    batch, points = value.shape[:2]
    if point_weight is None:
        weight = torch.ones(batch, points, device=value.device, dtype=torch.float32)
    else:
        weight = torch.as_tensor(point_weight, device=value.device, dtype=torch.float32)
        if weight.ndim == 1:
            if weight.shape != (points,):
                raise ValueError("one-dimensional point_weight must have shape [N]")
            weight = weight.unsqueeze(0).expand(batch, -1)
        if (
            weight.shape != (batch, points)
            or not torch.isfinite(weight).all()
            or torch.any(weight < 0)
        ):
            raise ValueError("point_weight must be finite non-negative [N] or [B,N]")
    if torch.any(weight.sum(dim=1) <= 0):
        raise ValueError("Every field item needs positive total point weight")

    rows: list[dict[str, Tensor]] = []
    for item in range(batch):
        current = value[item].float()
        probability = weight[item] / weight[item].sum()
        mean = torch.sum(probability[:, None] * current, dim=0, keepdim=True)
        centered = current - mean
        total_energy = torch.sum(probability * current.square().mean(dim=-1))
        centered_variance = torch.sum(
            probability * centered.square().mean(dim=-1)
        )
        spatial_mean_energy = mean.square().mean()
        rows.append(
            {
                "rms": total_energy.sqrt(),
                "centered_variance": centered_variance,
                "spatial_mean_energy_fraction": spatial_mean_energy
                / total_energy.clamp_min(float(variance_epsilon)),
                "entropy_effective_rank": _entropy_rank_from_centered(
                    probability.sqrt()[:, None] * centered
                ),
                "positive_weight_fraction": (weight[item] > 0).float().mean(),
            }
        )
    return {
        key: float(torch.stack([row[key] for row in rows]).mean().cpu())
        for key in rows[0]
    }


@torch.no_grad()
def representation_diagnostics(
    prediction: Tensor,
    target: Tensor,
    query_mask: Tensor,
    *,
    query_weight: Tensor | None = None,
    variance_epsilon: float = 1.0e-6,
) -> dict[str, float]:
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical [B,T,D] shapes")
    mask = torch.as_tensor(query_mask, device=prediction.device, dtype=torch.bool)
    if mask.shape != prediction.shape[:2] or torch.any(mask.sum(dim=1) < 2):
        raise ValueError("Diagnostics need at least two selected tokens per item")
    if query_weight is None:
        diagnostic_weight = mask.to(torch.float32)
    else:
        confidence = torch.as_tensor(
            query_weight, device=prediction.device, dtype=torch.float32
        )
        if (
            confidence.shape != mask.shape
            or not torch.isfinite(confidence).all()
            or torch.any(confidence < 0)
        ):
            raise ValueError("query_weight must be finite non-negative [B,T]")
        diagnostic_weight = confidence * mask
    if torch.any(diagnostic_weight.sum(dim=1) <= 0):
        raise ValueError("Diagnostics need positive selected query weight")
    rows: list[dict[str, Tensor]] = []
    for item in range(prediction.shape[0]):
        left = prediction[item, mask[item]].float()
        right = target[item, mask[item]].detach().float()
        item_weight = diagnostic_weight[item, mask[item]]
        probability = item_weight / item_weight.sum()
        left_mean = torch.sum(probability[:, None] * left, dim=0, keepdim=True)
        right_mean = torch.sum(probability[:, None] * right, dim=0, keepdim=True)
        left_centered = left - left_mean
        right_centered = right - right_mean
        left_variance = torch.sum(
            probability * left_centered.square().mean(dim=-1)
        )
        right_variance = torch.sum(
            probability * right_centered.square().mean(dim=-1)
        )
        left_energy = torch.sum(
            probability * left.square().mean(dim=-1)
        )
        right_energy = torch.sum(
            probability * right.square().mean(dim=-1)
        )
        left_total_energy = left_energy.clamp_min(variance_epsilon)
        right_total_energy = right_energy.clamp_min(variance_epsilon)
        left_rms = left_energy.sqrt()
        right_rms = right_energy.sqrt()
        left_spatial_mean_energy = left_mean.square().mean()
        right_spatial_mean_energy = right_mean.square().mean()
        cosine = F.cosine_similarity(left, right, dim=-1, eps=1.0e-6)
        rank_weight = probability.sqrt()[:, None]
        left_rank = _entropy_rank_from_centered(rank_weight * left_centered)
        right_rank = _entropy_rank_from_centered(rank_weight * right_centered)
        rows.append(
            {
                "rmse": torch.sum(
                    probability * (left - right).square().mean(dim=-1)
                ).sqrt(),
                "mae": torch.sum(
                    probability * (left - right).abs().mean(dim=-1)
                ),
                "cosine": torch.sum(probability * cosine),
                "prediction_variance": left_variance,
                "target_variance": right_variance,
                "variance_ratio": left_variance / right_variance.clamp_min(variance_epsilon),
                "centered_variance_recovery_ratio": left_variance
                / right_variance.clamp_min(variance_epsilon),
                "prediction_spatial_mean_energy_fraction": left_spatial_mean_energy / left_total_energy,
                "target_spatial_mean_energy_fraction": right_spatial_mean_energy / right_total_energy,
                "prediction_rms": left_rms,
                "target_rms": right_rms,
                "amplitude_ratio": left_rms / right_rms.clamp_min(variance_epsilon),
                "prediction_effective_rank": left_rank,
                "target_effective_rank": right_rank,
                "effective_rank_ratio": left_rank
                / right_rank.clamp_min(variance_epsilon),
                "near_zero_channel_fraction": (
                    torch.sum(
                        probability[:, None] * left_centered.square(), dim=0
                    )
                    < variance_epsilon
                ).float().mean(),
            }
        )
    return {
        key: float(torch.stack([row[key] for row in rows]).mean().cpu())
        for key in rows[0]
    }


__all__ = [
    "JEPAObjectiveResult",
    "MaskedJEPAObjective",
    "entropy_effective_rank",
    "field_representation_diagnostics",
    "masked_huber_objective",
    "representation_diagnostics",
]
