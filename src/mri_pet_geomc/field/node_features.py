"""Permutation-invariant aggregation of observations on shared FEM nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from ..geometry.interpolation import FixedSpatialInterpolator


@dataclass(frozen=True)
class ObservationTokenBatch:
    """Encoded tokens from one observation before token-to-node interpolation.

    Interpolation weights depend only on physical source and target coordinates.
    """

    features: Tensor
    interpolation: FixedSpatialInterpolator
    coverage: Tensor | None = None
    reliability: Tensor | None = None

    def validate(self, *, feature_dim: int) -> None:
        if not isinstance(self.interpolation, FixedSpatialInterpolator):
            raise TypeError(
                "Every observation must use FixedSpatialInterpolator."
            )
        if self.features.ndim != 3 or self.features.shape[-1] != int(feature_dim):
            raise ValueError(
                f"features must be [B,T,{feature_dim}], "
                f"got {tuple(self.features.shape)}."
            )
        batch, tokens = self.features.shape[:2]
        if tokens != self.interpolation.source_count:
            raise ValueError(
                "Feature count does not match the interpolation source count."
            )
        if (
            not self.features.is_floating_point()
            or not torch.isfinite(self.features).all()
        ):
            raise ValueError("features must be finite floating point values.")
        for name, value in (
            ("coverage", self.coverage),
            ("reliability", self.reliability),
        ):
            if value is not None:
                if value.shape != (batch, tokens):
                    raise ValueError(f"{name} must be [B,T].")
                if (
                    not torch.isfinite(value).all()
                    or torch.any(value < 0)
                    or torch.any(value > 1)
                ):
                    raise ValueError(f"{name} must be finite in [0,1].")


@dataclass(frozen=True)
class NodeFeatureField:
    """Aggregated node features with independently auditable spatial support."""

    features: Tensor
    coverage: Tensor
    reliability: Tensor
    weight_sum: Tensor
    observation_count: Tensor

    def validate(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must be [B,N,D].")
        expected = self.features.shape[:2]
        for name, value in (
            ("coverage", self.coverage),
            ("reliability", self.reliability),
            ("weight_sum", self.weight_sum),
            ("observation_count", self.observation_count),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must be [B,N].")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")
        if torch.any(self.coverage < 0) or torch.any(self.coverage > 1):
            raise ValueError("coverage must lie in [0,1].")
        if torch.any(self.reliability < 0) or torch.any(self.reliability > 1):
            raise ValueError("reliability must lie in [0,1].")
        if torch.any(self.weight_sum < 0) or torch.any(self.observation_count < 0):
            raise ValueError("Aggregation weights and counts cannot be negative.")

    def diagnostics(self) -> dict[str, float]:
        supported = self.coverage > 0
        supported_features = self.features[supported]
        return {
            "coverage_mean": float(self.coverage.mean().detach()),
            "coverage_fraction": float(supported.float().mean().detach()),
            "reliability_mean_supported": float(
                self.reliability[supported].mean().detach()
                if supported.any()
                else 0.0
            ),
            "weight_sum_mean": float(self.weight_sum.mean().detach()),
            "observations_per_supported_node": float(
                self.observation_count[supported].mean().detach()
                if supported.any()
                else 0.0
            ),
            "node_feature_rms_supported": float(
                supported_features.float().square().mean().sqrt().detach()
                if supported_features.numel()
                else 0.0
            ),
        }


class NodeFeatureAggregator(nn.Module):
    """Interpolate and combine any number of observations on shared FEM nodes.

    The same node-level MLP processes every observation. Coverage and
    reliability define a commutative weighted mean, so changing observation
    order cannot change the result.
    """

    def __init__(
        self,
        feature_dim: int,
        node_feature_dim: int,
        *,
        hidden_dim: int | None = None,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.node_feature_dim = int(node_feature_dim)
        hidden = int(hidden_dim or max(self.feature_dim, self.node_feature_dim))
        if min(self.feature_dim, self.node_feature_dim, hidden) < 1:
            raise ValueError(
                "feature_dim, node_feature_dim and hidden_dim must be positive."
            )
        self.feature_projection = nn.Linear(self.feature_dim, hidden)
        self.per_observation_node_mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.node_feature_dim),
        )
        self.eps = float(eps)

    def forward(
        self, observations: Sequence[ObservationTokenBatch]
    ) -> NodeFeatureField:
        if not observations:
            raise ValueError("At least one observation is required.")
        items = tuple(observations)
        for observation in items:
            observation.validate(feature_dim=self.feature_dim)
        batch = int(items[0].features.shape[0])
        nodes = items[0].interpolation.target_count
        device = items[0].features.device
        dtype = items[0].features.dtype
        weighted_feature_sum: Tensor | None = None
        weight_sum: Tensor | None = None
        # Coverage and reliability are physical weights. Their reductions stay
        # in FP32 even when encoded image features use BF16.
        uncovered_fraction_product = torch.ones(
            batch, nodes, device=device, dtype=torch.float32
        )
        reliability_numerator = torch.zeros(
            batch, nodes, device=device, dtype=torch.float32
        )
        reliability_denominator = torch.zeros(
            batch, nodes, device=device, dtype=torch.float32
        )
        observation_count = torch.zeros(
            batch, nodes, device=device, dtype=torch.float32
        )

        for observation in items:
            if observation.features.shape[0] != batch:
                raise ValueError("All observations must use the same batch size.")
            if observation.interpolation.target_count != nodes:
                raise ValueError("All observations must target the same FEM nodes.")
            if (
                observation.features.device != device
                or observation.features.dtype != dtype
            ):
                raise ValueError(
                    "All observation features must share device and dtype."
                )
            token_coverage = observation.coverage
            if token_coverage is None:
                token_coverage = torch.ones(
                    batch,
                    observation.interpolation.source_count,
                    device=device,
                    dtype=dtype,
                )
            token_reliability = observation.reliability
            if token_reliability is None:
                token_reliability = torch.ones_like(token_coverage)

            node_features, node_coverage = observation.interpolation(
                observation.features, token_coverage
            )
            node_reliability, reliability_coverage = observation.interpolation(
                token_reliability.unsqueeze(-1), token_coverage
            )
            if not torch.equal(node_coverage, reliability_coverage):
                raise RuntimeError(
                    "FixedSpatialInterpolator returned inconsistent support."
                )
            node_reliability = node_reliability.squeeze(-1).clamp(0.0, 1.0)
            hidden = self.feature_projection(node_features)
            encoded = self.per_observation_node_mlp(hidden)
            weight = node_coverage * node_reliability
            contribution = encoded * weight.unsqueeze(-1)
            weighted_feature_sum = (
                contribution
                if weighted_feature_sum is None
                else weighted_feature_sum + contribution
            )
            weight_sum = weight if weight_sum is None else weight_sum + weight
            uncovered_fraction_product = uncovered_fraction_product * (
                1.0 - node_coverage.clamp(0.0, 1.0)
            )
            reliability_numerator = (
                reliability_numerator + node_coverage * node_reliability
            )
            reliability_denominator = reliability_denominator + node_coverage
            observation_count = (
                observation_count + (node_coverage > self.eps).float()
            )

        assert weighted_feature_sum is not None and weight_sum is not None
        supported = weight_sum > self.eps
        pooled_features = weighted_feature_sum / weight_sum.clamp_min(
            self.eps
        ).unsqueeze(-1)
        pooled_features = torch.where(
            supported.unsqueeze(-1),
            pooled_features,
            torch.zeros_like(pooled_features),
        )
        features = self.output_projection(pooled_features)
        # Learned biases cannot create features at unsupported nodes.
        features = torch.where(
            supported.unsqueeze(-1), features, torch.zeros_like(features)
        )
        coverage = (1.0 - uncovered_fraction_product).clamp(0.0, 1.0)
        reliability = reliability_numerator / reliability_denominator.clamp_min(
            self.eps
        )
        reliability = torch.where(
            reliability_denominator > self.eps,
            reliability.clamp(0.0, 1.0),
            torch.zeros_like(reliability),
        )
        result = NodeFeatureField(
            features=features,
            coverage=coverage,
            reliability=reliability,
            weight_sum=weight_sum,
            observation_count=observation_count,
        )
        result.validate()
        return result


__all__ = [
    "NodeFeatureAggregator",
    "NodeFeatureField",
    "ObservationTokenBatch",
]
