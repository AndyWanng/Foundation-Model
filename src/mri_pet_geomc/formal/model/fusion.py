"""Anchor-centric fusion for zero or one legal companion per anchor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ...geometry.interpolation import FixedSpatialInterpolator


@dataclass(frozen=True)
class AnchorCentricField:
    """Fused FEM-node field without subjective quality/reliability scores."""

    features: Tensor
    coverage: Tensor
    anchor_coverage: Tensor
    companion_coverage: Tensor
    companion_gate: Tensor
    observation_count: Tensor

    def validate(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must be [B,N,D]")
        expected = self.features.shape[:2]
        for name, value in (
            ("coverage", self.coverage),
            ("anchor_coverage", self.anchor_coverage),
            ("companion_coverage", self.companion_coverage),
            ("companion_gate", self.companion_gate),
            ("observation_count", self.observation_count),
        ):
            if value.shape != expected or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite [B,N]")
        for value in (
            self.coverage,
            self.anchor_coverage,
            self.companion_coverage,
            self.companion_gate,
        ):
            if torch.any(value < 0) or torch.any(value > 1):
                raise ValueError("coverage and gate values must lie in [0,1]")

    def diagnostics(self) -> dict[str, float]:
        companion_supported = self.companion_coverage > 0
        return {
            "coverage_mean": float(self.coverage.detach().float().mean()),
            "anchor_coverage_mean": float(self.anchor_coverage.detach().float().mean()),
            "companion_coverage_mean": float(
                self.companion_coverage.detach().float().mean()
            ),
            "companion_gate_mean_supported": float(
                self.companion_gate[companion_supported].detach().float().mean()
                if companion_supported.any()
                else 0.0
            ),
            "node_feature_rms": float(
                self.features.detach().float().square().mean().sqrt()
            ),
        }


class AnchorCentricFusion(nn.Module):
    """Keep the anchor primary and add one learned, bounded companion residual.

    Companion support is controlled only by physical coverage and an explicit
    ``companion_present`` bit.  No image-quality or reliability scalar exists.
    """

    def __init__(
        self,
        token_to_node: FixedSpatialInterpolator,
        feature_dim: int,
        condition_dim: int,
        relation_dim: int,
        *,
        output_dim: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if min(int(feature_dim), int(condition_dim), int(relation_dim)) < 1:
            raise ValueError("fusion dimensions must be positive")
        self.token_to_node = token_to_node
        self.feature_dim = int(feature_dim)
        self.condition_dim = int(condition_dim)
        self.relation_dim = int(relation_dim)
        self.output_dim = int(output_dim or feature_dim)
        hidden = int(hidden_dim or max(self.feature_dim, self.output_dim))
        self.shared_observation_transform = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        gate_input_dim = 3 * hidden + 2 * self.condition_dim + self.relation_dim
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.197224577)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.output_dim),
        )

    def _validate_tokens(self, tokens: Tensor, coverage: Tensor, *, name: str) -> None:
        expected = (tokens.shape[0], self.token_to_node.source_count)
        if tokens.ndim != 3 or tokens.shape[1:] != (
            self.token_to_node.source_count,
            self.feature_dim,
        ):
            raise ValueError(
                f"{name}_tokens must be [B,{self.token_to_node.source_count},{self.feature_dim}]"
            )
        if coverage.shape != expected:
            raise ValueError(f"{name}_coverage must be [B,T]")
        if not torch.isfinite(tokens).all() or not torch.isfinite(coverage).all():
            raise ValueError(f"{name} tokens/coverage contain non-finite values")
        if torch.any(coverage < 0) or torch.any(coverage > 1):
            raise ValueError(f"{name}_coverage must lie in [0,1]")

    def forward(
        self,
        anchor_tokens: Tensor,
        anchor_coverage: Tensor,
        anchor_condition: Tensor,
        *,
        relation_condition: Tensor,
        companion_tokens: Tensor | None = None,
        companion_coverage: Tensor | None = None,
        companion_condition: Tensor | None = None,
        companion_present: Tensor | None = None,
    ) -> AnchorCentricField:
        self._validate_tokens(anchor_tokens, anchor_coverage, name="anchor")
        batch = anchor_tokens.shape[0]
        if anchor_condition.shape != (batch, self.condition_dim):
            raise ValueError(f"anchor_condition must be [B,{self.condition_dim}]")
        if relation_condition.shape != (batch, self.relation_dim):
            raise ValueError(f"relation_condition must be [B,{self.relation_dim}]")
        anchor_nodes, anchor_node_coverage = self.token_to_node(
            anchor_tokens, anchor_coverage
        )
        anchor_hidden = self.shared_observation_transform(anchor_nodes)
        anchor_hidden = torch.where(
            (anchor_node_coverage > 0).unsqueeze(-1),
            anchor_hidden,
            torch.zeros_like(anchor_hidden),
        )

        if companion_tokens is None:
            if any(
                value is not None
                for value in (companion_coverage, companion_condition, companion_present)
            ):
                raise ValueError("Companion inputs must either all be supplied or all be absent")
            companion_hidden = torch.zeros_like(anchor_hidden)
            companion_node_coverage = torch.zeros_like(anchor_node_coverage)
            gate = torch.zeros_like(anchor_node_coverage)
        else:
            if companion_coverage is None or companion_condition is None or companion_present is None:
                raise ValueError("Companion tokens require coverage, condition and present mask")
            self._validate_tokens(companion_tokens, companion_coverage, name="companion")
            if companion_tokens.shape[0] != batch:
                raise ValueError("Anchor and companion physical batch sizes differ")
            if companion_condition.shape != (batch, self.condition_dim):
                raise ValueError(f"companion_condition must be [B,{self.condition_dim}]")
            present = torch.as_tensor(
                companion_present, device=anchor_tokens.device, dtype=torch.bool
            )
            if present.shape != (batch,):
                raise ValueError("companion_present must be [B]")
            effective_coverage = companion_coverage * present.to(companion_coverage.dtype)[:, None]
            companion_nodes, companion_node_coverage = self.token_to_node(
                companion_tokens, effective_coverage
            )
            companion_hidden = self.shared_observation_transform(companion_nodes)
            anchor_condition_nodes = anchor_condition[:, None, :].expand(
                -1, anchor_hidden.shape[1], -1
            )
            companion_condition_nodes = companion_condition[:, None, :].expand(
                -1, anchor_hidden.shape[1], -1
            )
            relation_nodes = relation_condition[:, None, :].expand(
                -1, anchor_hidden.shape[1], -1
            )
            gate_features = torch.cat(
                (
                    anchor_hidden,
                    companion_hidden,
                    companion_hidden - anchor_hidden,
                    anchor_condition_nodes,
                    companion_condition_nodes,
                    relation_nodes,
                ),
                dim=-1,
            )
            gate = torch.sigmoid(self.gate(gate_features)).squeeze(-1)
            gate = gate * companion_node_coverage

        fused_hidden = anchor_hidden + gate.unsqueeze(-1) * companion_hidden
        union_coverage = 1.0 - (
            (1.0 - anchor_node_coverage) * (1.0 - companion_node_coverage)
        )
        supported = union_coverage > 0
        features = self.output_projection(fused_hidden)
        features = torch.where(supported.unsqueeze(-1), features, torch.zeros_like(features))
        result = AnchorCentricField(
            features=features,
            coverage=union_coverage.clamp(0.0, 1.0),
            anchor_coverage=anchor_node_coverage,
            companion_coverage=companion_node_coverage,
            companion_gate=gate.clamp(0.0, 1.0),
            observation_count=(anchor_node_coverage > 0).float()
            + (companion_node_coverage > 0).float(),
        )
        result.validate()
        return result


__all__ = ["AnchorCentricField", "AnchorCentricFusion"]
