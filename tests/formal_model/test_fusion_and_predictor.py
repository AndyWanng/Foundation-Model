from __future__ import annotations

import pytest
import torch

from mri_pet_geomc.formal.model import AnchorCentricFusion, QueryAwareJEPA3DPredictor
from mri_pet_geomc.geometry.interpolation import FixedSpatialInterpolator, InterpolationPlan


def _identity_interpolator(count: int) -> FixedSpatialInterpolator:
    return FixedSpatialInterpolator(
        InterpolationPlan(
            source_indices=torch.arange(count).reshape(count, 1),
            kernel_weights=torch.ones(count, 1),
            source_cell_volume=torch.ones(count),
            full_weight_sum=torch.ones(count),
        )
    )


def test_anchor_centric_fusion_supports_packed_batch_and_no_reliability_field() -> None:
    torch.manual_seed(5)
    fusion = AnchorCentricFusion(
        _identity_interpolator(4),
        feature_dim=6,
        condition_dim=5,
        relation_dim=3,
        output_dim=6,
        hidden_dim=8,
    )
    anchor = torch.randn(2, 4, 6, requires_grad=True)
    companion = torch.randn(2, 4, 6, requires_grad=True)
    field = fusion(
        anchor,
        torch.ones(2, 4),
        torch.randn(2, 5),
        relation_condition=torch.randn(2, 3),
        companion_tokens=companion,
        companion_coverage=torch.ones(2, 4),
        companion_condition=torch.randn(2, 5),
        companion_present=torch.tensor([True, False]),
    )
    assert field.features.shape == (2, 4, 6)
    assert torch.all(field.companion_coverage[1] == 0)
    assert torch.all(field.companion_gate[1] == 0)
    assert not hasattr(field, "reliability")
    field.features.square().mean().backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()


def test_query_aware_predictor_is_batch_safe_bounded_and_has_no_raw_skip() -> None:
    torch.manual_seed(7)
    predictor = QueryAwareJEPA3DPredictor(
        latent_dim=8,
        condition_dim=6,
        relation_dim=4,
        d_model=32,
        depth=2,
        num_heads=4,
        drop_path_rate=0.0,
        max_queries=3,
    )
    latent = torch.randn(2, 5, 8, requires_grad=True)
    output = predictor(
        latent,
        torch.randn(5, 3),
        torch.ones(2, 5),
        torch.randn(2, 3, 3),
        torch.tensor([[True, True, False], [True, True, True]]),
        torch.randn(2, 6),
        torch.randn(2, 4),
    )
    assert output.shape == (2, 3, 8)
    assert torch.all(output[0, 2] == 0)
    assert not hasattr(predictor, "encoder")
    output.square().mean().backward()
    assert latent.grad is not None and float(latent.grad.abs().sum()) > 0
    with pytest.raises(ValueError, match="Query count"):
        predictor(
            torch.randn(1, 5, 8),
            torch.randn(5, 3),
            torch.ones(1, 5),
            torch.randn(1, 4, 3),
            torch.ones(1, 4, dtype=torch.bool),
            torch.randn(1, 6),
            torch.randn(1, 4),
        )
