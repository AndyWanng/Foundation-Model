from __future__ import annotations

import torch

from mri_pet_geomc.field.node_features import (
    NodeFeatureAggregator,
    ObservationTokenBatch,
)
from mri_pet_geomc.geometry.interpolation import (
    FixedSpatialInterpolator,
    InterpolationPlan,
)


def _interpolator(offset: float = 0.0) -> FixedSpatialInterpolator:
    source_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    target_xyz = torch.tensor(
        [[0.2 + offset, 0.0, 0.0], [1.5 + offset, 0.0, 0.0], [2.8 + offset, 0.0, 0.0]],
        dtype=torch.float64,
    )
    plan = InterpolationPlan.from_coordinates(
        source_xyz, target_xyz, neighbors=2, sigma_mm=0.8
    )
    return FixedSpatialInterpolator(plan)


def _observation(seed: int, offset: float) -> ObservationTokenBatch:
    generator = torch.Generator().manual_seed(seed)
    return ObservationTokenBatch(
        features=torch.randn(2, 4, 5, generator=generator),
        interpolation=_interpolator(offset),
        coverage=torch.tensor([[1.0, 1.0, 0.5, 0.0], [0.5, 1.0, 1.0, 1.0]]),
        reliability=torch.tensor([[0.9, 0.8, 0.7, 0.4], [0.8, 0.9, 1.0, 0.7]]),
    )


def test_observation_aggregation_is_permutation_invariant() -> None:
    torch.manual_seed(3)
    aggregator = NodeFeatureAggregator(5, 7, hidden_dim=11)
    first = _observation(11, 0.0)
    second = _observation(13, 0.05)
    forward = aggregator([first, second])
    reverse = aggregator([second, first])
    assert torch.allclose(forward.features, reverse.features, atol=2.0e-7, rtol=1.0e-6)
    assert torch.allclose(forward.coverage, reverse.coverage, atol=1.0e-7)
    assert torch.allclose(forward.reliability, reverse.reliability, atol=1.0e-7)
    assert torch.equal(forward.observation_count, reverse.observation_count)
    assert forward.features.shape == (2, 3, 7)
    assert forward.diagnostics()["coverage_fraction"] > 0.0


def test_aggregator_returns_node_features_and_weight_sums() -> None:
    torch.manual_seed(5)
    aggregator = NodeFeatureAggregator(5, 7, hidden_dim=11)
    first = _observation(17, 0.0)
    second = _observation(19, 0.05)
    result = aggregator([first, second])
    assert result.features.shape == (2, 3, 7)
    assert result.weight_sum.shape == (2, 3)


def test_interpolation_preserves_constants_and_aggregation_preserves_gradients() -> None:
    interpolation = _interpolator()
    constant = torch.full((1, 4, 3), 2.75, requires_grad=True)
    node_values, coverage = interpolation(constant)
    assert torch.allclose(node_values, torch.full_like(node_values, 2.75), atol=1.0e-6)
    assert torch.allclose(coverage, torch.ones_like(coverage), atol=1.0e-6)
    aggregator = NodeFeatureAggregator(3, 4)
    field = aggregator([ObservationTokenBatch(constant, interpolation)])
    field.features.square().mean().backward()
    assert constant.grad is not None
    assert torch.isfinite(constant.grad).all()


def test_unsupported_nodes_cannot_be_created_by_network_bias() -> None:
    interpolation = _interpolator()
    observation = ObservationTokenBatch(
        features=torch.randn(1, 4, 3),
        interpolation=interpolation,
        coverage=torch.zeros(1, 4),
    )
    result = NodeFeatureAggregator(3, 4)([observation])
    assert torch.equal(result.features, torch.zeros_like(result.features))
    assert torch.equal(result.coverage, torch.zeros_like(result.coverage))
    assert torch.equal(result.reliability, torch.zeros_like(result.reliability))


def test_interpolation_support_is_feature_dtype_invariant_under_bfloat16() -> None:
    interpolation = _interpolator()
    coverage = torch.tensor([[1.0, 0.75, 0.5, 0.0]], dtype=torch.float32)
    bf16_features = torch.randn(1, 4, 3).to(torch.bfloat16)
    fp32_reliability = torch.rand(1, 4, 1)
    values, support_from_bf16 = interpolation(bf16_features, coverage)
    _, support_from_fp32 = interpolation(fp32_reliability, coverage)
    assert values.dtype == torch.bfloat16
    assert support_from_bf16.dtype == torch.float32
    assert torch.equal(support_from_bf16, support_from_fp32)