from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mri_pet_geomc.training.jepa import (
    entropy_effective_rank,
    masked_huber_objective,
    representation_diagnostics,
)


def test_masked_huber_matches_manual_subject_balanced_weighting_and_gradient() -> None:
    torch.manual_seed(211)
    prediction = torch.randn(2, 5, 4, requires_grad=True)
    target = torch.randn(2, 5, 4)
    mask = torch.tensor(
        [[True, True, False, True, False], [False, True, True, True, True]]
    )
    weight = torch.tensor(
        [[1.0, 0.5, 0.0, 2.0, 0.0], [0.0, 1.0, 0.5, 0.25, 2.0]]
    )
    result = masked_huber_objective(
        prediction,
        target,
        mask,
        query_weight=weight,
        beta=1.0,
    )
    element = F.smooth_l1_loss(
        prediction.float(), target.float(), beta=1.0, reduction="none"
    ).mean(dim=-1)
    effective_weight = weight * mask
    expected = (
        (element * effective_weight).sum(dim=1)
        / effective_weight.sum(dim=1)
    ).mean()
    assert torch.allclose(result.loss, expected, atol=0.0, rtol=0.0)
    assert float(result.mean_selected_token_count) == 3.5
    assert float(result.mean_absolute_error.detach()) >= 0.0
    expected.backward(retain_graph=True)
    expected_gradient = prediction.grad.detach().clone()
    prediction.grad.zero_()
    result.loss.backward()
    assert torch.equal(prediction.grad, expected_gradient)


def test_masked_huber_stops_target_gradient_and_rejects_invalid_weights() -> None:
    prediction = torch.randn(1, 3, 2, requires_grad=True)
    target = torch.randn(1, 3, 2, requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    result = masked_huber_objective(prediction, target, mask)
    result.loss.backward()
    assert prediction.grad is not None
    assert target.grad is None

    with pytest.raises(ValueError, match="positive total query weight"):
        masked_huber_objective(
            prediction.detach(),
            target.detach(),
            mask,
            query_weight=torch.zeros(1, 3),
        )
    with pytest.raises(ValueError, match="non-negative"):
        masked_huber_objective(
            prediction.detach(),
            target.detach(),
            mask,
            query_weight=torch.tensor([[1.0, -1.0, 0.0]]),
        )


def test_entropy_effective_rank_distinguishes_collapsed_and_multichannel_fields() -> None:
    collapsed = torch.ones(32, 8)
    diverse = torch.eye(8).repeat(4, 1)
    assert float(entropy_effective_rank(collapsed)) == 0.0
    # Centering removes the all-ones direction, leaving seven active directions.
    assert float(entropy_effective_rank(diverse)) > 6.9


def test_representation_diagnostics_honor_query_reliability_weights() -> None:
    prediction = torch.tensor(
        [[[0.0, 0.0], [10.0, 10.0], [4.0, -4.0]]]
    )
    target = torch.zeros_like(prediction)
    mask = torch.ones(1, 3, dtype=torch.bool)
    uniform = representation_diagnostics(prediction, target, mask)
    weighted = representation_diagnostics(
        prediction,
        target,
        mask,
        query_weight=torch.tensor([[100.0, 1.0, 1.0]]),
    )
    assert weighted["rmse"] < uniform["rmse"]
    assert weighted["prediction_rms"] < uniform["prediction_rms"]
