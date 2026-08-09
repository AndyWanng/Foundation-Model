from __future__ import annotations

import math

import pytest
import torch

from mri_pet_geomc.evaluation.metrics import (
    aggregate_subject_metrics,
    compare_pairing_conditions,
    masked_latent_metrics,
    paired_subject_bootstrap_difference,
    representation_statistics,
)


def test_masked_latent_metrics_are_subject_first_and_ignore_padding() -> None:
    prediction = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [float("nan"), float("nan")]],
            [[2.0, 0.0], [float("nan"), float("nan")], [float("nan"), float("nan")]],
        ]
    )
    target = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [float("nan"), float("nan")]],
            [[0.0, 0.0], [float("nan"), float("nan")], [float("nan"), float("nan")]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])

    metrics = masked_latent_metrics(prediction, target, mask, huber_beta=1.0)

    # Subject 1 is perfect.  Subject 2 has per-channel errors [2, 0], hence
    # Huber=(1.5+0)/2=.75 and MSE=(4+0)/2=2.  Subjects receive equal weight.
    assert metrics == pytest.approx(
        {"huber": 0.375, "mse": 1.0, "cosine": 0.5, "mean_selected_token_count": 1.5}
    )


def test_masked_latent_metrics_honor_token_weights_within_subject() -> None:
    prediction = torch.tensor([[[0.0], [2.0]], [[1.0], [3.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.ones((2, 2), dtype=torch.bool)
    weight = torch.tensor([[1.0, 3.0], [1.0, 1.0]])

    metrics = masked_latent_metrics(
        prediction, target, mask, token_weight=weight, huber_beta=1.0
    )

    # Weighted subject MSEs: 3 and 5; the cohort result is their mean, 4.
    assert metrics["mse"] == pytest.approx(4.0)
    assert metrics["mean_selected_token_count"] == 2.0


def test_masked_latent_metrics_reject_empty_subject_mask() -> None:
    value = torch.zeros((2, 2, 3))
    with pytest.raises(ValueError, match="at least one token"):
        masked_latent_metrics(
            value, value, torch.tensor([[True, False], [False, False]])
        )


def test_representation_statistics_measure_rank_variance_and_amplitude() -> None:
    latent = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]
    )
    mask = torch.ones((1, 4), dtype=torch.bool)

    metrics = representation_statistics(latent, mask)

    assert metrics["variance"] == pytest.approx(0.5)
    assert metrics["effective_rank"] == pytest.approx(2.0)
    assert metrics["rms_amplitude"] == pytest.approx(math.sqrt(0.5))


def test_representation_statistics_reports_constant_collapse_as_rank_zero() -> None:
    latent = torch.full((2, 3, 4), 2.0)
    mask = torch.ones((2, 3), dtype=torch.bool)

    metrics = representation_statistics(latent, mask)

    assert metrics == pytest.approx(
        {"variance": 0.0, "effective_rank": 0.0, "rms_amplitude": 2.0}
    )


def test_subject_aggregation_prevents_duplicate_views_from_reweighting_subjects() -> None:
    rows = [
        {"subject_id": "A", "condition": "verified", "loss": 0.0},
        {"subject_id": "A", "condition": "verified", "loss": 2.0},
        {"subject_id": "B", "condition": "verified", "loss": 4.0},
        {"subject_id": "A", "condition": "wrong_subject", "loss": 6.0},
        {"subject_id": "B", "condition": "wrong_subject", "loss": 8.0},
    ]

    result = aggregate_subject_metrics(
        rows, metric_keys=["loss"], group_keys=["condition"]
    )

    correct = next(
        row for row in result["summary_rows"] if row["condition"] == "verified"
    )
    assert correct["loss"] == pytest.approx(2.5)  # mean(subject A=1, subject B=4)
    assert correct["subject_count"] == 2
    assert correct["view_count"] == 3
    assert result["view_count"] == 5
    assert result["unique_subject_count"] == 2


def test_subject_aggregation_is_deterministically_sorted() -> None:
    rows = [
        {"subject_id": "B", "condition": "wrong_subject", "loss": 2.0},
        {"subject_id": "A", "condition": "verified", "loss": 1.0},
        {"subject_id": "A", "condition": "wrong_subject", "loss": 3.0},
    ]
    first = aggregate_subject_metrics(
        rows, metric_keys=["loss"], group_keys=["condition"]
    )
    second = aggregate_subject_metrics(
        list(reversed(rows)), metric_keys=["loss"], group_keys=["condition"]
    )
    assert first == second


def test_paired_subject_bootstrap_is_deterministic_and_uses_paired_differences() -> None:
    left = {"C": 6.0, "A": 2.0, "B": 4.0}
    right = {"A": 1.0, "B": 1.0, "C": 1.0}

    first = paired_subject_bootstrap_difference(
        left, right, n_resamples=2_000, confidence=0.90, seed=17
    )
    second = paired_subject_bootstrap_difference(
        left, right, n_resamples=2_000, confidence=0.90, seed=17
    )

    assert first == second
    assert first["estimate"] == pytest.approx(3.0)
    assert first["subject_ids"] == ["A", "B", "C"]
    assert 1.0 <= first["ci_low"] <= first["ci_high"] <= 5.0
    assert first["bootstrap_fraction_above_zero"] == 1.0


def test_paired_subject_bootstrap_rejects_unmatched_subject_sets() -> None:
    with pytest.raises(ValueError, match="identical subject sets"):
        paired_subject_bootstrap_difference(
            {"A": 1.0, "B": 2.0}, {"A": 0.0}, n_resamples=10
        )


def test_pairing_comparison_aggregates_views_and_orients_loss_advantage() -> None:
    rows = [
        {"subject_id": "A", "pairing_mode": "verified", "loss": 1.0},
        {"subject_id": "A", "pairing_mode": "verified", "loss": 3.0},
        {"subject_id": "B", "pairing_mode": "verified", "loss": 4.0},
        {"subject_id": "A", "pairing_mode": "wrong_subject", "loss": 6.0},
        {"subject_id": "B", "pairing_mode": "wrong_subject", "loss": 8.0},
        {"subject_id": "A", "pairing_mode": "unpaired", "loss": 5.0},
        {"subject_id": "B", "pairing_mode": "unpaired", "loss": 7.0},
    ]

    result = compare_pairing_conditions(rows, metric_key="loss", n_resamples=100)

    assert result["pairing_means"] == pytest.approx(
        {"verified": 3.0, "wrong_subject": 7.0, "unpaired": 6.0}
    )
    wrong = result["comparisons"]["verified_vs_mismatched"]
    unpaired = result["comparisons"]["verified_vs_unpaired"]
    assert wrong["estimate"] == pytest.approx(4.0)
    assert wrong["ci_low"] == pytest.approx(4.0)
    assert wrong["ci_high"] == pytest.approx(4.0)
    assert wrong["advantage_definition"] == "wrong_subject - verified"
    assert unpaired["estimate"] == pytest.approx(3.0)
    assert unpaired["positive_means"] == "verified_pairing_is_better"


def test_pairing_comparison_rejects_a_pairing_mode_with_different_subjects() -> None:
    rows = [
        {"subject_id": "A", "pairing_mode": "verified", "loss": 1.0},
        {"subject_id": "B", "pairing_mode": "verified", "loss": 2.0},
        {"subject_id": "A", "pairing_mode": "wrong_subject", "loss": 3.0},
        {"subject_id": "A", "pairing_mode": "unpaired", "loss": 4.0},
        {"subject_id": "B", "pairing_mode": "unpaired", "loss": 5.0},
    ]
    with pytest.raises(ValueError, match="identical subject sets"):
        compare_pairing_conditions(rows, metric_key="loss", n_resamples=10)
