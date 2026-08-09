from __future__ import annotations

import pytest
import torch

from mri_pet_geomc.evaluation.fixed_reference import (
    WeightedSimilarityTransform,
    subject_balanced_weights,
)


def _orthogonal(dimension: int, *, seed: int, reflection: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    )
    if reflection and float(torch.linalg.det(matrix)) > 0:
        matrix[:, 0] *= -1
    return matrix


def test_weighted_similarity_recovers_scale_rotation_reflection_and_offset() -> None:
    generator = torch.Generator().manual_seed(701)
    source = torch.randn(300, 9, generator=generator, dtype=torch.float64)
    rotation = _orthogonal(9, seed=703, reflection=True)
    scale = 2.75
    offset = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    target = scale * (source @ rotation) + offset
    weight = torch.linspace(0.2, 2.0, source.shape[0], dtype=torch.float64)

    transform = WeightedSimilarityTransform.fit(source, target, weight)
    recovered = transform.apply(source)

    assert torch.allclose(recovered, target, rtol=0.0, atol=2.0e-12)
    assert transform.scale == pytest.approx(scale, rel=0.0, abs=2.0e-12)
    assert torch.allclose(transform.rotation, rotation, rtol=0.0, atol=2.0e-12)
    assert transform.orthogonality_max_abs_error < 2.0e-12
    assert transform.weighted_fit_rmse < 2.0e-12


def test_fit_is_train_only_and_does_not_depend_on_held_out_rows() -> None:
    generator = torch.Generator().manual_seed(709)
    train_source = torch.randn(80, 5, generator=generator, dtype=torch.float64)
    rotation = _orthogonal(5, seed=719)
    train_target = 1.4 * (train_source @ rotation) + 0.3
    weight = torch.ones(80, dtype=torch.float64)
    first = WeightedSimilarityTransform.fit(train_source, train_target, weight)

    held_out_source = torch.randn(20, 5, generator=generator, dtype=torch.float64)
    held_out_target = torch.randn(20, 5, generator=generator, dtype=torch.float64) * 1.0e6
    # The held-out tensors are intentionally never passed to fit. Applying the
    # same immutable transform cannot mutate or refit its state.
    before = first.state_dict()
    _ = first.apply(held_out_source)
    _ = held_out_target
    after = first.state_dict()

    for key in before:
        if isinstance(before[key], torch.Tensor):
            assert torch.equal(before[key], after[key])
        else:
            assert before[key] == after[key]


def test_collapsed_source_and_collapsed_reference_fail_closed() -> None:
    generator = torch.Generator().manual_seed(727)
    diverse = torch.randn(40, 6, generator=generator, dtype=torch.float64)
    constant = torch.ones_like(diverse)
    weight = torch.ones(40, dtype=torch.float64)

    with pytest.raises(ValueError, match="source representation is collapsed"):
        WeightedSimilarityTransform.fit(constant, diverse, weight)
    with pytest.raises(ValueError, match="target reference is collapsed"):
        WeightedSimilarityTransform.fit(diverse, constant, weight)


def test_negligible_source_energy_relative_to_reference_fails_without_gain_clamp() -> None:
    generator = torch.Generator().manual_seed(733)
    base = torch.randn(60, 4, generator=generator, dtype=torch.float64)
    source = 1.0e-7 * base
    target = base

    with pytest.raises(ValueError, match="collapsed relative"):
        WeightedSimilarityTransform.fit(source, target, torch.ones(60))


@pytest.mark.parametrize(
    ("source", "target", "weight", "message"),
    [
        (torch.zeros(3, 2), torch.zeros(4, 2), torch.ones(3), "identical shapes"),
        (
            torch.tensor([[0.0, float("nan")], [1.0, 2.0]]),
            torch.zeros(2, 2),
            torch.ones(2),
            "source must contain only finite",
        ),
        (torch.zeros(3, 2), torch.ones(3, 2), torch.tensor([1.0, -1.0, 1.0]), "non-negative"),
        (torch.zeros(3, 2), torch.ones(3, 2), torch.tensor([1.0, 0.0, 0.0]), "two positive"),
    ],
)
def test_fit_rejects_invalid_contracts(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        WeightedSimilarityTransform.fit(source, target, weight)


def test_subject_balanced_weights_equalize_subjects_then_views() -> None:
    raw = torch.tensor([1.0, 3.0, 2.0, 2.0, 6.0, 2.0], dtype=torch.float64)
    subjects = ["A", "A", "A", "B", "B", "B"]
    groups = ["A-v1", "A-v1", "A-v2", "B-v1", "B-v1", "B-v1"]

    weight = subject_balanced_weights(raw, subjects, groups)

    assert float(weight[:3].sum()) == pytest.approx(0.5)
    assert float(weight[3:].sum()) == pytest.approx(0.5)
    assert float(weight[:2].sum()) == pytest.approx(0.25)
    assert float(weight[2]) == pytest.approx(0.25)
    assert float(weight[3:].sum()) == pytest.approx(0.5)
    assert float(weight.sum()) == pytest.approx(1.0)


def test_subject_balancing_is_invariant_to_token_duplication_with_split_weight() -> None:
    original = subject_balanced_weights(
        torch.tensor([2.0, 1.0]), ["A", "B"], ["A-v1", "B-v1"]
    )
    duplicated = subject_balanced_weights(
        torch.tensor([1.0, 1.0, 1.0]),
        ["A", "A", "B"],
        ["A-v1", "A-v1", "B-v1"],
    )

    assert float(original[0]) == pytest.approx(float(duplicated[:2].sum()))
    assert float(original[1]) == pytest.approx(float(duplicated[2]))


def test_subject_balancing_rejects_zero_reliability_group() -> None:
    with pytest.raises(ValueError, match="zero positive reliability"):
        subject_balanced_weights(
            torch.tensor([1.0, 0.0]),
            ["A", "B"],
            ["A-v1", "B-v1"],
        )


def test_transform_state_round_trip_is_exact() -> None:
    generator = torch.Generator().manual_seed(739)
    source = torch.randn(100, 7, generator=generator, dtype=torch.float64)
    target = 0.8 * (source @ _orthogonal(7, seed=743)) - 0.2
    fitted = WeightedSimilarityTransform.fit(source, target, torch.ones(100))

    restored = WeightedSimilarityTransform.from_state_dict(fitted.state_dict())

    assert torch.equal(restored.source_mean, fitted.source_mean)
    assert torch.equal(restored.target_mean, fitted.target_mean)
    assert torch.equal(restored.rotation, fitted.rotation)
    assert restored.diagnostics() == fitted.diagnostics()
    assert torch.equal(restored.apply(source), fitted.apply(source))


def test_apply_preserves_shape_dtype_and_rejects_wrong_dimension() -> None:
    generator = torch.Generator().manual_seed(751)
    source = torch.randn(50, 3, generator=generator, dtype=torch.float64)
    target = 1.2 * (source @ _orthogonal(3, seed=757))
    transform = WeightedSimilarityTransform.fit(source, target, torch.ones(50))
    batched = torch.randn(2, 4, 3, generator=generator, dtype=torch.float32)

    aligned = transform.apply(batched)

    assert aligned.shape == batched.shape
    assert aligned.dtype == batched.dtype
    with pytest.raises(ValueError, match="channel dimension"):
        transform.apply(torch.zeros(2, 4, 2))
