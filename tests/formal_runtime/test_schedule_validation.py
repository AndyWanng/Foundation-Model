from __future__ import annotations

import math

import pytest

from mri_pet_geomc.formal.runtime import (
    CoverageValidationCadence,
    FormalTrainingSchedule,
    RelationKind,
    ValidationTier,
)


def test_formal_schedule_fixes_horizon_batch_and_relation_curriculum() -> None:
    schedule = FormalTrainingSchedule()

    assert schedule.total_coverages == 100
    assert schedule.batch.micro_batch_anchors == 8
    assert schedule.batch.gradient_accumulation_steps == 2
    assert schedule.batch.global_batch_anchors == 16
    assert schedule.relation_mixture(1).as_dict() == {
        RelationKind.SELF_ONLY: 1.0,
        RelationKind.SAME_SESSION_CROSS_SEQUENCE: 0.0,
        RelationKind.LONGITUDINAL_SAME_ACQUISITION: 0.0,
        RelationKind.SAME_SESSION_REPEAT: 0.0,
    }

    ramp = [schedule.relation_mixture(coverage) for coverage in (3, 4, 5)]
    assert [item.same_session_cross_sequence for item in ramp] == pytest.approx(
        [0.075, 0.15, 0.225]
    )
    fixed = schedule.relation_mixture(6)
    assert fixed.self_only == pytest.approx(0.50)
    assert fixed.same_session_cross_sequence == pytest.approx(0.30)
    assert fixed.longitudinal_same_acquisition == pytest.approx(0.15)
    assert fixed.same_session_repeat == pytest.approx(0.05)
    assert math.isclose(sum(fixed.as_dict().values()), 1.0)


def test_relation_choice_is_deterministic_and_respects_boundaries() -> None:
    mixture = FormalTrainingSchedule().relation_mixture(6)

    assert mixture.choose(0.00) is RelationKind.SELF_ONLY
    assert mixture.choose(0.499999) is RelationKind.SELF_ONLY
    assert mixture.choose(0.50) is RelationKind.SAME_SESSION_CROSS_SEQUENCE
    assert mixture.choose(0.80) is RelationKind.LONGITUDINAL_SAME_ACQUISITION
    assert mixture.choose(0.95) is RelationKind.SAME_SESSION_REPEAT


def test_validation_cadence_is_observational_and_never_a_stop_gate() -> None:
    cadence = CoverageValidationCadence()

    first = cadence.requests(coverage=1, optimizer_step=10)
    assert [request.tier for request in first] == [ValidationTier.FIXED_MONITOR]
    fifth = cadence.requests(coverage=5, optimizer_step=50)
    assert [request.tier for request in fifth] == [
        ValidationTier.FIXED_MONITOR,
        ValidationTier.FULL_VALIDATION,
    ]
    final = cadence.requests(coverage=100, optimizer_step=1000)
    assert len(final) == 2
    assert all(request.observational_only for request in final)
    assert all(request.coverage == 100 for request in final)
