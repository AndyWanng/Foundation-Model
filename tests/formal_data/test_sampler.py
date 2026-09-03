from __future__ import annotations

import pytest

from mri_pet_geomc.formal.data.sampler import CoverageSampler, RelationSchedule
from mri_pet_geomc.formal.data.schema import FormalDataError


def test_coverage_sampler_no_replacement_and_exact_resume() -> None:
    uids = [f"uid-{index}" for index in range(11)]
    full = CoverageSampler(uids, coverage_id=6, seed=19)
    order = list(full)
    assert len(order) == 11
    assert len({key.anchor_index for key in order}) == 11
    state = full.state_dict(next_offset=4)
    resumed = CoverageSampler.from_state_dict(uids, state)
    assert list(resumed) == order[4:]
    with pytest.raises(FormalDataError, match="digest"):
        CoverageSampler.from_state_dict([*uids[:-1], "changed"], state)


def test_relation_schedule_is_one_based_and_matches_formal_ramp() -> None:
    schedule = RelationSchedule()
    assert schedule.weights(1)["same_observation"] == 1.0
    assert schedule.weights(2)["same_observation"] == 1.0
    assert schedule.weights(3) == {
        "same_session_cross_sequence": 0.075,
        "longitudinal": 0.0375,
        "same_session_repeat": 0.0125,
        "same_observation": 0.875,
    }
    assert schedule.weights(4)["same_observation"] == 0.75
    assert schedule.weights(5)["same_observation"] == 0.625
    assert schedule.weights(6) == {
        "same_session_cross_sequence": 0.30,
        "longitudinal": 0.15,
        "same_session_repeat": 0.05,
        "same_observation": 0.50,
    }
    with pytest.raises(FormalDataError, match="1-based"):
        schedule.weights(0)
