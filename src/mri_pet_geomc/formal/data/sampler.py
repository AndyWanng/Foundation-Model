"""Exact no-replacement coverage order and deterministic relation selection."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from torch.utils.data import Sampler

from ...utils import digest_object
from .schema import FormalDataError, ObservationRelation


@dataclass(frozen=True)
class SampleKey:
    anchor_index: int
    coverage_id: int
    view_id: int = 0


def _hash_bytes(*parts: object) -> bytes:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()


def _uniform(*parts: object) -> float:
    integer = int.from_bytes(_hash_bytes(*parts)[:8], "big", signed=False)
    return integer / float(2**64)


class CoverageSampler(Sampler[SampleKey]):
    """Visit each usable anchor exactly once per coverage.

    ``start_offset`` is the count of *consumed* anchors recorded by the training
    checkpoint.  It is intentionally not advanced by DataLoader prefetch.
    """

    def __init__(
        self,
        observation_uids: Sequence[str],
        *,
        coverage_id: int,
        seed: int,
        start_offset: int = 0,
        view_id: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if len(set(observation_uids)) != len(observation_uids):
            raise FormalDataError("CoverageSampler observation UIDs must be unique")
        if coverage_id < 1 or start_offset < 0 or world_size < 1 or not 0 <= rank < world_size:
            raise FormalDataError("Invalid coverage sampler cursor or distributed rank")
        self.observation_uids = tuple(observation_uids)
        self.coverage_id = int(coverage_id)
        self.seed = int(seed)
        self.start_offset = int(start_offset)
        self.view_id = int(view_id)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.dataset_digest = digest_object(self.observation_uids)
        if self.start_offset > len(self._rank_indices()):
            raise FormalDataError("CoverageSampler start_offset exceeds local coverage length")

    def _global_order(self) -> list[int]:
        return sorted(
            range(len(self.observation_uids)),
            key=lambda index: _hash_bytes(
                self.seed, self.coverage_id, self.observation_uids[index]
            ),
        )

    def _rank_indices(self) -> list[int]:
        return self._global_order()[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[SampleKey]:
        for index in self._rank_indices()[self.start_offset :]:
            yield SampleKey(index, self.coverage_id, self.view_id)

    def __len__(self) -> int:
        return len(self._rank_indices()) - self.start_offset

    def state_dict(self, *, next_offset: int | None = None) -> dict[str, Any]:
        offset = self.start_offset if next_offset is None else int(next_offset)
        if not 0 <= offset <= len(self._rank_indices()):
            raise FormalDataError("Sampler next_offset is outside this rank's coverage")
        return {
            "schema_version": 1,
            "coverage_id": self.coverage_id,
            "next_offset": offset,
            "seed": self.seed,
            "view_id": self.view_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "dataset_digest": self.dataset_digest,
        }

    @classmethod
    def from_state_dict(
        cls, observation_uids: Sequence[str], state: Mapping[str, Any]
    ) -> "CoverageSampler":
        sampler = cls(
            observation_uids,
            coverage_id=int(state["coverage_id"]),
            seed=int(state["seed"]),
            start_offset=int(state["next_offset"]),
            view_id=int(state.get("view_id", 0)),
            rank=int(state.get("rank", 0)),
            world_size=int(state.get("world_size", 1)),
        )
        if state.get("dataset_digest") != sampler.dataset_digest:
            raise FormalDataError("Sampler resume dataset digest mismatch")
        return sampler


@dataclass(frozen=True)
class RelationSchedule:
    """Injectable 1-based relation mix for the first formal run."""

    final_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "same_observation": 0.50,
            "same_session_cross_sequence": 0.30,
            "longitudinal": 0.15,
            "same_session_repeat": 0.05,
        }
    )
    ramp_multipliers: Mapping[int, float] = field(
        default_factory=lambda: {3: 0.25, 4: 0.50, 5: 0.75}
    )

    def __post_init__(self) -> None:
        required = {
            "same_observation",
            "same_session_cross_sequence",
            "longitudinal",
            "same_session_repeat",
        }
        if set(self.final_weights) != required:
            raise FormalDataError(
                f"RelationSchedule final_weights must have exactly {sorted(required)}"
            )
        if any(value < 0 for value in self.final_weights.values()) or not math.isclose(
            sum(self.final_weights.values()), 1.0
        ):
            raise FormalDataError("RelationSchedule final_weights must be non-negative and sum to 1")
        if any(not 0.0 <= value <= 1.0 for value in self.ramp_multipliers.values()):
            raise FormalDataError("RelationSchedule ramp multipliers must be in [0, 1]")

    def weights(self, coverage_id: int) -> Mapping[str, float]:
        if coverage_id < 1:
            raise FormalDataError("coverage_id is 1-based and must be positive")
        if coverage_id <= 2:
            return {
                "same_observation": 1.0,
                "same_session_cross_sequence": 0.0,
                "longitudinal": 0.0,
                "same_session_repeat": 0.0,
            }
        multiplier = self.ramp_multipliers.get(coverage_id, 1.0)
        relation_types = (
            "same_session_cross_sequence",
            "longitudinal",
            "same_session_repeat",
        )
        result = {
            relation_type: self.final_weights[relation_type] * multiplier
            for relation_type in relation_types
        }
        result["same_observation"] = 1.0 - sum(result.values())
        return result


class RelationSelector:
    """Choose at most one legal companion for an anchor, else self fallback."""

    def __init__(
        self,
        relations: Sequence[ObservationRelation],
        *,
        seed: int,
        schedule: RelationSchedule | None = None,
    ) -> None:
        self.seed = int(seed)
        self.schedule = schedule or RelationSchedule()
        self._self: dict[str, ObservationRelation] = {}
        self._by_anchor_type: dict[tuple[str, str], list[ObservationRelation]] = defaultdict(list)
        for relation in relations:
            if relation.relation_type == "same_observation":
                self._self[relation.anchor_uid] = relation
            else:
                self._by_anchor_type[(relation.anchor_uid, relation.relation_type)].append(
                    relation
                )
        for values in self._by_anchor_type.values():
            values.sort(key=lambda relation: relation.relation_uid)

    def select(self, anchor_uid: str, *, coverage_id: int, view_id: int = 0) -> ObservationRelation:
        try:
            fallback = self._self[anchor_uid]
        except KeyError as error:
            raise FormalDataError(f"No self relation for anchor {anchor_uid}") from error
        weights = self.schedule.weights(coverage_id)
        draw = _uniform(self.seed, coverage_id, anchor_uid, view_id, "relation-type")
        cumulative = 0.0
        selected_type = "same_observation"
        for relation_type, weight in weights.items():
            cumulative += weight
            if draw < cumulative:
                selected_type = relation_type
                break
        if selected_type == "same_observation":
            return fallback
        relation_types = (
            ("longitudinal_same_sequence", "longitudinal_same_tracer")
            if selected_type == "longitudinal"
            else (selected_type,)
        )
        candidates: list[ObservationRelation] = []
        for relation_type in relation_types:
            candidates.extend(self._by_anchor_type.get((anchor_uid, relation_type), ()))
        if not candidates:
            return fallback
        candidates.sort(key=lambda relation: relation.relation_uid)
        draw_index = int.from_bytes(
            _hash_bytes(self.seed, coverage_id, anchor_uid, view_id, "relation-candidate")[:8],
            "big",
        ) % len(candidates)
        return candidates[draw_index]
