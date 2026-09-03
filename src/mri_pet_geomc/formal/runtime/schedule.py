from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class RelationKind(str, Enum):
    SELF_ONLY = "self_only"
    SAME_SESSION_CROSS_SEQUENCE = "same_session_cross_sequence"
    LONGITUDINAL_SAME_ACQUISITION = "longitudinal_same_acquisition"
    SAME_SESSION_REPEAT = "same_session_repeat"


@dataclass(frozen=True)
class RelationMixture:
    self_only: float
    same_session_cross_sequence: float
    longitudinal_same_acquisition: float
    same_session_repeat: float

    def __post_init__(self) -> None:
        values = self.as_dict()
        if any(value < 0.0 for value in values.values()):
            raise ValueError(f"Relation mixture contains a negative probability: {values}")
        if abs(sum(values.values()) - 1.0) > 1.0e-9:
            raise ValueError(f"Relation mixture must sum to one: {values}")

    def as_dict(self) -> dict[RelationKind, float]:
        return {
            RelationKind.SELF_ONLY: float(self.self_only),
            RelationKind.SAME_SESSION_CROSS_SEQUENCE: float(
                self.same_session_cross_sequence
            ),
            RelationKind.LONGITUDINAL_SAME_ACQUISITION: float(
                self.longitudinal_same_acquisition
            ),
            RelationKind.SAME_SESSION_REPEAT: float(self.same_session_repeat),
        }

    def choose(self, unit_interval_value: float) -> RelationKind:
        """Map a deterministic value in [0, 1) to a relation kind."""

        value = float(unit_interval_value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"Expected a value in [0, 1), got {value}")
        cumulative = 0.0
        for kind, probability in self.as_dict().items():
            # Canonicalize decimal-configured boundaries (for example 0.5 +
            # 0.3 + 0.15) so an exact 0.95 draw belongs to the next interval.
            cumulative = round(cumulative + probability, 12)
            if value < cumulative:
                return kind
        return RelationKind.SAME_SESSION_REPEAT


@dataclass(frozen=True)
class BatchContract:
    micro_batch_anchors: int = 8
    gradient_accumulation_steps: int = 2
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.micro_batch_anchors < 1:
            raise ValueError("micro_batch_anchors must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")

    @property
    def global_batch_anchors(self) -> int:
        return (
            self.micro_batch_anchors
            * self.gradient_accumulation_steps
            * self.world_size
        )


@dataclass(frozen=True)
class FormalTrainingSchedule:
    """Configured horizon and batch size with the existing relation curriculum."""

    total_coverages: int = 100
    batch: BatchContract = BatchContract()
    fixed_mixture: RelationMixture = RelationMixture(
        self_only=0.50,
        same_session_cross_sequence=0.30,
        longitudinal_same_acquisition=0.15,
        same_session_repeat=0.05,
    )

    def __post_init__(self) -> None:
        if self.total_coverages < 1:
            raise ValueError("total_coverages must be positive")
        if self.batch.world_size != 1:
            raise ValueError(
                "The formal training workflow currently supports a single GPU; "
                "micro-batch and gradient accumulation are configurable"
            )

    def relation_mixture(self, coverage: int) -> RelationMixture:
        """Return the 1-based coverage relation mixture.

        Coverages 1-2 are self-only.  Coverages 3-5 use 25%, 50%, and
        75% of the final relation mass; coverage 6 onward uses the fixed
        50/30/15/5 mixture.
        """

        if not 1 <= int(coverage) <= self.total_coverages:
            raise ValueError(
                f"coverage must be in [1, {self.total_coverages}], got {coverage}"
            )
        if coverage <= 2:
            alpha = 0.0
        elif coverage <= 5:
            alpha = (coverage - 2) / 4.0
        else:
            alpha = 1.0
        target = self.fixed_mixture
        relation_mass = 1.0 - target.self_only
        return RelationMixture(
            self_only=1.0 - alpha * relation_mass,
            same_session_cross_sequence=alpha * target.same_session_cross_sequence,
            longitudinal_same_acquisition=(
                alpha * target.longitudinal_same_acquisition
            ),
            same_session_repeat=alpha * target.same_session_repeat,
        )

    def contract(self) -> Mapping[str, object]:
        return {
            "total_coverages": self.total_coverages,
            "micro_batch_anchors": self.batch.micro_batch_anchors,
            "gradient_accumulation_steps": self.batch.gradient_accumulation_steps,
            "world_size": self.batch.world_size,
            "global_batch_anchors": self.batch.global_batch_anchors,
            "self_only_coverages": [1, 2],
            "relation_ramp_coverages": [3, 4, 5],
            "fixed_relation_start_coverage": 6,
            "fixed_mixture": {
                key.value: value for key, value in self.fixed_mixture.as_dict().items()
            },
        }
