from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class ValidationTier(str, Enum):
    FIXED_MONITOR = "fixed_monitor"
    FULL_VALIDATION = "full_validation"


@dataclass(frozen=True)
class ValidationRequest:
    coverage: int
    optimizer_step: int
    tier: ValidationTier
    reason: str
    sample_limit: int | None
    masks_per_anchor: int
    observational_only: bool = True

    def __post_init__(self) -> None:
        if self.coverage < 1 or self.optimizer_step < 0:
            raise ValueError("Validation coverage/step is invalid")
        if self.sample_limit is not None and self.sample_limit < 1:
            raise ValueError("Validation sample_limit must be positive")
        if self.masks_per_anchor < 1:
            raise ValueError("Validation masks_per_anchor must be positive")
        if not self.observational_only:
            raise ValueError(
                "Formal first-run validation is observational and cannot gate training"
            )


@runtime_checkable
class ValidationCadence(Protocol):
    def requests(
        self,
        *,
        coverage: int,
        optimizer_step: int,
        is_final: bool = False,
    ) -> tuple[ValidationRequest, ...]: ...


@dataclass(frozen=True)
class CoverageValidationCadence:
    """Fixed monitor every coverage and full validation every five coverages."""

    total_coverages: int = 100
    monitor_every_coverages: int = 1
    full_every_coverages: int = 5
    monitor_sample_limit: int = 512
    monitor_masks_per_anchor: int = 2
    full_masks_per_anchor: int = 1

    def __post_init__(self) -> None:
        if self.total_coverages < 1:
            raise ValueError("Validation total_coverages must be positive")
        if min(self.monitor_every_coverages, self.full_every_coverages) < 1:
            raise ValueError("Validation intervals must be positive")
        if min(self.monitor_sample_limit, self.monitor_masks_per_anchor) < 1:
            raise ValueError("Fixed-monitor dimensions must be positive")
        if self.full_masks_per_anchor < 1:
            raise ValueError("Full-validation masks_per_anchor must be positive")

    def requests(
        self,
        *,
        coverage: int,
        optimizer_step: int,
        is_final: bool = False,
    ) -> tuple[ValidationRequest, ...]:
        if not 1 <= coverage <= self.total_coverages:
            raise ValueError(
                f"coverage must be in [1, {self.total_coverages}], got {coverage}"
            )
        final = bool(is_final or coverage == self.total_coverages)
        requests: list[ValidationRequest] = []
        if coverage % self.monitor_every_coverages == 0 or final:
            requests.append(
                ValidationRequest(
                    coverage=coverage,
                    optimizer_step=optimizer_step,
                    tier=ValidationTier.FIXED_MONITOR,
                    reason="fixed stratified monitor",
                    sample_limit=self.monitor_sample_limit,
                    masks_per_anchor=self.monitor_masks_per_anchor,
                )
            )
        if coverage % self.full_every_coverages == 0 or final:
            requests.append(
                ValidationRequest(
                    coverage=coverage,
                    optimizer_step=optimizer_step,
                    tier=ValidationTier.FULL_VALIDATION,
                    reason="final full validation" if final else "scheduled full validation",
                    sample_limit=None,
                    masks_per_anchor=self.full_masks_per_anchor,
                )
            )
        return tuple(requests)
