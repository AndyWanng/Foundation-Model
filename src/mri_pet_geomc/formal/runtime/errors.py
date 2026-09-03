from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


class FatalRuntimeError(RuntimeError):
    """Base class for errors for which continuing would invalidate a run."""


class ResumeMismatchError(FatalRuntimeError):
    """The persisted state does not belong to the requested run contract."""


class ManifestContractError(FatalRuntimeError):
    """The source manifest or immutable data contract is invalid or changed."""


class ModelContractError(FatalRuntimeError):
    """The model/checkpoint contract is invalid or cannot be restored strictly."""


class NonFiniteMetricError(FatalRuntimeError):
    """A loss, gradient, learning rate, or other numeric metric is not finite."""


@dataclass(eq=False)
class AcceptableCaseError(Exception):
    """A declared per-case problem that may be skipped with a prominent warning.

    Callbacks must raise this exception deliberately.  Arbitrary exceptions are
    never silently converted into warnings.
    """

    message: str
    code: str = "acceptable_case_error"
    details: Mapping[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


def _walk_numeric(value: Any, path: str) -> list[tuple[str, float]]:
    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Real):
        return [(path, float(value))]
    if isinstance(value, Mapping):
        result: list[tuple[str, float]] = []
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            result.extend(_walk_numeric(item, child))
        return result
    if isinstance(value, Sequence):
        result = []
        for index, item in enumerate(value):
            child = f"{path}[{index}]"
            result.extend(_walk_numeric(item, child))
        return result
    try:
        scalar = value.item()
    except (AttributeError, TypeError, ValueError):
        return []
    if scalar is value:
        return []
    return _walk_numeric(scalar, path)


def ensure_finite_metrics(metrics: Mapping[str, Any], *, context: str = "metrics") -> None:
    """Fail closed when any scalar-like numeric metric is NaN or infinite."""

    failures = [name for name, value in _walk_numeric(metrics, "") if not math.isfinite(value)]
    if failures:
        joined = ", ".join(failures[:8])
        suffix = " ..." if len(failures) > 8 else ""
        raise NonFiniteMetricError(f"Non-finite {context}: {joined}{suffix}")
