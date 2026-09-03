"""Acquisition-only metadata vocabulary and conditioning.

The formal model deliberately excludes subject, diagnosis, site, dataset and
quality/reliability fields.  Those values may remain in the audit catalogue,
but they cannot enter this encoder by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


CATEGORICAL_FIELDS: tuple[str, ...] = (
    "modality",
    "mri_sequence",
    "mri_product",
    "pet_family",
    "exact_tracer",
    "dwi_representation",
    "mp2rage_component",
    "derived_flag",
)

CONTINUOUS_FIELDS: tuple[str, ...] = (
    "native_spacing_x",
    "native_spacing_y",
    "native_spacing_z",
    "native_resolution_x",
    "native_resolution_y",
    "native_resolution_z",
)

_FORBIDDEN_FIELD_FRAGMENTS: tuple[str, ...] = (
    "quality",
    "reliability",
    "diagnosis",
    "subject",
    "patient",
    "dataset",
    "site",
    "centre",
    "center",
    "age",
    "sex",
    "qc",
)

MISSING_TOKEN = "<MISSING>"
UNKNOWN_TOKEN = "<UNK>"


def _normalise_category(value: Any) -> str:
    if value is None:
        return MISSING_TOKEN
    text = str(value).strip()
    return text if text else MISSING_TOKEN


@dataclass(frozen=True)
class ConditionBatch:
    """Packed acquisition metadata for a physical batch."""

    categorical: Tensor
    continuous: Tensor
    continuous_present: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.categorical.shape[0])

    def validate(self) -> None:
        batch = self.batch_size
        if self.categorical.shape != (batch, len(CATEGORICAL_FIELDS)):
            raise ValueError("categorical metadata must be [B,F_categorical]")
        if self.categorical.dtype != torch.long:
            raise TypeError("categorical metadata must use torch.long ids")
        expected = (batch, len(CONTINUOUS_FIELDS))
        if self.continuous.shape != expected or self.continuous_present.shape != expected:
            raise ValueError("continuous values and presence must be [B,F_continuous]")
        if not torch.is_floating_point(self.continuous):
            raise TypeError("continuous metadata must be floating point")
        if self.continuous_present.dtype != torch.bool:
            raise TypeError("continuous_present must be boolean")
        if torch.any(self.categorical < 0) or not torch.isfinite(self.continuous).all():
            raise ValueError("metadata contains invalid ids or non-finite values")

    def to(self, device: torch.device | str) -> "ConditionBatch":
        return ConditionBatch(
            categorical=self.categorical.to(device=device),
            continuous=self.continuous.to(device=device),
            continuous_present=self.continuous_present.to(device=device),
        )

    def index_select(self, indices: Tensor) -> "ConditionBatch":
        index = torch.as_tensor(indices, device=self.categorical.device, dtype=torch.long)
        return ConditionBatch(
            categorical=self.categorical.index_select(0, index),
            continuous=self.continuous.index_select(0, index),
            continuous_present=self.continuous_present.index_select(0, index),
        )


class MetadataVocabulary:
    """Frozen, serialisable vocabulary built only from acquisition fields."""

    def __init__(
        self,
        categories: Mapping[str, Sequence[str]],
        *,
        continuous_mean: Mapping[str, float] | None = None,
        continuous_scale: Mapping[str, float] | None = None,
    ) -> None:
        unknown_fields = set(categories) - set(CATEGORICAL_FIELDS)
        if unknown_fields:
            raise ValueError(f"Unsupported metadata vocabulary fields: {sorted(unknown_fields)}")
        self.categories: dict[str, tuple[str, ...]] = {}
        self._indices: dict[str, dict[str, int]] = {}
        for field in CATEGORICAL_FIELDS:
            values = {_normalise_category(value) for value in categories.get(field, ())}
            values.discard(MISSING_TOKEN)
            values.discard(UNKNOWN_TOKEN)
            ordered = (MISSING_TOKEN, UNKNOWN_TOKEN, *sorted(values))
            self.categories[field] = ordered
            self._indices[field] = {value: index for index, value in enumerate(ordered)}
        means = continuous_mean or {}
        scales = continuous_scale or {}
        self.continuous_mean = tuple(float(means.get(field, 0.0)) for field in CONTINUOUS_FIELDS)
        self.continuous_scale = tuple(float(scales.get(field, 1.0)) for field in CONTINUOUS_FIELDS)
        if not all(math.isfinite(value) for value in self.continuous_mean):
            raise ValueError("continuous means must be finite")
        if not all(math.isfinite(value) and value > 0 for value in self.continuous_scale):
            raise ValueError("continuous scales must be finite and positive")

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]]) -> "MetadataVocabulary":
        categories = {field: [] for field in CATEGORICAL_FIELDS}
        continuous_values: dict[str, list[float]] = {field: [] for field in CONTINUOUS_FIELDS}
        for record in records:
            cls._reject_forbidden_keys(record)
            for field in CATEGORICAL_FIELDS:
                categories[field].append(_normalise_category(record.get(field)))
            for field in CONTINUOUS_FIELDS:
                raw = record.get(field)
                if raw is None or str(raw).strip() == "":
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError(f"{field} contains a non-finite value")
                continuous_values[field].append(value)
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for field, values in continuous_values.items():
            if values:
                tensor = torch.tensor(values, dtype=torch.float64)
                means[field] = float(tensor.mean())
                scale = float(tensor.std(unbiased=False))
                scales[field] = max(scale, 1.0e-6)
        return cls(categories, continuous_mean=means, continuous_scale=scales)

    @staticmethod
    def _reject_forbidden_keys(record: Mapping[str, Any]) -> None:
        allowed = set(CATEGORICAL_FIELDS) | set(CONTINUOUS_FIELDS)
        unknown = [str(key) for key in record if str(key) not in allowed]
        blocked: list[str] = []
        forbidden = set(_FORBIDDEN_FIELD_FRAGMENTS)
        for key in unknown:
            # Match semantic key tokens, not arbitrary substrings.  In
            # particular, ``age`` must never reject the legitimate acquisition
            # field ``mp2rage_component``.
            tokens = {
                token
                for token in re.split(r"[^a-z0-9]+", key.lower())
                if token
            }
            if tokens & forbidden:
                blocked.append(key)
        if blocked:
            raise ValueError(
                "Forbidden non-acquisition metadata supplied to the model vocabulary: "
                + ", ".join(sorted(blocked))
            )
        if unknown:
            raise ValueError(
                "Unsupported metadata supplied to the acquisition-only vocabulary: "
                + ", ".join(sorted(unknown))
            )

    def encode(self, records: Sequence[Mapping[str, Any]]) -> ConditionBatch:
        if not records:
            raise ValueError("At least one metadata record is required")
        categorical_rows: list[list[int]] = []
        continuous_rows: list[list[float]] = []
        presence_rows: list[list[bool]] = []
        for record in records:
            self._reject_forbidden_keys(record)
            categorical_rows.append(
                [
                    self._indices[field].get(
                        _normalise_category(record.get(field)),
                        self._indices[field][UNKNOWN_TOKEN],
                    )
                    for field in CATEGORICAL_FIELDS
                ]
            )
            values: list[float] = []
            present: list[bool] = []
            for index, field in enumerate(CONTINUOUS_FIELDS):
                raw = record.get(field)
                is_present = raw is not None and str(raw).strip() != ""
                value = float(raw) if is_present else self.continuous_mean[index]
                if not math.isfinite(value):
                    raise ValueError(f"{field} contains a non-finite value")
                values.append(value)
                present.append(is_present)
            continuous_rows.append(values)
            presence_rows.append(present)
        result = ConditionBatch(
            categorical=torch.tensor(categorical_rows, dtype=torch.long),
            continuous=torch.tensor(continuous_rows, dtype=torch.float32),
            continuous_present=torch.tensor(presence_rows, dtype=torch.bool),
        )
        result.validate()
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "categorical_fields": list(CATEGORICAL_FIELDS),
            "continuous_fields": list(CONTINUOUS_FIELDS),
            "categories": {field: list(values) for field, values in self.categories.items()},
            "continuous_mean": dict(zip(CONTINUOUS_FIELDS, self.continuous_mean)),
            "continuous_scale": dict(zip(CONTINUOUS_FIELDS, self.continuous_scale)),
            "forbidden_model_fields": list(_FORBIDDEN_FIELD_FRAGMENTS),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetadataVocabulary":
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported metadata vocabulary schema")
        return cls(
            payload["categories"],
            continuous_mean=payload.get("continuous_mean"),
            continuous_scale=payload.get("continuous_scale"),
        )


class AcquisitionConditionEncoder(nn.Module):
    """Encode frozen vocabulary ids plus continuous values and missingness."""

    def __init__(
        self,
        vocabulary: MetadataVocabulary,
        *,
        output_dim: int = 128,
        categorical_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if min(int(output_dim), int(categorical_embedding_dim)) < 1:
            raise ValueError("condition dimensions must be positive")
        self.output_dim = int(output_dim)
        self.vocabulary = vocabulary
        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(len(vocabulary.categories[field]), categorical_embedding_dim)
                for field in CATEGORICAL_FIELDS
            }
        )
        self.register_buffer(
            "continuous_mean",
            torch.tensor(vocabulary.continuous_mean, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "continuous_scale",
            torch.tensor(vocabulary.continuous_scale, dtype=torch.float32),
            persistent=True,
        )
        input_dim = len(CATEGORICAL_FIELDS) * int(categorical_embedding_dim)
        input_dim += 2 * len(CONTINUOUS_FIELDS)
        hidden = max(self.output_dim, 128)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, batch: ConditionBatch) -> Tensor:
        batch.validate()
        categorical = batch.categorical
        for index, field in enumerate(CATEGORICAL_FIELDS):
            if torch.any(categorical[:, index] >= self.embeddings[field].num_embeddings):
                raise ValueError(f"categorical id outside frozen vocabulary for {field}")
        embedded = [
            self.embeddings[field](categorical[:, index])
            for index, field in enumerate(CATEGORICAL_FIELDS)
        ]
        mean = self.continuous_mean.to(batch.continuous)
        scale = self.continuous_scale.to(batch.continuous)
        normalised = (batch.continuous - mean) / scale
        normalised = torch.where(
            batch.continuous_present,
            normalised,
            torch.zeros_like(normalised),
        )
        combined = torch.cat(
            [*embedded, normalised, batch.continuous_present.to(normalised.dtype)], dim=-1
        )
        result = self.projection(combined)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Acquisition condition encoder produced non-finite values")
        return result


__all__ = [
    "AcquisitionConditionEncoder",
    "CATEGORICAL_FIELDS",
    "CONTINUOUS_FIELDS",
    "ConditionBatch",
    "MetadataVocabulary",
]
