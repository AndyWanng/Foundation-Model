"""Worker-safe relation-aware Dataset backed by mmap shards."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import CacheEntry, MMapCache, load_cache_index
from .sampler import RelationSelector, SampleKey
from .schema import CatalogObservation, FormalDataError, ObservationRelation


@dataclass(frozen=True)
class FormalSample:
    anchor_uid: str
    anchor_image: torch.Tensor
    anchor_support: torch.Tensor
    anchor_metadata: Mapping[str, Any]
    coverage_id: int
    view_id: int
    relation: ObservationRelation
    companion_uid: str = ""
    companion_image: torch.Tensor | None = None
    companion_support: torch.Tensor | None = None
    companion_metadata: Mapping[str, Any] | None = None


def load_preprocessing_overlay(
    receipt_dir: str | Path, observation_uid: str
) -> Mapping[str, Any]:
    """Load only conditioning-relevant structural metadata from a case receipt."""

    path = Path(receipt_dir) / f"{observation_uid}.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if (
        not isinstance(value, Mapping)
        or value.get("status") != "complete"
        or value.get("observation_uid") != observation_uid
    ):
        raise FormalDataError(f"Invalid preprocessing receipt for {observation_uid}")
    geometry = value.get("geometry") or {}
    if not isinstance(geometry, Mapping):
        raise FormalDataError(f"Receipt geometry must be a mapping for {observation_uid}")
    return {
        "geometry": dict(geometry),
        "registration_mode": str(value.get("registration_mode", "")),
        "final_interpolation": str(value.get("final_interpolation", "")),
        "intensity_storage_scale": float(value.get("intensity_storage_scale", 1.0)),
    }


def robust_normalize_visible(
    image: torch.Tensor,
    support: torch.Tensor,
    visible_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    """Median/MAD normalization using visible support only.

    This helper is called after JEPA query masking is known; hidden voxels never
    influence online-encoder intensity statistics.
    """

    if image.ndim != 4 or image.shape[0] != 1:
        raise FormalDataError(f"Expected [1,D,H,W] image, got {tuple(image.shape)}")
    if support.shape != image.shape[1:] or visible_mask.shape != support.shape:
        raise FormalDataError("support and visible_mask must match the image spatial shape")
    selected = support.bool() & visible_mask.bool()
    values = image.float()[0][selected]
    if values.numel() == 0:
        raise FormalDataError("Visible structural support is empty")
    median = values.median()
    mad = (values - median).abs().median()
    scale = torch.clamp(mad * 1.4826, min=epsilon)
    normalized = (image.float() - median) / scale
    normalized = torch.where(support.unsqueeze(0).bool(), normalized, torch.zeros_like(normalized))
    return normalized, {"median": float(median), "robust_scale": float(scale)}


class FormalVolumeDataset(Dataset[FormalSample]):
    """Return one anchor and zero/one deterministic same-subject companion."""

    def __init__(
        self,
        observations: Sequence[CatalogObservation],
        relations: Sequence[ObservationRelation],
        *,
        cache_root: str | Path,
        cache_index: Sequence[CacheEntry] | str | Path,
        split: str,
        relation_seed: int,
        max_open_shards: int = 2,
        receipt_dir: str | Path | None = None,
        transform: Callable[[FormalSample], FormalSample] | None = None,
    ) -> None:
        entries = (
            load_cache_index(cache_index)
            if isinstance(cache_index, (str, Path))
            else tuple(cache_index)
        )
        cached = {entry.observation_uid for entry in entries}
        selected = sorted(
            [row for row in observations if row.split == split and row.observation_uid in cached],
            key=lambda row: row.observation_uid,
        )
        if not selected:
            raise FormalDataError(f"No cached observations for split {split!r}")
        self.observations = tuple(selected)
        self.by_uid = {row.observation_uid: row for row in observations if row.observation_uid in cached}
        usable_relations = [
            relation
            for relation in relations
            if relation.anchor_uid in self.by_uid
            and relation.companion_uid in self.by_uid
            and relation.split == split
        ]
        self.selector = RelationSelector(usable_relations, seed=relation_seed)
        self.cache = MMapCache(
            cache_root, entries, max_open_shards=max_open_shards
        )
        self.receipt_dir = None if receipt_dir is None else Path(receipt_dir)
        self.transform = transform

    def _metadata(self, observation: CatalogObservation) -> Mapping[str, Any]:
        metadata = observation.to_dict()
        if self.receipt_dir is not None:
            overlay = load_preprocessing_overlay(
                self.receipt_dir, observation.observation_uid
            )
            if overlay:
                metadata["preprocessing"] = overlay
        return metadata

    @property
    def observation_uids(self) -> tuple[str, ...]:
        return tuple(row.observation_uid for row in self.observations)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int | SampleKey) -> FormalSample:
        key = index if isinstance(index, SampleKey) else SampleKey(int(index), 1, 0)
        anchor = self.observations[key.anchor_index]
        relation = self.selector.select(
            anchor.observation_uid,
            coverage_id=key.coverage_id,
            view_id=key.view_id,
        )
        anchor_value = self.cache.get(anchor.observation_uid)
        companion_uid = ""
        companion_image = None
        companion_support = None
        companion_metadata = None
        if relation.relation_type != "same_observation":
            companion = self.by_uid[relation.companion_uid]
            companion_value = self.cache.get(companion.observation_uid)
            companion_uid = companion.observation_uid
            companion_image = torch.from_numpy(companion_value.image)
            companion_support = torch.from_numpy(companion_value.support)
            companion_metadata = self._metadata(companion)
        sample = FormalSample(
            anchor_uid=anchor.observation_uid,
            anchor_image=torch.from_numpy(anchor_value.image),
            anchor_support=torch.from_numpy(anchor_value.support),
            anchor_metadata=self._metadata(anchor),
            coverage_id=key.coverage_id,
            view_id=key.view_id,
            relation=relation,
            companion_uid=companion_uid,
            companion_image=companion_image,
            companion_support=companion_support,
            companion_metadata=companion_metadata,
        )
        return self.transform(sample) if self.transform is not None else sample
