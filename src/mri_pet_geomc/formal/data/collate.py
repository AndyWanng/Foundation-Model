"""Packed collate: dense anchors plus only the companions that exist."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .dataset import FormalSample
from .schema import FormalDataError


def packed_collate(samples: Sequence[FormalSample]) -> dict[str, Any]:
    if not samples:
        raise FormalDataError("Cannot collate an empty formal batch")
    anchor_images = torch.stack([sample.anchor_image for sample in samples])
    anchor_support = torch.stack([sample.anchor_support for sample in samples])
    companion_samples = [
        (index, sample)
        for index, sample in enumerate(samples)
        if sample.companion_image is not None and sample.companion_support is not None
    ]
    if companion_samples:
        companion_images = torch.stack(
            [sample.companion_image for _, sample in companion_samples]  # type: ignore[list-item]
        )
        companion_support = torch.stack(
            [sample.companion_support for _, sample in companion_samples]  # type: ignore[list-item]
        )
    else:
        companion_images = anchor_images.new_empty((0, *anchor_images.shape[1:]))
        companion_support = anchor_support.new_empty((0, *anchor_support.shape[1:]))
    return {
        "anchor_image": anchor_images,
        "anchor_support": anchor_support,
        "anchor_uid": [sample.anchor_uid for sample in samples],
        "anchor_metadata": [sample.anchor_metadata for sample in samples],
        "coverage_id": torch.tensor([sample.coverage_id for sample in samples], dtype=torch.long),
        "view_id": torch.tensor([sample.view_id for sample in samples], dtype=torch.long),
        "relation_type": [sample.relation.relation_type for sample in samples],
        "relation_uid": [sample.relation.relation_uid for sample in samples],
        "temporal_delta_days": [sample.relation.temporal_delta_days for sample in samples],
        "companion_image": companion_images,
        "companion_support": companion_support,
        "companion_batch_index": torch.tensor(
            [index for index, _ in companion_samples], dtype=torch.long
        ),
        "companion_uid": [sample.companion_uid for _, sample in companion_samples],
        "companion_metadata": [
            sample.companion_metadata for _, sample in companion_samples
        ],
    }
