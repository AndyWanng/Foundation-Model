"""Derive legal same-subject context edges from the formal catalog."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ...utils import atomic_write_json, write_jsonl
from .schema import (
    CatalogObservation,
    ObservationRelation,
    stable_uid,
    validate_relations,
)


def _session_sort_key(row: CatalogObservation) -> tuple[float, str, str]:
    order = float("inf") if row.session_order is None else float(row.session_order)
    return (order, row.source_session_id, row.observation_uid)


def _date_delta_days(current: CatalogObservation, previous: CatalogObservation) -> int | None:
    if not current.study_date or not previous.study_date:
        return None
    try:
        return (date.fromisoformat(current.study_date) - date.fromisoformat(previous.study_date)).days
    except ValueError:
        return None


def _relation(
    relation_type: str,
    anchor: CatalogObservation,
    companion: CatalogObservation,
    *,
    direction: str = "symmetric",
    temporal_delta_days: int | None = None,
) -> ObservationRelation:
    return ObservationRelation(
        relation_uid=stable_uid(
            "formal-relation", relation_type, anchor.observation_uid, companion.observation_uid
        ),
        relation_type=relation_type,
        anchor_uid=anchor.observation_uid,
        companion_uid=companion.observation_uid,
        canonical_subject_id=anchor.canonical_subject_id,
        split=anchor.split,
        direction=direction,
        temporal_delta_days=temporal_delta_days,
    )


def _same_session_relations(rows: Sequence[CatalogObservation]) -> list[ObservationRelation]:
    output: list[ObservationRelation] = []
    sessions: dict[str, list[CatalogObservation]] = defaultdict(list)
    for row in rows:
        sessions[row.source_session_id].append(row)
    for session_rows in sessions.values():
        ordered = sorted(session_rows, key=lambda row: row.observation_uid)
        for anchor in ordered:
            for companion in ordered:
                if anchor.observation_uid == companion.observation_uid:
                    continue
                if anchor.modality == "mri" and companion.modality == "mri":
                    complementary = (
                        bool(anchor.relation_key and companion.relation_key)
                        and not anchor.derived
                        and not companion.derived
                        and (
                            anchor.sequence != companion.sequence
                            or anchor.product != companion.product
                        )
                    )
                    if complementary:
                        output.append(
                            _relation("same_session_cross_sequence", anchor, companion)
                        )
                    elif (
                        anchor.sequence == companion.sequence
                        and anchor.product == companion.product
                        and not anchor.derived
                        and not companion.derived
                    ):
                        output.append(_relation("same_session_repeat", anchor, companion))
                elif (
                    anchor.modality == "pet"
                    and companion.modality == "pet"
                    and anchor.relation_key
                    and anchor.tracer == companion.tracer
                ):
                    output.append(_relation("same_session_repeat", anchor, companion))
    return output


def _longitudinal_relations(rows: Sequence[CatalogObservation]) -> list[ObservationRelation]:
    output: list[ObservationRelation] = []
    keyed: dict[tuple[str, str], list[CatalogObservation]] = defaultdict(list)
    for row in rows:
        if row.relation_key:
            keyed[(row.modality, row.relation_key)].append(row)
    for (modality, _), key_rows in keyed.items():
        sessions: dict[str, list[CatalogObservation]] = defaultdict(list)
        for row in key_rows:
            sessions[row.source_session_id].append(row)
        ordered_sessions = sorted(
            sessions.values(), key=lambda values: min(_session_sort_key(row) for row in values)
        )
        for previous_rows, current_rows in zip(ordered_sessions, ordered_sessions[1:]):
            previous = sorted(previous_rows, key=lambda row: row.observation_uid)
            current = sorted(current_rows, key=lambda row: row.observation_uid)
            for anchor in current:
                for companion in previous:
                    relation_type = (
                        "longitudinal_same_sequence"
                        if modality == "mri"
                        else "longitudinal_same_tracer"
                    )
                    output.append(
                        _relation(
                            relation_type,
                            anchor,
                            companion,
                            direction="past_to_current",
                            temporal_delta_days=_date_delta_days(anchor, companion),
                        )
                    )
    return output


@dataclass(frozen=True)
class RelationBuildResult:
    relations: tuple[ObservationRelation, ...]
    summary: Mapping[str, Any]

    def write(self, output_dir: str | Path) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_jsonl(output / "relations.jsonl", (row.to_dict() for row in self.relations))
        atomic_write_json(output / "relations_summary.json", dict(self.summary))
        return output


def build_relations(observations: Sequence[CatalogObservation]) -> RelationBuildResult:
    """Build self, complementary/repeat and adjacent longitudinal relations.

    FOMO and ADNI subjects have disjoint canonical namespaces, therefore no
    MRI/PET relation can be emitted.  Longitudinal edges are represented as a
    current anchor with a past companion; no latent-equality target is implied.
    """

    by_subject: dict[str, list[CatalogObservation]] = defaultdict(list)
    for row in observations:
        by_subject[row.canonical_subject_id].append(row)
    relations: list[ObservationRelation] = []
    for row in observations:
        relations.append(_relation("same_observation", row, row))
    for subject_rows in by_subject.values():
        relations.extend(_same_session_relations(subject_rows))
        relations.extend(_longitudinal_relations(subject_rows))
    relations = sorted(relations, key=lambda row: row.relation_uid)
    validate_relations(relations, observations)
    counts = Counter(row.relation_type for row in relations)
    summary = {
        "relation_count": len(relations),
        "relation_type_counts": dict(counts),
        "cross_modal_relation_count": 0,
        "longitudinal_policy": "adjacent_source_sessions_past_context_only",
        "same_session_cross_sequence_bucket_semantics": (
            "same_session_complementary_acquisition_including_distinct_non_derived_products"
        ),
    }
    return RelationBuildResult(tuple(relations), summary)


def load_relations(path: str | Path) -> tuple[ObservationRelation, ...]:
    import json

    rows: list[ObservationRelation] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(ObservationRelation.from_dict(value))
    return tuple(rows)
