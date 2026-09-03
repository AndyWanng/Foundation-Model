"""Typed, quality-agnostic records for the formal FOMO/ADNI corpus.

These records deliberately describe only identity, acquisition, location and
geometry contracts.  They contain no human image rating, confidence weight or
per-case quality score.  A case is either structurally materialized and usable,
or its preprocessing attempt is recorded separately as a failure.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
UID_NAMESPACE = uuid.UUID("8920c803-5cd8-4cb5-aece-b449147d0bd5")
VALID_MODALITIES = {"mri", "pet"}
VALID_SPLITS = {"train", "validation", "test"}
VALID_SOURCE_KINDS = {"nifti", "zip_member", "dicom_series"}
VALID_RELATIONS = {
    "same_observation",
    "same_session_cross_sequence",
    "same_session_repeat",
    "longitudinal_same_sequence",
    "longitudinal_same_tracer",
}


class FormalDataError(RuntimeError):
    """Raised when a formal-data structural contract is violated."""


def stable_uid(kind: str, *parts: object) -> str:
    """Return a stable UUID independent of table and filesystem enumeration order."""

    payload = "|".join([kind, *(str(part) for part in parts)])
    return str(uuid.uuid5(UID_NAMESPACE, payload))


def digest_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clean_text(value: object) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class SourceLocator:
    """A physical input location.

    ``path`` is the ZIP file, DICOM series directory, or NIfTI file.  It may be
    workstation-absolute or relative to a configured data root.  ZIP members
    are never extracted by path; the preprocessor streams the exact member.
    """

    kind: str
    path: str
    relative_path: str
    archive_member: str = ""
    file_count: int = 1
    signature: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_SOURCE_KINDS:
            raise FormalDataError(f"Unsupported source kind: {self.kind!r}")
        if not _clean_text(self.path) or not _clean_text(self.relative_path):
            raise FormalDataError("Source path and relative_path must be non-empty")
        if self.kind == "zip_member" and not _clean_text(self.archive_member):
            raise FormalDataError("A zip_member source requires archive_member")
        if self.kind != "zip_member" and self.archive_member:
            raise FormalDataError(f"{self.kind} source cannot have archive_member")
        if int(self.file_count) < 1:
            raise FormalDataError("Source file_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "relative_path": self.relative_path,
            "archive_member": self.archive_member,
            "file_count": int(self.file_count),
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceLocator":
        return cls(
            kind=_clean_text(value.get("kind")),
            path=_clean_text(value.get("path")),
            relative_path=_clean_text(value.get("relative_path")),
            archive_member=_clean_text(value.get("archive_member")),
            file_count=int(value.get("file_count", 1)),
            signature=_clean_text(value.get("signature")),
        )


@dataclass(frozen=True)
class CatalogObservation:
    """One materialized MRI product or ADNI PET Image-ID series."""

    observation_uid: str
    dataset_id: str
    canonical_subject_id: str
    participant_id: str
    modality: str
    session_id: str
    source_session_id: str
    acquisition_uid: str
    source: SourceLocator
    split: str = ""
    sequence: str = ""
    product: str = ""
    tracer: str = ""
    tracer_family: str = ""
    study_date: str = ""
    session_order: float | None = None
    derived: bool = False
    acquisition_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {
            "observation_uid": self.observation_uid,
            "dataset_id": self.dataset_id,
            "canonical_subject_id": self.canonical_subject_id,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "source_session_id": self.source_session_id,
            "acquisition_uid": self.acquisition_uid,
        }
        missing = [name for name, value in required.items() if not _clean_text(value)]
        if missing:
            raise FormalDataError(f"Observation is missing required fields: {missing}")
        if self.modality not in VALID_MODALITIES:
            raise FormalDataError(f"Unsupported modality: {self.modality!r}")
        if self.split and self.split not in VALID_SPLITS:
            raise FormalDataError(f"Unsupported split: {self.split!r}")
        expected_prefix = f"{'FOMO' if self.modality == 'mri' else 'ADNI'}:"
        if not self.canonical_subject_id.startswith(expected_prefix):
            raise FormalDataError(
                f"Subject {self.canonical_subject_id!r} lacks {expected_prefix!r} namespace"
            )
        if self.session_order is not None and not math.isfinite(float(self.session_order)):
            raise FormalDataError("session_order must be finite when supplied")
        if not isinstance(self.acquisition_metadata, Mapping):
            raise FormalDataError("acquisition_metadata must be a mapping")

    @property
    def relation_key(self) -> str:
        if self.modality == "pet":
            value = self.tracer.upper()
            return "" if value in {"", "UNKNOWN"} else value
        value = self.sequence.lower()
        if value in {"", "unknown"}:
            return ""
        product = self.product.lower()
        if product and product != value:
            return f"{value}:{product}"
        return value

    def with_split(self, split: str) -> "CatalogObservation":
        value = self.to_dict()
        value["split"] = split
        return CatalogObservation.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "observation_uid": self.observation_uid,
            "dataset_id": self.dataset_id,
            "canonical_subject_id": self.canonical_subject_id,
            "participant_id": self.participant_id,
            "modality": self.modality,
            "session_id": self.session_id,
            "source_session_id": self.source_session_id,
            "acquisition_uid": self.acquisition_uid,
            "source": self.source.to_dict(),
            "split": self.split,
            "sequence": self.sequence,
            "product": self.product,
            "tracer": self.tracer,
            "tracer_family": self.tracer_family,
            "study_date": self.study_date,
            "session_order": self.session_order,
            "derived": bool(self.derived),
            "acquisition_metadata": dict(self.acquisition_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CatalogObservation":
        order = value.get("session_order")
        return cls(
            observation_uid=_clean_text(value.get("observation_uid")),
            dataset_id=_clean_text(value.get("dataset_id")),
            canonical_subject_id=_clean_text(value.get("canonical_subject_id")),
            participant_id=_clean_text(value.get("participant_id")),
            modality=_clean_text(value.get("modality")).lower(),
            session_id=_clean_text(value.get("session_id")),
            source_session_id=_clean_text(value.get("source_session_id")),
            acquisition_uid=_clean_text(value.get("acquisition_uid")),
            source=SourceLocator.from_dict(value.get("source") or {}),
            split=_clean_text(value.get("split")).lower(),
            sequence=_clean_text(value.get("sequence")),
            product=_clean_text(value.get("product")),
            tracer=_clean_text(value.get("tracer")),
            tracer_family=_clean_text(value.get("tracer_family")),
            study_date=_clean_text(value.get("study_date")),
            session_order=None if order in (None, "") else float(order),
            derived=bool(value.get("derived", False)),
            acquisition_metadata=dict(value.get("acquisition_metadata") or {}),
        )


@dataclass(frozen=True)
class SubjectSplit:
    canonical_subject_id: str
    dataset_id: str
    split: str

    def __post_init__(self) -> None:
        if not self.canonical_subject_id or not self.dataset_id:
            raise FormalDataError("SubjectSplit identifiers must be non-empty")
        if self.split not in VALID_SPLITS:
            raise FormalDataError(f"Unsupported split: {self.split!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "canonical_subject_id": self.canonical_subject_id,
            "dataset_id": self.dataset_id,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubjectSplit":
        return cls(
            canonical_subject_id=_clean_text(value.get("canonical_subject_id")),
            dataset_id=_clean_text(value.get("dataset_id")),
            split=_clean_text(value.get("split")).lower(),
        )


@dataclass(frozen=True)
class ObservationRelation:
    """A legal same-subject context edge; never an unverified MRI/PET pair."""

    relation_uid: str
    relation_type: str
    anchor_uid: str
    companion_uid: str
    canonical_subject_id: str
    split: str
    direction: str = "symmetric"
    temporal_delta_days: int | None = None

    def __post_init__(self) -> None:
        if self.relation_type not in VALID_RELATIONS:
            raise FormalDataError(f"Unsupported relation: {self.relation_type!r}")
        if not all(
            [self.relation_uid, self.anchor_uid, self.companion_uid, self.canonical_subject_id]
        ):
            raise FormalDataError("Relation identifiers must be non-empty")
        if self.split not in VALID_SPLITS:
            raise FormalDataError(f"Unsupported relation split: {self.split!r}")
        if self.relation_type.startswith("longitudinal_") and self.direction != "past_to_current":
            raise FormalDataError("Longitudinal relations must be directed past_to_current")
        if self.relation_type == "same_observation" and self.anchor_uid != self.companion_uid:
            raise FormalDataError("same_observation must have identical endpoints")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "relation_uid": self.relation_uid,
            "relation_type": self.relation_type,
            "anchor_uid": self.anchor_uid,
            "companion_uid": self.companion_uid,
            "canonical_subject_id": self.canonical_subject_id,
            "split": self.split,
            "direction": self.direction,
            "temporal_delta_days": self.temporal_delta_days,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationRelation":
        delta = value.get("temporal_delta_days")
        return cls(
            relation_uid=_clean_text(value.get("relation_uid")),
            relation_type=_clean_text(value.get("relation_type")),
            anchor_uid=_clean_text(value.get("anchor_uid")),
            companion_uid=_clean_text(value.get("companion_uid")),
            canonical_subject_id=_clean_text(value.get("canonical_subject_id")),
            split=_clean_text(value.get("split")).lower(),
            direction=_clean_text(value.get("direction")) or "symmetric",
            temporal_delta_days=None if delta in (None, "") else int(delta),
        )


def validate_catalog(
    observations: Sequence[CatalogObservation], splits: Sequence[SubjectSplit]
) -> None:
    if not observations:
        raise FormalDataError("Catalog has no materialized observations")
    split_map: dict[str, str] = {}
    dataset_map: dict[str, str] = {}
    for row in splits:
        if row.canonical_subject_id in split_map:
            raise FormalDataError(f"Duplicate subject split: {row.canonical_subject_id}")
        split_map[row.canonical_subject_id] = row.split
        dataset_map[row.canonical_subject_id] = row.dataset_id
    seen_observations: set[str] = set()
    seen_acquisitions: set[tuple[str, str]] = set()
    observed_subjects: set[str] = set()
    for row in observations:
        if row.observation_uid in seen_observations:
            raise FormalDataError(f"Duplicate observation_uid: {row.observation_uid}")
        seen_observations.add(row.observation_uid)
        acquisition_key = (row.dataset_id, row.acquisition_uid)
        if acquisition_key in seen_acquisitions:
            raise FormalDataError(f"Duplicate acquisition identity: {acquisition_key}")
        seen_acquisitions.add(acquisition_key)
        if split_map.get(row.canonical_subject_id) != row.split:
            raise FormalDataError(
                f"Observation {row.observation_uid} conflicts with subject split"
            )
        if dataset_map.get(row.canonical_subject_id) != row.dataset_id:
            raise FormalDataError(
                f"Subject {row.canonical_subject_id} conflicts across dataset namespaces"
            )
        observed_subjects.add(row.canonical_subject_id)
    if set(split_map) != observed_subjects:
        raise FormalDataError("Split table and materialized catalog subject sets differ")


def validate_relations(
    relations: Sequence[ObservationRelation], observations: Sequence[CatalogObservation]
) -> None:
    by_uid = {row.observation_uid: row for row in observations}
    seen: set[str] = set()
    for relation in relations:
        if relation.relation_uid in seen:
            raise FormalDataError(f"Duplicate relation_uid: {relation.relation_uid}")
        seen.add(relation.relation_uid)
        try:
            anchor = by_uid[relation.anchor_uid]
            companion = by_uid[relation.companion_uid]
        except KeyError as error:
            raise FormalDataError(f"Relation references unknown endpoint: {error.args[0]}") from error
        if anchor.canonical_subject_id != companion.canonical_subject_id:
            raise FormalDataError("Relation crosses subjects")
        if anchor.split != companion.split or relation.split != anchor.split:
            raise FormalDataError("Relation crosses splits")
        if anchor.canonical_subject_id != relation.canonical_subject_id:
            raise FormalDataError("Relation subject does not match endpoints")
        if anchor.modality != companion.modality:
            raise FormalDataError("Formal relation table cannot contain MRI/PET edges")


def path_text(path: str | Path) -> str:
    return Path(path).as_posix()
