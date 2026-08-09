"""MRI/PET observation records and deterministic relation sampling.

An observation is one actually available acquisition derivative.  It is valid
on its own: an MRI-only or PET-only subject is not an error.  Cross-modal
correspondence is represented separately by :class:`CrossModalLink` and is
usable only when the link is explicitly verified.

The order of operations is intentional and enforced throughout this module::

    observations -> subject-level split assignment -> verified links -> relations

``verified_pair_retention_fraction`` in :class:`ObservationRelationSampler` deletes a deterministic
fraction of *known verified links*.  It is an experimental pair-link deletion
stress test and must not be reported as evidence from a genuinely unpaired
cohort.

The volume interface keeps three concepts separate:

``brain_mask``
    Voxels included by the registered brain mask.
``visible_mask``
    Which in-mask voxels are exposed to the online encoder for this view.
``reliability``
    Observation/QC confidence, independent of whether a voxel is visible.

No product of these fields is stored implicitly. A downstream model must make
any weighting rule explicit.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch

from ..utils import atomic_write_json, digest_object, read_jsonl, write_jsonl


OBSERVATION_SCHEMA_VERSION = 1
LINK_SCHEMA_VERSION = 1
VOLUME_CONTRACT_VERSION = "brain-mask-zscore-single-channel-v1"
VALID_MODALITIES = ("mri", "pet")
VALID_SPLITS = ("train", "validation", "test")
RELATION_ORDER = ("mri->mri", "pet->pet", "mri->pet", "pet->mri")
UID_NAMESPACE = uuid.UUID("996de111-b24c-4cf2-947d-dfbed40ccf69")


class ObservationValidationError(RuntimeError):
    """Raised when observation, link, split, or volume metadata is invalid."""


def stable_observation_uid(kind: str, *parts: object) -> str:
    """Return a stable identifier without relying on row or filesystem order."""

    value = "|".join([str(kind), *(str(part) for part in parts)])
    return str(uuid.uuid5(UID_NAMESPACE, value))


def _path(value: Any, *, base: Path | None = None) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ObservationValidationError("A required path is empty")
    result = Path(text)
    if base is not None and not result.is_absolute():
        result = base / result
    # Do not call resolve(): legacy server paths may intentionally be imported
    # on a Windows workstation where they do not exist.
    return result


def _optional_path(value: Any, *, base: Path | None = None) -> Path | None:
    return None if not str(value or "").strip() else _path(value, base=base)


def _finite_probability(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ObservationValidationError(f"{name} must be numeric, got {value!r}") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ObservationValidationError(f"{name} must be finite and in [0, 1], got {number}")
    return number


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ObservationValidationError(f"{name} must be an explicit boolean, got {value!r}")


def _legacy_acquisition_metadata(
    row: Mapping[str, Any], modality: str
) -> tuple[dict[str, Any], bool]:
    """Import acquisition values only when the legacy row states them explicitly.

    Provenance, registration QC, quantification notes and other audit metadata
    are deliberately not evidence that acquisition parameters are available.
    """

    key = f"{modality}_acquisition_metadata"
    raw_values = row.get(key) or {}
    if not isinstance(raw_values, Mapping):
        raise ObservationValidationError(f"{key} must be a mapping, got {raw_values!r}")
    values = dict(raw_values)
    availability_key = f"{modality}_acquisition_metadata_available"
    raw_available = row.get(availability_key)
    available = (
        bool(values)
        if raw_available is None or str(raw_available).strip() == ""
        else _strict_bool(raw_available, name=availability_key)
    )
    if available and not values:
        raise ObservationValidationError(
            f"{availability_key}=true but {key} contains no acquisition values"
        )
    return values, available


@dataclass(frozen=True)
class Observation:
    """One available, model-space MRI or PET volume.

    ``acquisition_uid`` preserves raw-acquisition provenance; ``observation_uid``
    identifies this particular model-ready derivative and preprocessing definition.
    """

    observation_uid: str
    subject_id: str
    session_id: str
    split: str
    modality: str
    volume_path: Path
    brain_mask_path: Path
    dataset_id: str = ""
    acquisition_uid: str = ""
    source_path: Path | None = None
    preprocessing_record_path: Path | None = None
    reliability: float = 1.0
    intensity_preprocessing: str = VOLUME_CONTRACT_VERSION
    spatial_reference: str = "common_model_space"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "observation_uid": self.observation_uid,
            "acquisition_uid": self.acquisition_uid,
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "split": self.split,
            "modality": self.modality,
            "volume_path": str(self.volume_path),
            "brain_mask_path": str(self.brain_mask_path),
            "source_path": str(self.source_path) if self.source_path is not None else "",
            "preprocessing_record_path": (
                str(self.preprocessing_record_path)
                if self.preprocessing_record_path is not None
                else ""
            ),
            "reliability": float(self.reliability),
            "intensity_preprocessing": self.intensity_preprocessing,
            "spatial_reference": self.spatial_reference,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, base: Path | None = None) -> "Observation":
        modality = str(row.get("modality", "")).lower()
        acquisition_uid = str(row.get("acquisition_uid") or row.get("acq_uid") or "")
        dataset_id = str(row.get("dataset_id", ""))
        volume_value = row.get("volume_path") or row.get("path")
        observation_uid = str(row.get("observation_uid") or "")
        if not observation_uid:
            observation_uid = stable_observation_uid(
                "observation",
                dataset_id,
                acquisition_uid or row.get("subject_id", ""),
                row.get("session_id", ""),
                modality,
                str(volume_value or ""),
            )
        return cls(
            observation_uid=observation_uid,
            acquisition_uid=acquisition_uid,
            dataset_id=dataset_id,
            subject_id=str(row.get("subject_id", "")),
            session_id=str(row.get("session_id", "")),
            split=str(row.get("split", "")).lower(),
            modality=modality,
            volume_path=_path(volume_value, base=base),
            brain_mask_path=_path(
                row.get("brain_mask_path") or row.get("mask_path"), base=base
            ),
            source_path=_optional_path(row.get("source_path"), base=base),
            preprocessing_record_path=_optional_path(
                row.get("preprocessing_record_path") or row.get("receipt_path"),
                base=base,
            ),
            reliability=_finite_probability(row.get("reliability", 1.0), name="reliability"),
            intensity_preprocessing=str(
                row.get("intensity_preprocessing")
                or row.get("intensity_semantics", VOLUME_CONTRACT_VERSION)
            ),
            spatial_reference=str(
                row.get("spatial_reference")
                or row.get("spatial_semantics", "common_model_space")
            ),
            metadata=dict(row.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CrossModalLink:
    """An independently audited MRI/PET correspondence record."""

    link_uid: str
    subject_id: str
    split: str
    mri_observation_uid: str
    pet_observation_uid: str
    verified: bool
    verification_method: str
    reliability: float = 1.0
    verification_scope: str = "subject_session_identity"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LINK_SCHEMA_VERSION,
            "link_uid": self.link_uid,
            "subject_id": self.subject_id,
            "split": self.split,
            "mri_observation_uid": self.mri_observation_uid,
            "pet_observation_uid": self.pet_observation_uid,
            "verified": bool(self.verified),
            "verification_method": self.verification_method,
            "verification_scope": self.verification_scope,
            "reliability": float(self.reliability),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "CrossModalLink":
        mri_uid = str(row.get("mri_observation_uid", ""))
        pet_uid = str(row.get("pet_observation_uid", ""))
        link_uid = str(row.get("link_uid") or row.get("pair_uid") or "")
        if not link_uid:
            link_uid = stable_observation_uid("link", mri_uid, pet_uid)
        return cls(
            link_uid=link_uid,
            subject_id=str(row.get("subject_id", "")),
            split=str(row.get("split", "")).lower(),
            mri_observation_uid=mri_uid,
            pet_observation_uid=pet_uid,
            verified=_strict_bool(row.get("verified", False), name="verified"),
            verification_method=str(row.get("verification_method", "")),
            verification_scope=str(
                row.get("verification_scope", "subject_session_identity")
            ),
            reliability=_finite_probability(
                row.get("reliability", 1.0), name="link reliability"
            ),
            metadata=dict(row.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ObservationTables:
    """Validated observation and independent link tables."""

    observations: tuple[Observation, ...]
    links: tuple[CrossModalLink, ...]
    subject_splits: Mapping[str, str]

    def __post_init__(self) -> None:
        validate_observation_tables(self)

    @property
    def observations_by_uid(self) -> dict[str, Observation]:
        return {item.observation_uid: item for item in self.observations}

    def observations_for_split(self, split: str) -> tuple[Observation, ...]:
        key = str(split).lower()
        return tuple(item for item in self.observations if item.split == key)

    def links_for_split(
        self,
        split: str,
        *,
        verified_only: bool = True,
    ) -> tuple[CrossModalLink, ...]:
        key = str(split).lower()
        return tuple(
            item
            for item in self.links
            if item.split == key and (item.verified or not verified_only)
        )

    def contract_digest(self) -> str:
        return digest_object(
            {
                "schema": [OBSERVATION_SCHEMA_VERSION, LINK_SCHEMA_VERSION],
                "subject_splits": dict(sorted(self.subject_splits.items())),
                "observations": [item.to_dict() for item in self.observations],
                "links": [item.to_dict() for item in self.links],
            }
        )


def validate_observation_tables(tables: ObservationTables) -> None:
    """Fail closed on leakage or ambiguous observation/link identities."""

    if not tables.observations:
        raise ObservationValidationError("Observation table is empty")
    split_map = {
        str(subject): str(split).lower()
        for subject, split in tables.subject_splits.items()
    }
    invalid_splits = sorted(set(split_map.values()) - set(VALID_SPLITS))
    if invalid_splits:
        raise ObservationValidationError(f"Unsupported subject splits: {invalid_splits}")

    by_uid: dict[str, Observation] = {}
    acquisition_identities: set[tuple[str, str]] = set()
    for observation in tables.observations:
        if not observation.observation_uid or observation.observation_uid in by_uid:
            raise ObservationValidationError(
                f"Duplicate or empty observation_uid: {observation.observation_uid!r}"
            )
        if not observation.subject_id:
            raise ObservationValidationError(
                f"Observation {observation.observation_uid} has no subject_id"
            )
        if observation.modality not in VALID_MODALITIES:
            raise ObservationValidationError(
                f"Observation {observation.observation_uid} has unsupported modality "
                f"{observation.modality!r}"
            )
        acquisition_available = observation.metadata.get(
            "acquisition_metadata_available", False
        )
        if not isinstance(acquisition_available, bool):
            raise ObservationValidationError(
                f"Observation {observation.observation_uid} acquisition metadata "
                "availability must be an explicit boolean"
            )
        acquisition_metadata = observation.metadata.get("acquisition_metadata", {})
        if not isinstance(acquisition_metadata, Mapping):
            raise ObservationValidationError(
                f"Observation {observation.observation_uid} acquisition_metadata "
                "must be a mapping"
            )
        if acquisition_available and not acquisition_metadata:
            raise ObservationValidationError(
                f"Observation {observation.observation_uid} claims acquisition "
                "metadata availability but provides no values"
            )
        expected_split = split_map.get(observation.subject_id)
        if expected_split is None:
            raise ObservationValidationError(
                f"Subject {observation.subject_id} was not assigned a split before observations"
            )
        if observation.split != expected_split:
            raise ObservationValidationError(
                f"Subject {observation.subject_id} leaks/conflicts across splits: "
                f"observation={observation.split}, subject_table={expected_split}"
            )
        _finite_probability(observation.reliability, name="observation reliability")
        if observation.acquisition_uid:
            identity = (observation.acquisition_uid, observation.modality)
            if identity in acquisition_identities:
                raise ObservationValidationError(
                    f"Duplicate acquisition/modality identity: {identity}"
                )
            acquisition_identities.add(identity)
        by_uid[observation.observation_uid] = observation

    observed_subjects = {item.subject_id for item in tables.observations}
    unused_split_subjects = sorted(set(split_map) - observed_subjects)
    if unused_split_subjects:
        raise ObservationValidationError(
            "Subject split table contains subjects with no available observation: "
            f"{unused_split_subjects}"
        )

    seen_links: set[str] = set()
    seen_endpoints: set[tuple[str, str]] = set()
    for link in tables.links:
        if not link.link_uid or link.link_uid in seen_links:
            raise ObservationValidationError(f"Duplicate or empty link_uid: {link.link_uid!r}")
        endpoints = (link.mri_observation_uid, link.pet_observation_uid)
        if endpoints in seen_endpoints:
            raise ObservationValidationError(f"Duplicate MRI/PET link endpoints: {endpoints}")
        try:
            mri = by_uid[link.mri_observation_uid]
            pet = by_uid[link.pet_observation_uid]
        except KeyError as error:
            raise ObservationValidationError(
                f"Link {link.link_uid} references an unknown observation: {error.args[0]}"
            ) from error
        if mri.modality != "mri" or pet.modality != "pet":
            raise ObservationValidationError(
                f"Link {link.link_uid} endpoints are not MRI then PET: "
                f"{mri.modality}, {pet.modality}"
            )
        if mri.subject_id != pet.subject_id or link.subject_id != mri.subject_id:
            raise ObservationValidationError(
                f"Link {link.link_uid} crosses subjects: "
                f"link={link.subject_id}, MRI={mri.subject_id}, PET={pet.subject_id}"
            )
        expected_split = split_map[mri.subject_id]
        if mri.split != pet.split or link.split != expected_split:
            raise ObservationValidationError(
                f"Link {link.link_uid} crosses/conflicts across splits: "
                f"link={link.split}, MRI={mri.split}, PET={pet.split}"
            )
        if link.verified and not link.verification_method.strip():
            raise ObservationValidationError(
                f"Verified link {link.link_uid} needs an explicit verification_method"
            )
        _finite_probability(link.reliability, name="link reliability")
        seen_links.add(link.link_uid)
        seen_endpoints.add(endpoints)


def build_observation_tables(
    observations: Sequence[Observation | Mapping[str, Any]],
    links: Sequence[CrossModalLink | Mapping[str, Any]],
    subject_splits: Mapping[str, str],
) -> ObservationTables:
    """Assign subject splits first, then construct/validate observation links.

    A row may include a split as an audit assertion.  It may not override the
    subject-level split table.
    """

    split_map = {
        str(subject): str(split).lower() for subject, split in subject_splits.items()
    }
    parsed_observations: list[Observation] = []
    for value in observations:
        item = value if isinstance(value, Observation) else Observation.from_mapping(value)
        assigned = split_map.get(item.subject_id)
        if assigned is None:
            raise ObservationValidationError(
                f"Subject {item.subject_id} must be split before observation construction"
            )
        if item.split and item.split != assigned:
            raise ObservationValidationError(
                f"Observation {item.observation_uid} asserts split {item.split}, "
                f"expected {assigned}"
            )
        parsed_observations.append(replace(item, split=assigned))

    parsed_links: list[CrossModalLink] = []
    for value in links:
        item = value if isinstance(value, CrossModalLink) else CrossModalLink.from_mapping(value)
        assigned = split_map.get(item.subject_id)
        if assigned is None:
            raise ObservationValidationError(
                f"Subject {item.subject_id} must be split before link construction"
            )
        if item.split and item.split != assigned:
            raise ObservationValidationError(
                f"Link {item.link_uid} asserts split {item.split}, expected {assigned}"
            )
        parsed_links.append(replace(item, split=assigned))

    return ObservationTables(
        observations=tuple(
            sorted(
                parsed_observations,
                key=lambda item: (
                    item.subject_id,
                    VALID_MODALITIES.index(item.modality),
                    item.session_id,
                    item.observation_uid,
                ),
            )
        ),
        links=tuple(sorted(parsed_links, key=lambda item: item.link_uid)),
        subject_splits=dict(sorted(split_map.items())),
    )


@dataclass(frozen=True)
class ObservationTableFiles:
    observations_path: Path
    links_path: Path
    splits_path: Path
    summary_path: Path

    @property
    def outputs(self) -> list[Path]:
        return [self.observations_path, self.links_path, self.splits_path, self.summary_path]


def write_observation_tables(
    tables: ObservationTables,
    output_dir: str | Path,
) -> ObservationTableFiles:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = ObservationTableFiles(
        observations_path=root / "observations.jsonl",
        links_path=root / "verified_links.jsonl",
        splits_path=root / "subject_splits.jsonl",
        summary_path=root / "observation_summary.json",
    )
    write_jsonl(paths.observations_path, (item.to_dict() for item in tables.observations))
    write_jsonl(paths.links_path, (item.to_dict() for item in tables.links))
    write_jsonl(
        paths.splits_path,
        (
            {"subject_id": subject, "split": split}
            for subject, split in sorted(tables.subject_splits.items())
        ),
    )
    modality_counts = {
        modality: sum(item.modality == modality for item in tables.observations)
        for modality in VALID_MODALITIES
    }
    acquisition_metadata_available_counts = {
        modality: sum(
            item.modality == modality
            and item.metadata.get("acquisition_metadata_available", False) is True
            for item in tables.observations
        )
        for modality in VALID_MODALITIES
    }
    atomic_write_json(
        paths.summary_path,
        {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "contract_digest": tables.contract_digest(),
            "observation_count": len(tables.observations),
            "verified_link_count": sum(item.verified for item in tables.links),
            "unverified_link_count": sum(not item.verified for item in tables.links),
            "subject_count": len(tables.subject_splits),
            "modality_counts": modality_counts,
            "acquisition_metadata_available_counts": (
                acquisition_metadata_available_counts
            ),
            "singleton_subject_count": sum(
                len(
                    {
                        item.modality
                        for item in tables.observations
                        if item.subject_id == subject
                    }
                )
                == 1
                for subject in tables.subject_splits
            ),
        },
    )
    return paths


def load_observation_tables(
    observations_path: str | Path,
    links_path: str | Path,
    splits_path: str | Path,
) -> ObservationTables:
    observation_source = Path(observations_path)
    observations = tuple(
        Observation.from_mapping(row, base=observation_source.parent)
        for row in read_jsonl(observation_source)
    )
    links = tuple(CrossModalLink.from_mapping(row) for row in read_jsonl(links_path))
    split_rows = read_jsonl(splits_path)
    split_map: dict[str, str] = {}
    for row in split_rows:
        subject = str(row.get("subject_id", ""))
        split = str(row.get("split", "")).lower()
        previous = split_map.setdefault(subject, split)
        if previous != split:
            raise ObservationValidationError(
                f"Subject {subject} appears in multiple split rows: {previous}, {split}"
            )
    return build_observation_tables(observations, links, split_map)


def _legacy_rows(
    source: str | Path | Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], Path | None]:
    if source is None:
        return [], None
    if isinstance(source, (str, Path)):
        path = Path(source)
        return read_jsonl(path), path.parent
    return [dict(row) for row in source], None


def _legacy_split_map(
    pairs: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Freeze the subject split map before any observation/link construction.

    When a separate frozen split manifest is supplied, its subjects define the
    cohort allowlist.  Legacy pair manifests may retain rejected, unused reserve,
    or otherwise non-analysis rows for audit provenance; those rows must not be
    assigned a new split implicitly.  Without a separate manifest, every pair
    row must continue to carry its own valid split.
    """

    mapping: dict[str, str] = {}
    source = split_rows if split_rows else pairs
    for row in source:
        subject = str(row.get("subject_id", ""))
        split = str(row.get("split", "")).lower()
        if not subject or split not in VALID_SPLITS:
            raise ObservationValidationError(
                f"Legacy split row needs subject_id and one of {VALID_SPLITS}: {row}"
            )
        previous = mapping.setdefault(subject, split)
        if previous != split:
            raise ObservationValidationError(
                f"Legacy subject {subject} leaks across splits: {previous}, {split}"
            )
    pair_subjects: set[str] = set()
    for row in pairs:
        subject = str(row.get("subject_id", ""))
        if not subject:
            raise ObservationValidationError(
                f"Legacy pair row needs a nonempty subject_id: {row}"
            )
        pair_subjects.add(subject)
        asserted = str(row.get("split", "")).lower()
        if subject not in mapping:
            if split_rows:
                # A separate frozen split manifest is also the analysis-cohort
                # allowlist.  Preserve out-of-cohort pair rows in the legacy
                # source manifest, but do not import or silently re-split them.
                continue
            raise ObservationValidationError(f"Legacy subject {subject} has no frozen split")
        if asserted and asserted != mapping[subject]:
            raise ObservationValidationError(
                f"Legacy pair split conflicts for {subject}: {asserted}, {mapping[subject]}"
            )
    missing_pair_subjects = sorted(set(mapping) - pair_subjects)
    if missing_pair_subjects:
        raise ObservationValidationError(
            "Frozen split subjects have no legacy pair rows: "
            f"{missing_pair_subjects}"
        )
    return mapping


def import_legacy_pair_pet128(
    pair_manifest: str | Path | Sequence[Mapping[str, Any]],
    pet128_manifest: str | Path | Sequence[Mapping[str, Any]] | None = None,
    *,
    split_manifest: str | Path | Sequence[Mapping[str, Any]] | None = None,
) -> ObservationTables:
    """Import the old pair-first and optional PET128 manifests without requiring pairing.

    Available MRI and PET derivatives become independent observations.  A PET
    derivative is considered cross-modal-ready only when the PET128 manifest
    supplies a common-space model input and support mask.  Missing derivatives
    therefore create legal singleton observations instead of failing the whole
    cohort.

    When ``split_manifest`` is supplied, it is the frozen cohort allowlist as
    well as the immutable subject-level split assignment.  Audit-only pair rows
    outside that allowlist are not observations and are never assigned a split.

    Legacy ``mri_pet_eligible`` verifies subject/session identity, not manual
    registration quality.  That restricted scope and the legacy QC status are
    retained explicitly on the imported link.
    """

    pair_rows, pair_base = _legacy_rows(pair_manifest)
    pet_rows, pet_base = _legacy_rows(pet128_manifest)
    split_rows, _ = _legacy_rows(split_manifest)
    if not pair_rows:
        raise ObservationValidationError("Legacy pair manifest is empty")
    split_map = _legacy_split_map(pair_rows, split_rows)

    selected_pair_rows = [
        row for row in pair_rows if str(row.get("subject_id", "")) in split_map
    ]
    if not selected_pair_rows:
        raise ObservationValidationError(
            "Frozen split manifest selected no rows from the legacy pair manifest"
        )

    all_pair_uids: set[str] = set()
    for row in pair_rows:
        pair_uid = str(row.get("pair_uid", ""))
        if not pair_uid or pair_uid in all_pair_uids:
            raise ObservationValidationError(
                f"Legacy pair rows need unique nonempty pair_uid: {pair_uid!r}"
            )
        all_pair_uids.add(pair_uid)
    selected_pair_uids = {
        str(row.get("pair_uid", "")) for row in selected_pair_rows
    }

    all_pet_by_pair: dict[str, dict[str, Any]] = {}
    for row in pet_rows:
        pair_uid = str(row.get("pair_uid", ""))
        if not pair_uid or pair_uid in all_pet_by_pair:
            raise ObservationValidationError(
                f"Legacy PET128 rows need unique nonempty pair_uid: {pair_uid!r}"
            )
        all_pet_by_pair[pair_uid] = row
    unmatched_pet_rows = sorted(set(all_pet_by_pair) - all_pair_uids)
    if unmatched_pet_rows:
        raise ObservationValidationError(
            f"PET128 rows have no corresponding legacy pair: {unmatched_pet_rows}"
        )
    pet_by_pair = {
        pair_uid: row
        for pair_uid, row in all_pet_by_pair.items()
        if pair_uid in selected_pair_uids
    }

    observations: list[Observation] = []
    links: list[CrossModalLink] = []
    for pair in sorted(
        selected_pair_rows, key=lambda row: str(row.get("pair_uid", ""))
    ):
        pair_uid = str(pair.get("pair_uid", ""))
        subject = str(pair.get("subject_id", ""))
        session = str(pair.get("session_id", ""))
        dataset_id = str(pair.get("dataset_id", ""))
        split = split_map[subject]
        merged = dict(pair)
        pet_row = pet_by_pair.get(pair_uid)
        if pet_row is not None:
            pet_subject = str(pet_row.get("subject_id") or subject)
            pet_session = str(pet_row.get("session_id") or session)
            if pet_subject != subject or pet_session != session:
                raise ObservationValidationError(
                    f"PET128 identity conflicts for pair {pair_uid}: "
                    f"pair={subject}/{session}, PET128={pet_subject}/{pet_session}"
                )
            pet_asserted_split = str(pet_row.get("split", "")).lower()
            if pet_asserted_split and pet_asserted_split != split:
                raise ObservationValidationError(
                    f"PET128 split conflicts for {subject}: "
                    f"{pet_asserted_split}, {split}"
                )
            merged.update(pet_row)
        legacy_receipts = dict(merged.get("legacy_receipt_paths") or {})

        mri_observation: Observation | None = None
        mri_volume = pair.get("mri_path") or merged.get("mri_path")
        mri_mask = pair.get("mri_mask_path") or merged.get("mri_mask_path")
        if mri_volume and mri_mask:
            mri_acq = str(merged.get("mri_acq_uid", ""))
            mri_acquisition_metadata, mri_acquisition_available = (
                _legacy_acquisition_metadata(merged, "mri")
            )
            mri_observation = Observation(
                observation_uid=stable_observation_uid(
                    "legacy-observation", dataset_id, mri_acq or pair_uid, "mri"
                ),
                acquisition_uid=mri_acq,
                dataset_id=dataset_id,
                subject_id=subject,
                session_id=session,
                split=split,
                modality="mri",
                volume_path=_path(mri_volume, base=pair_base or pet_base),
                brain_mask_path=_path(mri_mask, base=pair_base or pet_base),
                source_path=_optional_path(merged.get("mri_raw_path"), base=pair_base),
                preprocessing_record_path=_optional_path(legacy_receipts.get("mri"), base=pair_base),
                reliability=_finite_probability(
                    merged.get("mri_reliability", 1.0), name="MRI reliability"
                ),
                intensity_preprocessing=VOLUME_CONTRACT_VERSION,
                spatial_reference="common_model_space",
                metadata={
                    "legacy_pair_uid": pair_uid,
                    "legacy_analysis_status": merged.get("legacy_analysis_status", ""),
                    "acquisition_metadata": mri_acquisition_metadata,
                    "acquisition_metadata_available": mri_acquisition_available,
                },
            )
            observations.append(mri_observation)

        pet_observation: Observation | None = None
        # The already-z-scored SAT3D input is preferred; re-zscoring it inside
        # ObservationVolumeCache is idempotent up to floating-point tolerance.
        pet_volume = merged.get("pet_sat3d_input_path") or merged.get("pet_relative_128_path")
        pet_specific_mask = (
            (pet_row or {}).get("pet_brain_mask_path")
            or (pet_row or {}).get("reference_mask_path")
        )
        pet_mask = pet_specific_mask or pair.get("mri_mask_path") or merged.get("mri_mask_path")
        pet_mask_base = (pet_base or pair_base) if pet_specific_mask else (pair_base or pet_base)
        if pet_volume and pet_mask:
            pet_acq = str(merged.get("pet_acq_uid", ""))
            pet_acquisition_metadata, pet_acquisition_available = (
                _legacy_acquisition_metadata(merged, "pet")
            )
            pet_observation = Observation(
                observation_uid=stable_observation_uid(
                    "legacy-observation", dataset_id, pet_acq or pair_uid, "pet"
                ),
                acquisition_uid=pet_acq,
                dataset_id=dataset_id,
                subject_id=subject,
                session_id=session,
                split=split,
                modality="pet",
                volume_path=_path(pet_volume, base=pet_base or pair_base),
                brain_mask_path=_path(pet_mask, base=pet_mask_base),
                source_path=_optional_path(
                    merged.get("pet_motion_path") or merged.get("pet_raw_path"), base=pair_base
                ),
                preprocessing_record_path=_optional_path(
                    merged.get("pet128_receipt_path") or legacy_receipts.get("pet_registration"),
                    base=pet_base or pair_base,
                ),
                reliability=_finite_probability(
                    merged.get("pet_reliability", 1.0), name="PET reliability"
                ),
                intensity_preprocessing=str(
                    merged.get("pet_intensity_preprocessing")
                    or merged.get("pet_intensity_semantics", "relative_pet_brain_mask_zscore")
                ),
                spatial_reference="common_model_space",
                metadata={
                    "legacy_pair_uid": pair_uid,
                    "pet_quantification": dict(merged.get("pet_quantification") or {}),
                    "acquisition_metadata": pet_acquisition_metadata,
                    "acquisition_metadata_available": pet_acquisition_available,
                    "support_provenance": (
                        "legacy_pet_or_reference_mask"
                        if merged.get("pet_brain_mask_path") or merged.get("reference_mask_path")
                        else "legacy_mri_mask_shared_in_common_space"
                    ),
                },
            )
            observations.append(pet_observation)

        if mri_observation is not None and pet_observation is not None:
            verified = _strict_bool(
                merged.get("mri_pet_eligible", False), name="mri_pet_eligible"
            )
            links.append(
                CrossModalLink(
                    link_uid=pair_uid
                    or stable_observation_uid(
                        "legacy-link",
                        mri_observation.observation_uid,
                        pet_observation.observation_uid,
                    ),
                    subject_id=subject,
                    split=split,
                    mri_observation_uid=mri_observation.observation_uid,
                    pet_observation_uid=pet_observation.observation_uid,
                    verified=verified,
                    verification_method=str(
                        merged.get("pairing_method") or "legacy_pair_manifest"
                    ),
                    verification_scope="subject_session_identity_only",
                    reliability=_finite_probability(
                        merged.get("pair_reliability", 1.0), name="pair reliability"
                    ),
                    metadata={
                        "registration_qc_status": merged.get(
                            "registration_qc_status", "not_recorded"
                        ),
                        "t1_pet_delta_hours": merged.get("t1_pet_delta_hours"),
                        "not_a_manual_registration_qc_claim": True,
                    },
                )
            )

    if not observations:
        raise ObservationValidationError(
            "Legacy manifests contained no model-ready MRI or PET observations"
        )
    return build_observation_tables(observations, links, split_map)


@dataclass(frozen=True)
class ObservationVolume:
    """Normalized single-channel image and its separate input masks."""

    observation: Observation
    volume: torch.Tensor
    brain_mask: torch.Tensor
    visible_mask: torch.Tensor
    reliability: torch.Tensor
    normalization_mean: float
    normalization_std: float
    voxel_volume_mm3: float

    def __post_init__(self) -> None:
        shapes = {
            tuple(self.volume.shape),
            tuple(self.brain_mask.shape),
            tuple(self.visible_mask.shape),
            tuple(self.reliability.shape),
        }
        if len(shapes) != 1 or self.volume.ndim != 4 or self.volume.shape[0] != 1:
            raise ObservationValidationError(
                f"Observation tensors must share [1,X,Y,Z], got {sorted(shapes)}"
            )


def _single_channel_array(image: nib.spatialimages.SpatialImage, *, name: str) -> np.ndarray:
    array = np.asarray(image.dataobj, dtype=np.float32)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ObservationValidationError(
            f"{name} must be a static 3-D volume (or singleton fourth axis), got {array.shape}"
        )
    return array


def _field_tensor(
    value: np.ndarray | torch.Tensor | float,
    *,
    shape: tuple[int, int, int],
    name: str,
    support: torch.Tensor,
    require_within_support: bool,
) -> torch.Tensor:
    if isinstance(value, (int, float)):
        tensor = torch.full((1, *shape), float(value), dtype=torch.float32)
        if require_within_support:
            tensor = tensor * support
    elif torch.is_tensor(value):
        tensor = value.detach().to(dtype=torch.float32, device="cpu")
    else:
        tensor = torch.from_numpy(np.asarray(value, dtype=np.float32))
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != (1, *shape):
        raise ObservationValidationError(
            f"{name} must have shape {shape} or {(1, *shape)}, got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ObservationValidationError(f"{name} contains non-finite values")
    if bool(((tensor < 0.0) | (tensor > 1.0)).any()):
        raise ObservationValidationError(f"{name} must be in [0, 1]")
    if require_within_support and bool((tensor > support + 1.0e-6).any()):
        raise ObservationValidationError(f"{name} marks voxels outside the brain mask")
    return tensor.contiguous()


class ObservationVolumeCache:
    """Process-local LRU of immutable normalized observation volumes.

    NIfTI I/O and brain-mask z-scoring occur only on the first access to an
    observation.  The visible mask may vary online without invalidating the cached
    volume or conflating it with the brain mask or reliability.
    """

    def __init__(
        self,
        observations: Sequence[Observation],
        *,
        max_items: int = 16,
        minimum_support_voxels: int = 2,
        standard_deviation_epsilon: float = 1.0e-6,
    ) -> None:
        self.observations = {item.observation_uid: item for item in observations}
        if len(self.observations) != len(observations):
            raise ObservationValidationError("Volume cache received duplicate observation UIDs")
        self.max_items = max(1, int(max_items))
        self.minimum_support_voxels = max(2, int(minimum_support_voxels))
        self.standard_deviation_epsilon = float(standard_deviation_epsilon)
        self._items: OrderedDict[str, ObservationVolume] = OrderedDict()
        self.load_count = 0

    def _load(self, observation: Observation) -> ObservationVolume:
        if not observation.volume_path.is_file():
            raise ObservationValidationError(
                f"Observation volume does not exist: {observation.volume_path}"
            )
        if not observation.brain_mask_path.is_file():
            raise ObservationValidationError(
                f"Observation brain mask does not exist: {observation.brain_mask_path}"
            )
        try:
            image = nib.load(str(observation.volume_path), mmap=True)
            mask_image = nib.load(str(observation.brain_mask_path), mmap=True)
        except Exception as error:
            raise ObservationValidationError(
                f"Cannot load observation {observation.observation_uid}: {error}"
            ) from error
        array = _single_channel_array(image, name="observation volume")
        mask_array = _single_channel_array(mask_image, name="brain mask")
        if array.shape != mask_array.shape:
            raise ObservationValidationError(
                f"Volume/mask shape mismatch for {observation.observation_uid}: "
                f"{array.shape}, {mask_array.shape}"
            )
        if not np.allclose(image.affine, mask_image.affine, atol=1.0e-5, rtol=0.0):
            raise ObservationValidationError(
                f"Volume/mask affine mismatch for {observation.observation_uid}"
            )
        if not np.isfinite(mask_array).all():
            raise ObservationValidationError(
                f"Brain mask contains non-finite values for {observation.observation_uid}"
            )
        support_np = mask_array > 0.5
        if int(support_np.sum()) < self.minimum_support_voxels:
            raise ObservationValidationError(
                f"Brain mask for {observation.observation_uid} has only "
                f"{int(support_np.sum())} supported voxels"
            )
        if not np.isfinite(array[support_np]).all():
            raise ObservationValidationError(
                f"Supported volume contains non-finite values for {observation.observation_uid}"
            )
        values = array[support_np].astype(np.float64, copy=False)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not math.isfinite(std) or std <= self.standard_deviation_epsilon:
            raise ObservationValidationError(
                f"Supported intensity standard deviation is degenerate for "
                f"{observation.observation_uid}: {std}"
            )
        normalized = np.zeros(array.shape, dtype=np.float32)
        normalized[support_np] = ((values - mean) / std).astype(np.float32)
        # Non-finite values outside support are deliberately ignored and stay 0.
        support = torch.from_numpy(support_np.astype(np.float32)).unsqueeze(0).contiguous()
        volume = torch.from_numpy(normalized).unsqueeze(0).contiguous()
        voxel_volume = float(abs(np.linalg.det(np.asarray(image.affine)[:3, :3])))
        if not math.isfinite(voxel_volume) or voxel_volume <= 0.0:
            raise ObservationValidationError(
                f"Invalid physical voxel volume for {observation.observation_uid}: {voxel_volume}"
            )
        reliability = support * float(observation.reliability)
        self.load_count += 1
        return ObservationVolume(
            observation=observation,
            volume=volume,
            brain_mask=support,
            visible_mask=support.clone(),
            reliability=reliability.contiguous(),
            normalization_mean=mean,
            normalization_std=std,
            voxel_volume_mm3=voxel_volume,
        )

    def get(
        self,
        observation_uid: str,
        *,
        visible_mask: np.ndarray | torch.Tensor | float | None = None,
        reliability: np.ndarray | torch.Tensor | float | None = None,
    ) -> ObservationVolume:
        uid = str(observation_uid)
        if uid in self._items:
            base = self._items.pop(uid)
            self._items[uid] = base
        else:
            try:
                observation = self.observations[uid]
            except KeyError as error:
                raise ObservationValidationError(f"Unknown observation_uid: {uid}") from error
            base = self._load(observation)
            self._items[uid] = base
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

        shape = tuple(int(value) for value in base.volume.shape[1:])
        visible = (
            base.visible_mask
            if visible_mask is None
            else _field_tensor(
                visible_mask,
                shape=shape,
                name="visible_mask",
                support=base.brain_mask,
                require_within_support=True,
            )
        )
        reliable = (
            base.reliability
            if reliability is None
            else _field_tensor(
                reliability,
                shape=shape,
                name="reliability",
                support=base.brain_mask,
                require_within_support=True,
            )
        )

        return replace(
            base,
            visible_mask=visible,
            reliability=reliable,
        )

    def clear(self) -> None:
        self._items.clear()

    def contract_digest(self) -> str:
        return digest_object(
            {
                "version": VOLUME_CONTRACT_VERSION,
                "minimum_support_voxels": self.minimum_support_voxels,
                "standard_deviation_epsilon": self.standard_deviation_epsilon,
                "observations": [
                    item.to_dict()
                    for item in sorted(
                        self.observations.values(), key=lambda value: value.observation_uid
                    )
                ],
            }
        )


@dataclass(frozen=True)
class RelationExample:
    source_observation_uid: str
    target_observation_uid: str
    source_modality: str
    target_modality: str
    relation: str
    subject_id: str
    target_subject_id: str
    split: str
    link_uid: str | None
    link_reliability: float
    pairing_type: str
    is_pairing_control: bool
    source_link_uid: str | None = None
    target_link_uid: str | None = None

    @property
    def is_cross_modal(self) -> bool:
        return self.source_modality != self.target_modality


def _hash_integer(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _weighted_relation_cycle(
    weights: Mapping[str, float],
    *,
    length: int,
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Build a deterministic, evenly interleaved finite weighted cycle."""

    if length < len(weights):
        raise ObservationValidationError(
            f"relation_cycle_length={length} is smaller than {len(weights)} active relations"
        )
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ObservationValidationError("Active relation weights must sum to a positive value")
    normalized = {name: value / total for name, value in weights.items()}
    # Give every active positive-weight relation one slot.  Allocate the
    # remainder by largest residual; the default 20-slot cycle is therefore
    # exactly 7/7/3/3 for 0.35/0.35/0.15/0.15.
    remaining = length - len(normalized)
    exact = {name: normalized[name] * remaining for name in normalized}
    counts = {name: 1 + int(math.floor(exact[name])) for name in normalized}
    leftovers = length - sum(counts.values())
    order_index = {name: index for index, name in enumerate(RELATION_ORDER)}
    residual_order = sorted(
        normalized,
        key=lambda name: (-(exact[name] - math.floor(exact[name])), order_index[name]),
    )
    for name in residual_order[:leftovers]:
        counts[name] += 1

    # Midpoints distribute each relation's slots across the cycle instead of
    # emitting long modality blocks.
    points: list[tuple[float, int, str]] = []
    for name in RELATION_ORDER:
        if name not in counts:
            continue
        for index in range(counts[name]):
            points.append(((index + 0.5) / counts[name], order_index[name], name))
    cycle = tuple(name for _, _, name in sorted(points))
    if len(cycle) != length:
        raise AssertionError("Internal weighted relation cycle length mismatch")
    return cycle, counts


class ObservationRelationSampler:
    """Stateless, balanced sampler over supported observation relations.

    Intra-modal examples exist for every observation, including MRI/PET
    singletons.  Cross-modal examples exist only for retained, verified links.
    Sampling depends only on the configuration, seed, optimizer step, and batch
    offset, so exact resume does not depend on mutable RNG state.
    """

    def __init__(
        self,
        tables: ObservationTables,
        split: str,
        *,
        verified_pair_retention_fraction: float = 1.0,
        pairing_mode: str = "verified",
        relation_weights: Mapping[str, float] | None = None,
        allowed_relations: Sequence[str] | None = None,
        retained_subject_order: Sequence[str] | None = None,
        relation_cycle_length: int = 20,
        seed: int = 0,
    ) -> None:
        self.tables = tables
        self.split = str(split).lower()
        if self.split not in VALID_SPLITS:
            raise ObservationValidationError(f"Unknown split: {split!r}")
        self.verified_pair_retention_fraction = _finite_probability(verified_pair_retention_fraction, name="verified_pair_retention_fraction")
        self.pairing_mode = str(pairing_mode).lower()
        if self.pairing_mode not in {"verified", "unpaired", "wrong_subject"}:
            raise ObservationValidationError(
                f"pairing_mode must be verified, unpaired, or wrong_subject; got {pairing_mode!r}"
            )
        self.seed = int(seed)
        observations = sorted(
            tables.observations_for_split(self.split), key=lambda item: item.observation_uid
        )
        if not observations:
            raise ObservationValidationError(f"No observations in split {self.split}")
        by_uid = tables.observations_by_uid

        verified_links = sorted(
            tables.links_for_split(self.split, verified_only=True), key=lambda item: item.link_uid
        )
        verified_by_subject: dict[str, list[CrossModalLink]] = {}
        for link in verified_links:
            verified_by_subject.setdefault(link.subject_id, []).append(link)
        available_subjects = tuple(sorted(verified_by_subject))
        if retained_subject_order is None:
            subject_order = tuple(
                sorted(
                    available_subjects,
                    key=lambda subject: (
                        _hash_integer("pair-subject-retention", self.seed, subject), subject
                    ),
                )
            )
            retention_order_source = "seeded_subject_hash"
        else:
            subject_order = tuple(str(subject) for subject in retained_subject_order)
            if len(subject_order) != len(set(subject_order)):
                raise ObservationValidationError("retained_subject_order contains duplicates")
            missing = sorted(set(available_subjects) - set(subject_order))
            unknown = sorted(set(subject_order) - set(available_subjects))
            if missing or unknown:
                raise ObservationValidationError(
                    "retained_subject_order must be an exact permutation of verified-link "
                    f"subjects; missing={missing}, unknown={unknown}"
                )
            retention_order_source = "explicit_frozen_subject_order"
        retain_subject_count = int(
            math.floor(self.verified_pair_retention_fraction * len(subject_order) + 0.5)
        )
        self.available_verified_subject_order = subject_order
        self.retained_subjects = subject_order[:retain_subject_count]
        retained_subject_set = set(self.retained_subjects)
        self.retained_links = tuple(
            link for link in verified_links if link.subject_id in retained_subject_set
        )
        self.retention_order_source = retention_order_source
        self.available_verified_link_count = len(verified_links)
        self.available_verified_subject_count = len(available_subjects)
        self.excluded_unverified_link_count = sum(
            not item.verified for item in tables.links_for_split(self.split, verified_only=False)
        )

        groups: dict[str, list[RelationExample]] = {name: [] for name in RELATION_ORDER}
        for observation in observations:
            relation = f"{observation.modality}->{observation.modality}"
            groups[relation].append(
                RelationExample(
                    source_observation_uid=observation.observation_uid,
                    target_observation_uid=observation.observation_uid,
                    source_modality=observation.modality,
                    target_modality=observation.modality,
                    relation=relation,
                    subject_id=observation.subject_id,
                    target_subject_id=observation.subject_id,
                    split=self.split,
                    link_uid=None,
                    link_reliability=1.0,
                    pairing_type="within_observation",
                    is_pairing_control=False,
                )
            )
        if self.pairing_mode == "verified":
            for link in self.retained_links:
                mri = by_uid[link.mri_observation_uid]
                pet = by_uid[link.pet_observation_uid]
                groups["mri->pet"].append(
                    RelationExample(
                        source_observation_uid=mri.observation_uid,
                        target_observation_uid=pet.observation_uid,
                        source_modality="mri",
                        target_modality="pet",
                        relation="mri->pet",
                        subject_id=link.subject_id,
                        target_subject_id=link.subject_id,
                        split=self.split,
                        link_uid=link.link_uid,
                        link_reliability=link.reliability,
                        pairing_type="verified_pair",
                        is_pairing_control=False,
                        source_link_uid=link.link_uid,
                        target_link_uid=link.link_uid,
                    )
                )
                groups["pet->mri"].append(
                    RelationExample(
                        source_observation_uid=pet.observation_uid,
                        target_observation_uid=mri.observation_uid,
                        source_modality="pet",
                        target_modality="mri",
                        relation="pet->mri",
                        subject_id=link.subject_id,
                        target_subject_id=link.subject_id,
                        split=self.split,
                        link_uid=link.link_uid,
                        link_reliability=link.reliability,
                        pairing_type="verified_pair",
                        is_pairing_control=False,
                        source_link_uid=link.link_uid,
                        target_link_uid=link.link_uid,
                    )
                )
        elif self.pairing_mode == "wrong_subject" and self.retained_links:
            links_by_subject = {
                subject: [
                    link for link in self.retained_links if link.subject_id == subject
                ]
                for subject in self.retained_subjects
            }
            ambiguous = {
                subject: len(values)
                for subject, values in links_by_subject.items()
                if len(values) != 1
            }
            if ambiguous:
                raise ObservationValidationError(
                    "pairing_mode=wrong_subject requires exactly one retained verified link per subject; "
                    f"counts={ambiguous}"
                )
            if len(self.retained_subjects) < 2:
                raise ObservationValidationError(
                    "pairing_mode=wrong_subject needs at least two retained verified subjects "
                    "for a derangement"
                )
            source_links = [links_by_subject[subject][0] for subject in self.retained_subjects]
            target_links = source_links[1:] + source_links[:1]
            for source_link, target_link in zip(source_links, target_links, strict=True):
                source_mri = by_uid[source_link.mri_observation_uid]
                source_pet = by_uid[source_link.pet_observation_uid]
                target_mri = by_uid[target_link.mri_observation_uid]
                target_pet = by_uid[target_link.pet_observation_uid]
                control_reliability = min(source_link.reliability, target_link.reliability)
                groups["mri->pet"].append(
                    RelationExample(
                        source_observation_uid=source_mri.observation_uid,
                        target_observation_uid=target_pet.observation_uid,
                        source_modality="mri",
                        target_modality="pet",
                        relation="mri->pet",
                        subject_id=source_link.subject_id,
                        target_subject_id=target_link.subject_id,
                        split=self.split,
                        link_uid=None,
                        link_reliability=control_reliability,
                        pairing_type="mismatched_subject_pair",
                        is_pairing_control=True,
                        source_link_uid=source_link.link_uid,
                        target_link_uid=target_link.link_uid,
                    )
                )
                groups["pet->mri"].append(
                    RelationExample(
                        source_observation_uid=source_pet.observation_uid,
                        target_observation_uid=target_mri.observation_uid,
                        source_modality="pet",
                        target_modality="mri",
                        relation="pet->mri",
                        subject_id=source_link.subject_id,
                        target_subject_id=target_link.subject_id,
                        split=self.split,
                        link_uid=None,
                        link_reliability=control_reliability,
                        pairing_type="mismatched_subject_pair",
                        is_pairing_control=True,
                        source_link_uid=source_link.link_uid,
                        target_link_uid=target_link.link_uid,
                    )
                )
        self.groups = {
            name: tuple(
                sorted(
                    values,
                    key=lambda item: (
                        item.subject_id,
                        item.target_subject_id,
                        item.source_observation_uid,
                    ),
                )
            )
            for name, values in groups.items()
            if values
        }
        if allowed_relations is None:
            allowed = RELATION_ORDER
        else:
            allowed = tuple(str(name).lower() for name in allowed_relations)
            if len(allowed) != len(set(allowed)):
                raise ObservationValidationError("allowed_relations contains duplicates")
            invalid_allowed = sorted(set(allowed) - set(RELATION_ORDER))
            if invalid_allowed:
                raise ObservationValidationError(
                    f"Unsupported allowed_relations: {invalid_allowed}"
                )
        requested_weights = (
            {"mri->mri": 0.35, "pet->pet": 0.35, "mri->pet": 0.15, "pet->mri": 0.15}
            if relation_weights is None
            else {str(name).lower(): float(value) for name, value in relation_weights.items()}
        )
        invalid_weight_names = sorted(set(requested_weights) - set(RELATION_ORDER))
        if invalid_weight_names:
            raise ObservationValidationError(
                f"Unsupported relation weight names: {invalid_weight_names}"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in requested_weights.values()):
            raise ObservationValidationError("relation_weights must be finite and non-negative")
        active_weights = {
            name: requested_weights.get(name, 0.0)
            for name in RELATION_ORDER
            if name in self.groups and name in allowed and requested_weights.get(name, 0.0) > 0.0
        }
        if not active_weights:
            raise ObservationValidationError(
                "No supported positive-weight relations remain after applying availability and "
                "allowed_relations"
            )
        self.requested_relation_weights = dict(requested_weights)
        self.allowed_relations = tuple(allowed)
        self.relation_cycle, self.relation_cycle_counts = _weighted_relation_cycle(
            active_weights, length=int(relation_cycle_length)
        )
        self.active_relations = tuple(name for name in RELATION_ORDER if name in active_weights)
        self.effective_relation_weights = {
            name: self.relation_cycle_counts[name] / len(self.relation_cycle)
            for name in self.active_relations
        }
        self.supported_examples = tuple(
            example for name in self.active_relations for example in self.groups[name]
        )

    def sample(self, step: int, batch_size: int) -> tuple[RelationExample, ...]:
        if int(step) < 0:
            raise ObservationValidationError("step must be non-negative")
        if int(batch_size) <= 0:
            raise ObservationValidationError("batch_size must be positive")
        examples: list[RelationExample] = []
        base = int(step) * int(batch_size)
        for offset in range(int(batch_size)):
            relation = self.relation_cycle[(base + offset) % len(self.relation_cycle)]
            choices = self.groups[relation]
            index = _hash_integer(
                "relation-example", self.seed, int(step), offset, relation
            ) % len(choices)
            examples.append(choices[index])
        return tuple(examples)

    def sampling_summary(self) -> dict[str, Any]:
        if self.pairing_mode == "wrong_subject":
            pairing_setup = "mismatched_subject_pairing_control"
        elif self.pairing_mode == "unpaired":
            pairing_setup = "cross_modal_relations_disabled"
        elif self.verified_pair_retention_fraction < 1.0:
            pairing_setup = "verified_pair_subsampling"
        else:
            pairing_setup = "verified_pairs"
        return {
            "pairing_setup": pairing_setup,
            "pairing_mode": self.pairing_mode,
            "verified_pair_retention_fraction": self.verified_pair_retention_fraction,
            "available_verified_subject_count": self.available_verified_subject_count,
            "available_verified_subject_order": list(self.available_verified_subject_order),
            "retained_verified_subject_count": len(self.retained_subjects),
            "retained_subjects": list(self.retained_subjects),
            "retention_order_source": self.retention_order_source,
            "available_verified_link_count": self.available_verified_link_count,
            "retained_verified_link_count": len(self.retained_links),
            "deleted_verified_link_count": (
                self.available_verified_link_count - len(self.retained_links)
            ),
            "excluded_unverified_link_count": self.excluded_unverified_link_count,
            "supports_within_modality_singletons": True,
            "verified_links_modified": False,
            "wrong_subject_pairs_written_to_manifest": False,
            "mismatched_pair_generation": (
                "deterministic_cyclic_subject_derangement"
                if self.pairing_mode == "wrong_subject" and self.retained_links
                else "not_applicable"
            ),
            "allowed_relations": list(self.allowed_relations),
            "requested_relation_weights": self.requested_relation_weights,
            "effective_relation_weights": self.effective_relation_weights,
            "relation_cycle": list(self.relation_cycle),
            "relation_cycle_counts": self.relation_cycle_counts,
            "seed": self.seed,
        }

    def contract_digest(self) -> str:
        return digest_object(
            {
                "observation_contract": self.tables.contract_digest(),
                "split": self.split,
                "seed": self.seed,
                "sampling_summary": self.sampling_summary(),
                "supported_examples": [example.__dict__ for example in self.supported_examples],
            }
        )


def relation_counts(examples: Iterable[RelationExample]) -> dict[str, int]:
    counts = {name: 0 for name in RELATION_ORDER}
    for example in examples:
        counts[example.relation] += 1
    return counts
