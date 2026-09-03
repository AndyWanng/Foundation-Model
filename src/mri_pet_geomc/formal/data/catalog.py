"""Build the formal FOMO-MRI + ADNI-PET catalog by physical inner join."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ...utils import atomic_write_json, digest_object, write_jsonl
from .inventory import PhysicalInventory
from .schema import (
    CatalogObservation,
    FormalDataError,
    SourceLocator,
    SubjectSplit,
    stable_uid,
    validate_catalog,
)


FOMO_DATASETS = (
    "PT001_ClevelandCCF",
    "PT005_ADHD_200",
    "PT014_CoRR",
    "PT015_MSD_BrainTumor",
    "PT021_IXI",
    "PT028_OASIS1",
)
PET_MANIFESTS = {
    "Amyloid": "Fully_Processed_Amyloid_PET_Manifest_26Jul2026.csv",
    "FDG": "Fully_Processed_FDG_PET_Manifest_26Jul2026.csv",
    "Tau": "Fully_Processed_Tau_PET_Manifest_26Jul2026.csv",
}
KNOWN_TRACERS = ("NAV4694", "MK6240", "PI2620", "AV1451", "AV45", "FBB", "PIB", "FDG")
ACQUISITION_COLUMNS = (
    "Modality",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturersModelName",
    "SoftwareVersions",
    "MRAcquisitionType",
    "SeriesDescription",
    "ProtocolName",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "SequenceName",
    "EchoTime",
    "SliceThickness",
    "RepetitionTime",
    "InversionTime",
    "FlipAngle",
)
ProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _emit(callback: ProgressCallback | None, event: str, **payload: Any) -> None:
    if callback is not None:
        callback(event, payload)


def _read_table(path: Path, delimiter: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FormalDataError(f"Required metadata table does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter=delimiter)
        ]


def _fomo_metadata_base(metadata_root: Path) -> Path:
    candidates = (metadata_root / "MRI", metadata_root / "fomo", metadata_root / "FOMO")
    for candidate in candidates:
        if all((candidate / dataset_id / "mapping.tsv").is_file() for dataset_id in FOMO_DATASETS):
            return candidate
    raise FormalDataError(
        "Could not locate all six FOMO metadata folders under MRI/, fomo/ or FOMO/"
    )


def _adni_manifest_base(metadata_root: Path) -> Path:
    candidates = (
        metadata_root / "PET" / "ADNI" / "manifest",
        metadata_root / "adni_pet",
        metadata_root / "ADNI_PET",
    )
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in PET_MANIFESTS.values()):
            return candidate
    raise FormalDataError(
        "Could not locate all ADNI PET manifests under PET/ADNI/manifest/, "
        "adni_pet/ or ADNI_PET/"
    )


def _source_session(old_path: str, fallback: str) -> tuple[str, float | None]:
    match = re.search(r"(?:^|/)(?:session|ses)[_-]?(\d+)(?:/|$)", old_path, re.IGNORECASE)
    if match is None:
        match = re.search(r"(\d+)$", fallback)
    if match is None:
        return fallback, None
    number = int(match.group(1))
    return f"source-session-{number:02d}", float(number)


def _sequence_product(new_filename: str, old_path: str, old_filename: str) -> tuple[str, str, bool]:
    text = new_filename.removesuffix(".gz").removesuffix(".nii")
    lower_text = text.lower()
    dwi_match = re.search(r"_dwi(?:_(bval\d+))?$", lower_text)
    suffix = text.rsplit("_", maxsplit=1)[-1].lower()
    aliases = {
        "t1w": "T1w",
        "t2w": "T2w",
        "pdw": "PDw",
        "flair": "FLAIR",
        "dwi": "DWI",
        "bold": "BOLD",
        "asl": "ASL",
        "cbf": "CBF",
        "mp2rage": "MP2RAGE",
        "swi": "SWI",
    }
    old = f"{old_path}/{old_filename}".lower()
    if dwi_match is not None:
        sequence = "DWI"
        b_value = dwi_match.group(1)
        if b_value:
            product = b_value.upper()
        elif "trace" in old:
            product = "TRACE"
        else:
            product = "DWI"
    else:
        sequence = aliases.get(suffix, suffix.upper() or "UNKNOWN")
        product = sequence
    if sequence == "MP2RAGE":
        if "inv1" in old:
            product = "INV1"
        elif "inv2" in old:
            product = "INV2"
        elif "uni" in old:
            product = "UNI"
        else:
            product = "COMPOSITE"
    return sequence, product, sequence in {"CBF"}


def _infer_tracer(family: str, *texts: str) -> str:
    combined = " ".join(texts).upper().replace("-", "").replace("_", "")
    for tracer in KNOWN_TRACERS:
        if tracer in combined:
            return tracer
    return "FDG" if family.upper() == "FDG" else "UNKNOWN"


def _parse_iso_date(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return text


def _fomo_observations(
    metadata_root: Path,
    inventory: PhysicalInventory,
    callback: ProgressCallback | None,
) -> tuple[list[CatalogObservation], Counter[str]]:
    rows: list[CatalogObservation] = []
    skipped: Counter[str] = Counter()
    fomo_base = _fomo_metadata_base(metadata_root)
    for dataset_id in FOMO_DATASETS:
        dataset_root = fomo_base / dataset_id
        mappings = _read_table(dataset_root / "mapping.tsv", "\t")
        info_rows = _read_table(dataset_root / "mri_info.tsv", "\t")
        info_by_filename: dict[str, dict[str, str]] = {}
        for info in info_rows:
            filename = info.get("filename", "").replace("\\", "/")
            if filename and filename not in info_by_filename:
                info_by_filename[filename] = info
        for mapping in mappings:
            participant = mapping.get("participant_id", "")
            session = mapping.get("session_id", "")
            new_path = mapping.get("new_path", "").replace("\\", "/")
            new_filename = mapping.get("new_filename", "") or Path(new_path).name
            old_path = mapping.get("old_path", "").replace("\\", "/")
            old_filename = mapping.get("old_filename", "")
            if not all([participant, session, new_path, new_filename]):
                skipped["fomo_incomplete_metadata"] += 1
                continue
            extracted = inventory.fomo_file(dataset_id, new_path)
            physical = extracted or inventory.fomo(dataset_id, participant, session)
            if physical is None:
                skipped["fomo_source_not_materialized"] += 1
                continue
            source_session, session_order = _source_session(old_path, session)
            sequence, product, derived = _sequence_product(new_filename, old_path, old_filename)
            info = info_by_filename.get(new_path, {})
            acquisition_metadata = {
                key: info[key]
                for key in ACQUISITION_COLUMNS
                if str(info.get(key, "")).strip()
            }
            canonical_subject = f"FOMO:{dataset_id}:{participant}"
            acquisition_uid = stable_uid(
                "fomo-acquisition", dataset_id, old_path or new_path, new_filename
            )
            rows.append(
                CatalogObservation(
                    observation_uid=stable_uid(
                        "fomo-observation", dataset_id, participant, source_session, new_path
                    ),
                    dataset_id=dataset_id,
                    canonical_subject_id=canonical_subject,
                    participant_id=participant,
                    modality="mri",
                    session_id=session,
                    source_session_id=source_session,
                    session_order=session_order,
                    acquisition_uid=acquisition_uid,
                    source=SourceLocator(
                        kind="nifti" if extracted is not None else "zip_member",
                        path=physical.path,
                        relative_path=physical.relative_path,
                        archive_member="" if extracted is not None else new_path,
                        file_count=1,
                        signature=physical.signature,
                    ),
                    sequence=sequence,
                    product=product,
                    derived=derived,
                    acquisition_metadata=acquisition_metadata,
                )
            )
        _emit(callback, "catalog_dataset", dataset_id=dataset_id, materialized=len(rows))
    return rows, skipped


def _adni_observations(
    metadata_root: Path,
    inventory: PhysicalInventory,
    callback: ProgressCallback | None,
) -> tuple[list[CatalogObservation], Counter[str]]:
    rows: list[CatalogObservation] = []
    skipped: Counter[str] = Counter()
    seen_images: set[str] = set()
    manifest_root = _adni_manifest_base(metadata_root)
    for family, filename in PET_MANIFESTS.items():
        manifest = _read_table(manifest_root / filename, ",")
        family_count = 0
        for raw in manifest:
            image_id = raw.get("image_id", "").lstrip("I")
            participant = raw.get("subject_id", "")
            if not image_id or not participant:
                skipped["adni_incomplete_metadata"] += 1
                continue
            physical = inventory.adni(image_id)
            if physical is None:
                skipped[f"adni_{family.lower()}_manifest_only"] += 1
                continue
            if image_id in seen_images:
                raise FormalDataError(f"ADNI Image ID I{image_id} occurs more than once")
            seen_images.add(image_id)
            if physical.subject_id and physical.subject_id != participant:
                raise FormalDataError(
                    f"ADNI I{image_id} subject mismatch: manifest={participant}, "
                    f"physical={physical.subject_id}"
                )
            study_date = _parse_iso_date(raw.get("image_date", ""))
            order: float | None = None
            if study_date:
                try:
                    order = float(date.fromisoformat(study_date).toordinal())
                except ValueError:
                    pass
            visit = raw.get("image_visit", "") or study_date or f"I{image_id}"
            tracer = _infer_tracer(
                family,
                physical.series_label,
                raw.get("image_description", ""),
            )
            canonical_subject = f"ADNI:{participant}"
            rows.append(
                CatalogObservation(
                    observation_uid=stable_uid("adni-observation", image_id),
                    dataset_id="ADNI_PET",
                    canonical_subject_id=canonical_subject,
                    participant_id=participant,
                    modality="pet",
                    session_id=visit,
                    source_session_id=study_date or visit,
                    acquisition_uid=f"ADNI:I{image_id}",
                    source=SourceLocator(
                        kind="dicom_series",
                        path=physical.path,
                        relative_path=physical.relative_path,
                        file_count=physical.file_count,
                        signature=physical.signature,
                    ),
                    tracer=tracer,
                    tracer_family=family,
                    product="fully_processed",
                    study_date=study_date,
                    session_order=order,
                    acquisition_metadata={
                        key: raw[key]
                        for key in (
                            "study_id",
                            "series_id",
                            "image_visit",
                            "image_description",
                        )
                        if raw.get(key, "")
                    },
                )
            )
            family_count += 1
        _emit(callback, "catalog_dataset", dataset_id=f"ADNI_{family}", materialized=family_count)
    return rows, skipped


def _hamilton_counts(
    sizes: Mapping[str, int], total: int, ratio: float, capacities: Mapping[str, int]
) -> dict[str, int]:
    ideals = {key: sizes[key] * ratio for key in sizes}
    counts = {key: min(int(math.floor(ideals[key])), capacities[key]) for key in sizes}
    remaining = total - sum(counts.values())
    order = sorted(
        sizes,
        key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key),
    )
    while remaining > 0:
        changed = False
        for key in order:
            if counts[key] < capacities[key]:
                counts[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise FormalDataError("Unable to allocate requested subject split counts")
    return counts


def assign_subject_splits(
    subject_datasets: Mapping[str, str],
    *,
    seed: int = 260809,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> list[SubjectSplit]:
    """Exact deterministic 80/10/10 split with dataset-stratified apportionment."""

    if not subject_datasets:
        raise FormalDataError("Cannot split an empty subject table")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value < 0 for value in fractions) or not math.isclose(sum(fractions), 1.0):
        raise FormalDataError(f"Split fractions must be non-negative and sum to one: {fractions}")
    grouped: dict[str, list[str]] = defaultdict(list)
    for subject, dataset_id in subject_datasets.items():
        grouped[dataset_id].append(subject)
    sizes = {key: len(values) for key, values in grouped.items()}
    subject_count = len(subject_datasets)
    test_total = round(subject_count * test_fraction)
    validation_total = round(subject_count * validation_fraction)
    test_counts = _hamilton_counts(sizes, test_total, test_fraction, sizes)
    remaining_capacities = {key: sizes[key] - test_counts[key] for key in sizes}
    validation_counts = _hamilton_counts(
        sizes, validation_total, validation_fraction, remaining_capacities
    )
    assignments: list[SubjectSplit] = []
    for dataset_id, subjects in sorted(grouped.items()):
        ordered = sorted(
            subjects,
            key=lambda subject: hashlib.sha256(
                f"{seed}|{dataset_id}|{subject}".encode("utf-8")
            ).digest(),
        )
        test_count = test_counts[dataset_id]
        validation_count = validation_counts[dataset_id]
        for index, subject in enumerate(ordered):
            if index < test_count:
                split = "test"
            elif index < test_count + validation_count:
                split = "validation"
            else:
                split = "train"
            assignments.append(SubjectSplit(subject, dataset_id, split))
    return sorted(assignments, key=lambda row: row.canonical_subject_id)


@dataclass(frozen=True)
class CatalogBuildResult:
    observations: tuple[CatalogObservation, ...]
    subject_splits: tuple[SubjectSplit, ...]
    summary: Mapping[str, Any]

    @property
    def digest(self) -> str:
        return digest_object(
            {
                "observations": [row.to_dict() for row in self.observations],
                "subject_splits": [row.to_dict() for row in self.subject_splits],
            }
        )

    def write(self, output_dir: str | Path) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_jsonl(output / "observations.jsonl", (row.to_dict() for row in self.observations))
        write_jsonl(
            output / "subject_splits.jsonl", (row.to_dict() for row in self.subject_splits)
        )
        sources: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in self.observations:
            sources[
                (row.dataset_id, row.source.relative_path, row.source.archive_member)
            ] = {
                "dataset_id": row.dataset_id,
                "observation_uid": row.observation_uid,
                **row.source.to_dict(),
            }
        write_jsonl(output / "sources.jsonl", (sources[key] for key in sorted(sources)))
        vocab = {
            "dataset_id": sorted({row.dataset_id for row in self.observations}),
            "modality": sorted({row.modality for row in self.observations}),
            "sequence": sorted({row.sequence for row in self.observations if row.sequence}),
            "product": sorted({row.product for row in self.observations if row.product}),
            "tracer": sorted({row.tracer for row in self.observations if row.tracer}),
            "tracer_family": sorted(
                {row.tracer_family for row in self.observations if row.tracer_family}
            ),
        }
        atomic_write_json(output / "vocab.json", vocab)
        atomic_write_json(output / "catalog_summary.json", {**dict(self.summary), "digest": self.digest})
        return output


def build_catalog(
    metadata_root: str | Path,
    inventory: PhysicalInventory,
    *,
    split_seed: int = 260809,
    progress: ProgressCallback | None = None,
) -> CatalogBuildResult:
    """Build, namespace, split and validate only physically materialized cases."""

    root = Path(metadata_root)
    _emit(progress, "catalog_start", metadata_root=str(root))
    fomo, fomo_skipped = _fomo_observations(root, inventory, progress)
    adni, adni_skipped = _adni_observations(root, inventory, progress)
    raw_observations = fomo + adni
    subject_datasets: dict[str, str] = {}
    for observation in raw_observations:
        previous = subject_datasets.setdefault(
            observation.canonical_subject_id, observation.dataset_id
        )
        if previous != observation.dataset_id:
            raise FormalDataError(
                f"Subject namespace collision for {observation.canonical_subject_id}"
            )
    splits = assign_subject_splits(subject_datasets, seed=split_seed)
    split_map = {row.canonical_subject_id: row.split for row in splits}
    observations = sorted(
        (row.with_split(split_map[row.canonical_subject_id]) for row in raw_observations),
        key=lambda row: row.observation_uid,
    )
    validate_catalog(observations, splits)
    split_subject_counts = Counter(row.split for row in splits)
    summary = {
        "inventory": inventory.summary(),
        "observation_count": len(observations),
        "subject_count": len(splits),
        "modality_observation_counts": dict(Counter(row.modality for row in observations)),
        "dataset_observation_counts": dict(Counter(row.dataset_id for row in observations)),
        "split_subject_counts": dict(split_subject_counts),
        "skipped_counts": dict(fomo_skipped + adni_skipped),
        "split_seed": split_seed,
        "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
    }
    result = CatalogBuildResult(tuple(observations), tuple(splits), summary)
    _emit(progress, "catalog_complete", **summary)
    return result


def load_catalog(
    observations_path: str | Path, subject_splits_path: str | Path
) -> CatalogBuildResult:
    import json

    def read_rows(path: str | Path) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FormalDataError(f"Expected object at {path}:{number}")
                values.append(value)
        return values

    observations = tuple(CatalogObservation.from_dict(row) for row in read_rows(observations_path))
    splits = tuple(SubjectSplit.from_dict(row) for row in read_rows(subject_splits_path))
    validate_catalog(observations, splits)
    return CatalogBuildResult(observations, splits, {})
