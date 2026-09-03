"""Streaming physical inventory for FOMO ZIP archives and ADNI DICOM series."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator

from .schema import FormalDataError, digest_payload


_FOMO_ZIP = re.compile(
    r"^FOMO_MRI/(?P<dataset>PT\d+_[^/]+)/(?P<subject>sub-[^/]+)/"
    r"(?P<session>ses-[^/]+)\.zip$",
    re.IGNORECASE,
)
_ADNI_IMAGE = re.compile(
    r"^ADNI_PET/(?P<family>[^/]+)/(?P<subject>[^/]+)/(?P<series>[^/]+)/"
    r"(?P<date>[^/]+)/I(?P<image_id>\d+)(?:/(?P<file>[^/]+\.dcm))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InventoryEntry:
    relative_path: str
    path: str
    file_count: int
    signature: str
    series_label: str = ""
    family: str = ""
    subject_id: str = ""


class PhysicalInventory:
    """Only materialized archive/series entries; cache paths are never indexed."""

    def __init__(
        self,
        *,
        fomo_zips: dict[tuple[str, str, str], InventoryEntry] | None = None,
        fomo_files: dict[tuple[str, str], InventoryEntry] | None = None,
        adni_series: dict[str, InventoryEntry] | None = None,
        source: str = "",
    ) -> None:
        self.fomo_zips = dict(fomo_zips or {})
        self.fomo_files = dict(fomo_files or {})
        self.adni_series = dict(adni_series or {})
        self.source = source

    def fomo(self, dataset_id: str, participant_id: str, session_id: str) -> InventoryEntry | None:
        return self.fomo_zips.get((dataset_id, participant_id, session_id))

    def fomo_file(self, dataset_id: str, mapping_new_path: str) -> InventoryEntry | None:
        normalized = _normalize_mapping_path(mapping_new_path)
        return self.fomo_files.get((dataset_id, normalized))

    def adni(self, image_id: str | int) -> InventoryEntry | None:
        return self.adni_series.get(str(image_id).lstrip("I"))

    def summary(self) -> dict[str, object]:
        return {
            "source": self.source,
            "fomo_archive_count": len(self.fomo_zips),
            "fomo_extracted_file_count": len(self.fomo_files),
            "adni_image_series_count": len(self.adni_series),
            "adni_dicom_file_count": sum(row.file_count for row in self.adni_series.values()),
        }

    @classmethod
    def from_hierarchy(
        cls,
        hierarchy_path: str | Path,
        *,
        fomo_root: str | Path,
        adni_root: str | Path,
    ) -> "PhysicalInventory":
        """Parse the 200 MB hierarchy in one streaming pass with bounded memory."""

        hierarchy = Path(hierarchy_path)
        if not hierarchy.is_file():
            raise FormalDataError(f"Hierarchy file does not exist: {hierarchy}")
        fomo_base = Path(fomo_root)
        adni_base = Path(adni_root)
        fomo: dict[tuple[str, str, str], InventoryEntry] = {}
        fomo_files: dict[tuple[str, str], InventoryEntry] = {}
        adni_work: dict[str, InventoryEntry] = {}
        with hierarchy.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for raw_line in handle:
                relative = raw_line.strip().replace("\\", "/")
                if not relative or "/.cache/" in f"/{relative}/":
                    continue
                fomo_match = _FOMO_ZIP.match(relative)
                if fomo_match is not None:
                    dataset_id = fomo_match.group("dataset")
                    if dataset_id.upper().startswith("PT030_"):
                        continue
                    local_relative = relative.removeprefix("FOMO_MRI/")
                    key = (
                        dataset_id,
                        fomo_match.group("subject"),
                        fomo_match.group("session"),
                    )
                    fomo[key] = InventoryEntry(
                        relative_path=relative,
                        path=str(fomo_base / Path(local_relative)),
                        file_count=1,
                        signature=digest_payload(["hierarchy", relative]),
                    )
                    continue
                extracted = _parse_extracted_fomo(relative)
                if extracted is not None:
                    dataset_id, mapping_path, local_relative = extracted
                    if dataset_id.upper().startswith("PT030_"):
                        continue
                    fomo_files[(dataset_id, mapping_path)] = InventoryEntry(
                        relative_path=relative,
                        path=str(fomo_base / Path(local_relative)),
                        file_count=1,
                        signature=digest_payload(["hierarchy", relative]),
                    )
                    continue
                adni_match = _ADNI_IMAGE.match(relative)
                if adni_match is None:
                    continue
                image_id = adni_match.group("image_id")
                image_relative = relative.split(f"/I{image_id}", maxsplit=1)[0] + f"/I{image_id}"
                local_relative = image_relative.removeprefix("ADNI_PET/")
                previous = adni_work.get(image_id)
                file_count = (previous.file_count if previous else 0) + int(
                    adni_match.group("file") is not None
                )
                adni_work[image_id] = InventoryEntry(
                    relative_path=image_relative,
                    path=str(adni_base / Path(local_relative)),
                    file_count=file_count,
                    signature="",
                    series_label=adni_match.group("series"),
                    family=adni_match.group("family"),
                    subject_id=adni_match.group("subject"),
                )
        adni = {
            image_id: replace(
                row,
                signature=digest_payload(
                    ["hierarchy", row.relative_path, row.file_count, row.series_label]
                ),
            )
            for image_id, row in adni_work.items()
            if row.file_count > 0
        }
        return cls(
            fomo_zips=fomo,
            fomo_files=fomo_files,
            adni_series=adni,
            source=str(hierarchy),
        )

    @classmethod
    def from_filesystem(
        cls,
        *,
        fomo_root: str | Path,
        adni_root: str | Path,
    ) -> "PhysicalInventory":
        """Walk a live workstation tree, again excluding any ``.cache`` subtree."""

        fomo_base = Path(fomo_root)
        adni_base = Path(adni_root)
        if not fomo_base.is_dir() or not adni_base.is_dir():
            raise FormalDataError(
                f"Live data roots must exist: fomo={fomo_base}, adni={adni_base}"
            )
        fomo: dict[tuple[str, str, str], InventoryEntry] = {}
        for archive in sorted(fomo_base.rglob("*.zip")):
            if ".cache" in archive.parts:
                continue
            relative_parts = archive.relative_to(fomo_base).parts
            dataset_index = next(
                (index for index, part in enumerate(relative_parts) if part.startswith("PT")),
                -1,
            )
            subject_index = next(
                (index for index, part in enumerate(relative_parts) if part.startswith("sub-")),
                -1,
            )
            if dataset_index < 0 or subject_index < 0:
                continue
            dataset_id = relative_parts[dataset_index]
            participant_id = relative_parts[subject_index]
            filename = archive.name
            if dataset_id.upper().startswith("PT030_"):
                continue
            session_id = Path(filename).stem
            stat = archive.stat()
            relative = (Path("FOMO_MRI") / archive.relative_to(fomo_base)).as_posix()
            fomo[(dataset_id, participant_id, session_id)] = InventoryEntry(
                relative_path=relative,
                path=str(archive),
                file_count=1,
                signature=digest_payload([relative, stat.st_size, stat.st_mtime_ns]),
            )
        fomo_files: dict[tuple[str, str], InventoryEntry] = {}
        for nifti in sorted([*fomo_base.rglob("*.nii"), *fomo_base.rglob("*.nii.gz")]):
            if ".cache" in nifti.parts:
                continue
            relative_local = nifti.relative_to(fomo_base).as_posix()
            parsed = _parse_extracted_fomo(f"FOMO_MRI/{relative_local}")
            if parsed is None:
                continue
            dataset_id, mapping_path, _ = parsed
            if dataset_id.upper().startswith("PT030_"):
                continue
            stat = nifti.stat()
            key = (dataset_id, mapping_path)
            if key in fomo_files:
                raise FormalDataError(
                    f"Two extracted FOMO files resolve to {dataset_id}/{mapping_path}"
                )
            fomo_files[key] = InventoryEntry(
                relative_path=f"FOMO_MRI/{relative_local}",
                path=str(nifti),
                file_count=1,
                signature=digest_payload(
                    ["filesystem", relative_local, stat.st_size, stat.st_mtime_ns]
                ),
            )
        grouped: dict[Path, int] = {}
        for dicom in adni_base.rglob("*.dcm"):
            if ".cache" in dicom.parts:
                continue
            parent = dicom.parent
            if re.fullmatch(r"I\d+", parent.name, flags=re.IGNORECASE):
                grouped[parent] = grouped.get(parent, 0) + 1
        adni: dict[str, InventoryEntry] = {}
        for directory, count in sorted(grouped.items(), key=lambda item: item[0].as_posix()):
            image_id = directory.name.lstrip("Ii")
            relative_local = directory.relative_to(adni_base)
            parts = relative_local.parts
            if len(parts) < 5:
                continue
            relative = (Path("ADNI_PET") / relative_local).as_posix()
            stat = directory.stat()
            entry = InventoryEntry(
                relative_path=relative,
                path=str(directory),
                file_count=count,
                signature=digest_payload([relative, count, stat.st_mtime_ns]),
                family=parts[0],
                subject_id=parts[1],
                series_label=parts[2],
            )
            if image_id in adni:
                raise FormalDataError(f"ADNI Image ID I{image_id} appears in two directories")
            adni[image_id] = entry
        return cls(
            fomo_zips=fomo,
            fomo_files=fomo_files,
            adni_series=adni,
            source=f"filesystem:{fomo_base}|{adni_base}",
        )


def _normalize_mapping_path(value: str) -> str:
    parts = [part for part in value.replace("\\", "/").split("/") if part and part != "."]
    return "/".join(parts)


def _parse_extracted_fomo(relative: str) -> tuple[str, str, str] | None:
    """Return dataset, mapping.new_path and root-local path.

    Some fully extracted trees contain ``sub-X/ses-Y/ses-Y/...``.  The second
    identical session wrapper is a storage artifact and is removed only for the
    exact ``mapping.new_path`` lookup key.
    """

    normalized = relative.replace("\\", "/").strip("/")
    if not normalized.lower().endswith((".nii", ".nii.gz")):
        return None
    parts = normalized.split("/")
    try:
        root_index = parts.index("FOMO_MRI")
    except ValueError:
        return None
    if root_index + 2 >= len(parts):
        return None
    dataset_id = parts[root_index + 1]
    local_parts = parts[root_index + 1 :]
    mapping_parts = parts[root_index + 2 :]
    subject_index = next(
        (index for index, part in enumerate(mapping_parts) if part.startswith("sub-")),
        -1,
    )
    if subject_index < 0:
        return None
    mapping_parts = mapping_parts[subject_index:]
    if (
        len(mapping_parts) >= 3
        and mapping_parts[1].startswith("ses-")
        and mapping_parts[2] == mapping_parts[1]
    ):
        mapping_parts = [mapping_parts[0], mapping_parts[1], *mapping_parts[3:]]
    return dataset_id, _normalize_mapping_path("/".join(mapping_parts)), "/".join(local_parts)


def iter_hierarchy_paths(path: str | Path) -> Iterator[str]:
    """Small public helper for diagnostics without retaining the full tree."""

    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            value = raw_line.strip().replace("\\", "/")
            if value and "/.cache/" not in f"/{value}/":
                yield value
