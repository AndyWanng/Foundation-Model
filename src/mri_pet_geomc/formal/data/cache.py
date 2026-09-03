"""Deterministic float16 image shards and bit-packed support masks.

Shard and slot assignment depends only on the sorted observation UIDs.  A
preprocessing restart therefore writes the same case to the same mmap slot and
can trust an atomic per-case receipt without rebuilding earlier shards.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ...utils import atomic_write_json, digest_object, read_json, write_jsonl
from .schema import CatalogObservation, FormalDataError


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CacheEntry:
    observation_uid: str
    shard_id: int
    slot: int
    image_path: str
    support_path: str
    shape: tuple[int, int, int]
    dtype: str = "float16"
    contract_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "observation_uid": self.observation_uid,
            "shard_id": int(self.shard_id),
            "slot": int(self.slot),
            "image_path": self.image_path,
            "support_path": self.support_path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "contract_digest": self.contract_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheEntry":
        return cls(
            observation_uid=str(value["observation_uid"]),
            shard_id=int(value["shard_id"]),
            slot=int(value["slot"]),
            image_path=str(value["image_path"]),
            support_path=str(value["support_path"]),
            shape=tuple(int(item) for item in value["shape"]),  # type: ignore[arg-type]
            dtype=str(value.get("dtype", "float16")),
            contract_digest=str(value.get("contract_digest", "")),
        )


@dataclass(frozen=True)
class CacheValue:
    image: np.ndarray
    support: np.ndarray


class ShardStore:
    """Preallocated deterministic mmap store used by offline preprocessing."""

    def __init__(
        self,
        root: str | Path,
        observations: Sequence[CatalogObservation],
        *,
        shape: tuple[int, int, int],
        shard_size: int = 128,
        contract_digest: str,
    ) -> None:
        if shard_size < 1:
            raise FormalDataError("shard_size must be positive")
        self.root = Path(root)
        self.shard_root = self.root / "shards"
        self.shard_root.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(int(value) for value in shape)
        if len(self.shape) != 3 or any(value <= 0 for value in self.shape):
            raise FormalDataError(f"Invalid cache shape: {shape}")
        self.shard_size = int(shard_size)
        self.contract_digest = contract_digest
        ordered = sorted(observations, key=lambda row: row.observation_uid)
        if len({row.observation_uid for row in ordered}) != len(ordered):
            raise FormalDataError("Cannot shard duplicate observation UIDs")
        self._entries: dict[str, CacheEntry] = {}
        self._shard_counts: dict[int, int] = {}
        for index, row in enumerate(ordered):
            shard_id, slot = divmod(index, self.shard_size)
            self._shard_counts[shard_id] = self._shard_counts.get(shard_id, 0) + 1
            self._entries[row.observation_uid] = CacheEntry(
                observation_uid=row.observation_uid,
                shard_id=shard_id,
                slot=slot,
                image_path=f"shards/shard-{shard_id:05d}.images.npy",
                support_path=f"shards/shard-{shard_id:05d}.support.npy",
                shape=self.shape,
                contract_digest=contract_digest,
            )
        self.plan_digest = digest_object(
            {
                "schema": CACHE_SCHEMA_VERSION,
                "observation_uids": [row.observation_uid for row in ordered],
                "shape": self.shape,
                "shard_size": self.shard_size,
                "contract_digest": contract_digest,
            }
        )
        self._ensure_layout()

    @property
    def entries(self) -> Mapping[str, CacheEntry]:
        return self._entries

    def entry(self, observation_uid: str) -> CacheEntry:
        try:
            return self._entries[observation_uid]
        except KeyError as error:
            raise FormalDataError(f"Observation not present in shard plan: {observation_uid}") from error

    def _ensure_array(
        self, path: Path, *, dtype: np.dtype[Any], shape: tuple[int, ...]
    ) -> None:
        if not path.exists():
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            array[...] = 0
            array.flush()
            del array
            return
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise FormalDataError(
                f"Existing shard layout mismatch at {path}: "
                f"shape={array.shape}/{shape}, dtype={array.dtype}/{dtype}"
            )
        del array

    def _ensure_layout(self) -> None:
        packed_voxels = (int(np.prod(self.shape)) + 7) // 8
        for shard_id, count in self._shard_counts.items():
            image_path = self.root / f"shards/shard-{shard_id:05d}.images.npy"
            support_path = self.root / f"shards/shard-{shard_id:05d}.support.npy"
            self._ensure_array(
                image_path,
                dtype=np.dtype(np.float16),
                shape=(count, 1, *self.shape),
            )
            self._ensure_array(
                support_path,
                dtype=np.dtype(np.uint8),
                shape=(count, packed_voxels),
            )
        atomic_write_json(
            self.root / "shard_plan.json",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "plan_digest": self.plan_digest,
                "contract_digest": self.contract_digest,
                "shape": list(self.shape),
                "shard_size": self.shard_size,
                "observation_count": len(self._entries),
                "shard_counts": {str(key): value for key, value in self._shard_counts.items()},
            },
        )

    def write(self, observation_uid: str, image: np.ndarray, support: np.ndarray) -> CacheEntry:
        entry = self.entry(observation_uid)
        values = np.asarray(image)
        if values.shape == self.shape:
            values = values[np.newaxis]
        if values.shape != (1, *self.shape):
            raise FormalDataError(
                f"Image {observation_uid} has shape {values.shape}, expected {(1, *self.shape)}"
            )
        support_values = np.asarray(support, dtype=bool)
        if support_values.shape != self.shape:
            raise FormalDataError(
                f"Support {observation_uid} has shape {support_values.shape}, expected {self.shape}"
            )
        if not np.isfinite(values).all():
            raise FormalDataError(f"Image {observation_uid} contains non-finite values")
        packed = np.packbits(support_values.reshape(-1), bitorder="little")
        images = np.load(self.root / entry.image_path, mmap_mode="r+", allow_pickle=False)
        supports = np.load(self.root / entry.support_path, mmap_mode="r+", allow_pickle=False)
        stored = values.astype(np.float16, copy=False)
        if not np.isfinite(stored).all():
            raise FormalDataError(
                f"Image {observation_uid} overflows or becomes non-finite in float16"
            )
        images[entry.slot] = stored
        supports[entry.slot] = packed
        images.flush()
        supports.flush()
        del images, supports
        return entry

    def freeze_manifest(
        self,
        receipt_dir: str | Path,
        *,
        manifest_path: str | Path | None = None,
    ) -> tuple[CacheEntry, ...]:
        receipts = Path(receipt_dir)
        complete: list[CacheEntry] = []
        for uid, entry in sorted(self._entries.items()):
            path = receipts / f"{uid}.json"
            if not path.is_file():
                continue
            value = read_json(path)
            if (
                isinstance(value, Mapping)
                and value.get("status") == "complete"
                and value.get("observation_uid") == uid
                and value.get("contract_digest") == self.contract_digest
                and value.get("plan_digest") == self.plan_digest
                and int(value.get("shard_id", -1)) == entry.shard_id
                and int(value.get("slot", -1)) == entry.slot
            ):
                complete.append(entry)
        destination = Path(manifest_path) if manifest_path else self.root / "cache_index.jsonl"
        write_jsonl(destination, (entry.to_dict() for entry in complete))
        atomic_write_json(
            self.root / "cache_summary.json",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "plan_digest": self.plan_digest,
                "contract_digest": self.contract_digest,
                "planned_count": len(self._entries),
                "complete_count": len(complete),
                "failed_or_pending_count": len(self._entries) - len(complete),
                "manifest": str(destination),
            },
        )
        return tuple(complete)


def load_cache_index(path: str | Path) -> tuple[CacheEntry, ...]:
    rows: list[CacheEntry] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FormalDataError(f"Expected object at {path}:{number}")
            rows.append(CacheEntry.from_dict(value))
    if len({row.observation_uid for row in rows}) != len(rows):
        raise FormalDataError("Cache index contains duplicate observation UIDs")
    return tuple(rows)


class MMapCache:
    """Worker-local LRU mmap reader.  No full shard is copied into RAM."""

    def __init__(
        self,
        cache_root: str | Path,
        entries: Sequence[CacheEntry] | str | Path,
        *,
        max_open_shards: int = 2,
    ) -> None:
        self.root = Path(cache_root)
        loaded = load_cache_index(entries) if isinstance(entries, (str, Path)) else tuple(entries)
        self.entries = {entry.observation_uid: entry for entry in loaded}
        self.max_open_shards = max(1, int(max_open_shards))
        self._handles: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        return state

    def close(self) -> None:
        while self._handles:
            _, handles = self._handles.popitem(last=False)
            for array in handles:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()

    def _open(self, entry: CacheEntry) -> tuple[np.ndarray, np.ndarray]:
        if entry.shard_id in self._handles:
            handles = self._handles.pop(entry.shard_id)
            self._handles[entry.shard_id] = handles
            return handles
        images = np.load(self.root / entry.image_path, mmap_mode="r", allow_pickle=False)
        supports = np.load(self.root / entry.support_path, mmap_mode="r", allow_pickle=False)
        handles = (images, supports)
        self._handles[entry.shard_id] = handles
        while len(self._handles) > self.max_open_shards:
            _, evicted = self._handles.popitem(last=False)
            for array in evicted:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        return handles

    def get(self, observation_uid: str) -> CacheValue:
        try:
            entry = self.entries[observation_uid]
        except KeyError as error:
            raise KeyError(f"Observation is not in frozen cache: {observation_uid}") from error
        images, supports = self._open(entry)
        image = np.array(images[entry.slot], dtype=np.float16, copy=True)
        packed = np.asarray(supports[entry.slot])
        voxel_count = int(np.prod(entry.shape))
        support = np.unpackbits(packed, bitorder="little", count=voxel_count).reshape(entry.shape)
        return CacheValue(image=image, support=support.astype(bool, copy=False))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
