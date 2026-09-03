"""Formal 24k FOMO-MRI and ADNI-PET data contracts."""

from .schema import (
    CatalogObservation,
    FormalDataError,
    ObservationRelation,
    SourceLocator,
    SubjectSplit,
    validate_catalog,
    validate_relations,
)
from .inventory import InventoryEntry, PhysicalInventory
from .catalog import CatalogBuildResult, assign_subject_splits, build_catalog, load_catalog
from .relations import RelationBuildResult, build_relations, load_relations
from .reference import MaterializedReference, ReferenceSpec, materialize_reference
from .cache import CacheEntry, MMapCache, ShardStore, load_cache_index
from .preprocessing import (
    OfflinePreprocessor,
    PreprocessProgress,
    PreprocessRunSummary,
    PreprocessingConfig,
)
from .sampler import CoverageSampler, RelationSchedule, RelationSelector, SampleKey
from .dataset import (
    FormalSample,
    FormalVolumeDataset,
    load_preprocessing_overlay,
    robust_normalize_visible,
)
from .collate import packed_collate

__all__ = [
    "CatalogObservation",
    "FormalDataError",
    "ObservationRelation",
    "SourceLocator",
    "SubjectSplit",
    "validate_catalog",
    "validate_relations",
    "InventoryEntry",
    "PhysicalInventory",
    "CatalogBuildResult",
    "assign_subject_splits",
    "build_catalog",
    "load_catalog",
    "RelationBuildResult",
    "build_relations",
    "load_relations",
    "MaterializedReference",
    "ReferenceSpec",
    "materialize_reference",
    "CacheEntry",
    "MMapCache",
    "ShardStore",
    "load_cache_index",
    "OfflinePreprocessor",
    "PreprocessProgress",
    "PreprocessRunSummary",
    "PreprocessingConfig",
    "CoverageSampler",
    "RelationSchedule",
    "RelationSelector",
    "SampleKey",
    "FormalSample",
    "FormalVolumeDataset",
    "load_preprocessing_overlay",
    "robust_normalize_visible",
    "packed_collate",
]
