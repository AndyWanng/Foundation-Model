"""FEM-node feature aggregation and geometry embeddings."""

from .geometry_embedding import (
    GeometryEmbedding,
    GeometryEmbeddingDiagnostics,
    GeometryFeatureType,
    basis_invariant_geometry_features,
)
from .node_features import (
    NodeFeatureAggregator,
    NodeFeatureField,
    ObservationTokenBatch,
)

__all__ = [
    "GeometryEmbedding",
    "GeometryEmbeddingDiagnostics",
    "GeometryFeatureType",
    "NodeFeatureAggregator",
    "NodeFeatureField",
    "ObservationTokenBatch",
    "basis_invariant_geometry_features",
]