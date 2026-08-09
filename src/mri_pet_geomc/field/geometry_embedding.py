"""Basis-invariant geometry embeddings for the shared FEM-node latent field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ..geometry.frame import ParsevalResolventFrame


GeometryFeatureType = Literal["none", "spectral", "spectral_xyz"]


def _mass_standardize(features: Tensor, mass: Tensor, *, eps: float) -> Tensor:
    probability = mass / mass.sum()
    mean = torch.sum(probability[:, None] * features, dim=0, keepdim=True)
    centered = features - mean
    variance = torch.sum(probability[:, None] * centered.square(), dim=0, keepdim=True)
    scale = variance.sqrt()
    return torch.where(scale > eps, centered / scale.clamp_min(eps), torch.zeros_like(centered))


def basis_invariant_geometry_features(
    frame: ParsevalResolventFrame,
    *,
    node_xyz_mm: Tensor | None = None,
    include_coordinates: bool = True,
    include_mass: bool = True,
    eps: float = 1.0e-8,
) -> Tensor:
    """Build node features without exposing arbitrary eigenvector coordinates.

    Spectral features are diagonals (leverage scores) of the frame operators:
    ``diag(S_b) = m_n sum_k s_b(lambda_k) phi_k(n)^2``.  They are unchanged by
    eigenvector signs and by rotations within exactly degenerate eigenspaces.
    Raw eigenvector columns are intentionally never returned.
    """

    basis = frame.eigenvectors.double()
    mass = frame.mass.double()
    squared = basis.square()
    band_leverage = mass[:, None] * (squared @ frame.band_weights.double().T)
    retained_leverage = mass * squared.sum(dim=1)
    complement_leverage = (1.0 - retained_leverage).clamp_min(0.0).unsqueeze(-1)
    columns = [complement_leverage, band_leverage]
    if include_mass:
        relative_mass = torch.log(mass / mass.median().clamp_min(float(eps))).unsqueeze(-1)
        columns.append(relative_mass)
    if include_coordinates:
        if node_xyz_mm is None:
            raise ValueError("node_xyz_mm is required when include_coordinates=True.")
        coordinates = torch.as_tensor(node_xyz_mm, dtype=torch.float64, device="cpu")
        if coordinates.shape != (frame.node_count, 3) or not torch.isfinite(coordinates).all():
            raise ValueError("node_xyz_mm must be finite [N,3].")
        columns.append(coordinates)
    features = torch.cat(columns, dim=-1)
    return _mass_standardize(features, mass, eps=float(eps))


@dataclass(frozen=True)
class GeometryEmbeddingDiagnostics:
    feature_type: str
    feature_dim: int
    feature_effective_rank: int
    embedding_rms: float
    embedding_spatial_std: float
    scale: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "feature_type": self.feature_type,
            "feature_dim": self.feature_dim,
            "feature_effective_rank": self.feature_effective_rank,
            "embedding_rms": self.embedding_rms,
            "embedding_spatial_std": self.embedding_spatial_std,
            "scale": self.scale,
        }


class GeometryEmbedding(nn.Module):
    """Project fixed geometry descriptors into an optional latent geometry embedding.

    The output is query independent.  ``feature_type='none'`` is an exact zero-embedding
    ablation with identical calling convention; ``spectral`` tests only stable
    operator-diagonal descriptors; ``spectral_xyz`` additionally tests template
    coordinates.  This makes the optional embedding an explicit empirical question
    rather than an unremovable assumption.
    """

    def __init__(
        self,
        frame: ParsevalResolventFrame,
        d_model: int,
        *,
        node_xyz_mm: Tensor | None = None,
        feature_type: GeometryFeatureType = "spectral",
        hidden_dim: int | None = None,
        initial_scale: float = 0.1,
        trainable_scale: bool = True,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if feature_type not in {"none", "spectral", "spectral_xyz"}:
            raise ValueError(f"Unsupported geometry-embedding feature type: {feature_type}.")
        if int(d_model) < 1:
            raise ValueError("d_model must be positive.")
        self.feature_type = str(feature_type)
        self.d_model = int(d_model)
        self.node_count = frame.node_count
        self.eps = float(eps)
        if feature_type == "none":
            features = torch.empty(frame.node_count, 0, dtype=torch.float64)
            self.projector: nn.Module | None = None
        else:
            features = basis_invariant_geometry_features(
                frame,
                node_xyz_mm=node_xyz_mm,
                include_coordinates=feature_type == "spectral_xyz",
                include_mass=True,
                eps=eps,
            )
            hidden = int(hidden_dim or max(16, min(4 * self.d_model, 128)))
            self.projector = nn.Sequential(
                nn.Linear(features.shape[1], hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.d_model, bias=False),
            )
        self.register_buffer("features", features, persistent=True)
        scale = torch.tensor(float(initial_scale), dtype=torch.float32)
        if trainable_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale, persistent=True)

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def forward(
        self,
        batch_size: int | None = None,
        *,
        reference: Tensor | None = None,
    ) -> Tensor:
        """Return one geometry embedding as [N,D] or expanded [B,N,D]."""

        if reference is not None:
            device, dtype = reference.device, reference.dtype
        else:
            device = self.scale.device
            dtype = self.scale.dtype
        if self.projector is None:
            embedding = torch.zeros(self.node_count, self.d_model, device=device, dtype=dtype)
        else:
            parameter = next(self.projector.parameters())
            features = self.features.to(device=parameter.device, dtype=parameter.dtype)
            embedding = self.projector(features) * self.scale.to(
                device=parameter.device, dtype=parameter.dtype
            )
            embedding = embedding.to(device=device, dtype=dtype)
        if batch_size is None:
            return embedding
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive.")
        return embedding.unsqueeze(0).expand(int(batch_size), -1, -1)

    @torch.no_grad()
    def diagnostics(self) -> GeometryEmbeddingDiagnostics:
        embedding = self.forward().float()
        if self.feature_dim:
            rank = int(torch.linalg.matrix_rank(self.features.float()))
        else:
            rank = 0
        return GeometryEmbeddingDiagnostics(
            feature_type=self.feature_type,
            feature_dim=self.feature_dim,
            feature_effective_rank=rank,
            embedding_rms=float(embedding.square().mean().sqrt()),
            embedding_spatial_std=float(embedding.std(dim=0, unbiased=False).mean()),
            scale=float(self.scale.detach()),
        )


__all__ = [
    "GeometryEmbedding",
    "GeometryEmbeddingDiagnostics",
    "GeometryFeatureType",
    "basis_invariant_geometry_features",
]
