"""Query-aware 3D JEPA ViT predictor applied strictly after GeoMC."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PhysicalPositionEncoding3D(nn.Module):
    """Fixed sinusoidal encoding of physical world coordinates in millimetres."""

    def __init__(
        self,
        d_model: int,
        *,
        min_wavelength_mm: float = 4.0,
        max_wavelength_mm: float = 512.0,
        origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        if int(d_model) < 6:
            raise ValueError("d_model must be at least 6 for 3D position encoding")
        if not 0 < float(min_wavelength_mm) <= float(max_wavelength_mm):
            raise ValueError("physical wavelengths must be positive and ordered")
        self.d_model = int(d_model)
        frequency_count = math.ceil(self.d_model / 6)
        wavelengths = torch.logspace(
            math.log10(float(min_wavelength_mm)),
            math.log10(float(max_wavelength_mm)),
            frequency_count,
            dtype=torch.float32,
        )
        self.register_buffer("frequencies", 2.0 * torch.pi / wavelengths, persistent=True)
        self.register_buffer("origin_mm", torch.tensor(origin_mm, dtype=torch.float32), persistent=True)

    def forward(self, xyz_mm: Tensor) -> Tensor:
        if xyz_mm.ndim not in (2, 3) or xyz_mm.shape[-1] != 3:
            raise ValueError("xyz_mm must be [P,3] or [B,P,3]")
        if not torch.is_floating_point(xyz_mm) or not torch.isfinite(xyz_mm).all():
            raise ValueError("xyz_mm must be finite floating point coordinates")
        centered = xyz_mm.float() - self.origin_mm.to(xyz_mm)
        phase = centered.unsqueeze(-1) * self.frequencies.to(xyz_mm)
        encoded = torch.cat((phase.sin(), phase.cos()), dim=-1)
        encoded = encoded.flatten(start_dim=-2)[..., : self.d_model]
        if encoded.shape[-1] < self.d_model:
            encoded = torch.nn.functional.pad(encoded, (0, self.d_model - encoded.shape[-1]))
        return encoded.to(dtype=xyz_mm.dtype)


class DropPath(nn.Module):
    """Per-item stochastic depth without a timm dependency at model runtime."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= float(probability) < 1.0:
            raise ValueError("drop-path probability must lie in [0,1)")
        self.probability = float(probability)

    def forward(self, value: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return value
        keep = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = torch.empty(shape, device=value.device, dtype=value.dtype).bernoulli_(keep)
        return value * mask / keep


class JEPA3DBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        hidden = int(round(float(mlp_ratio) * int(d_model)))
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=0.0, batch_first=True
        )
        self.attention_drop_path = DropPath(drop_path)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.mlp_drop_path = DropPath(drop_path)

    def forward(self, tokens: Tensor, *, key_padding_mask: Tensor) -> Tensor:
        normalised = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalised,
            normalised,
            normalised,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.attention_drop_path(attended)
        tokens = tokens + self.mlp_drop_path(self.mlp(self.mlp_norm(tokens)))
        return tokens


class QueryAwareJEPA3DPredictor(nn.Module):
    """ViT predictor over GeoMC context nodes and physical query tokens.

    This module has no image/SAT3D reference and therefore cannot form a raw
    encoder skip.  At most ``max_queries`` masked target locations are decoded.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        condition_dim: int = 128,
        relation_dim: int = 32,
        *,
        d_model: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        max_queries: int = 128,
    ) -> None:
        super().__init__()
        if min(
            int(latent_dim),
            int(condition_dim),
            int(relation_dim),
            int(d_model),
            int(depth),
            int(num_heads),
            int(max_queries),
        ) < 1:
            raise ValueError("predictor dimensions must be positive")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.relation_dim = int(relation_dim)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.max_queries = int(max_queries)
        self.context_projection = nn.Linear(self.latent_dim, self.d_model)
        self.target_condition_projection = nn.Linear(self.condition_dim, self.d_model)
        self.relation_projection = nn.Linear(self.relation_dim, self.d_model)
        self.query_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.position_encoding = PhysicalPositionEncoding3D(self.d_model)
        probabilities = torch.linspace(0.0, float(drop_path_rate), self.depth).tolist()
        self.blocks = nn.ModuleList(
            JEPA3DBlock(
                self.d_model,
                int(num_heads),
                mlp_ratio=float(mlp_ratio),
                drop_path=float(probability),
            )
            for probability in probabilities
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.latent_dim)

    @staticmethod
    def _expand_positions(positions: Tensor, batch: int) -> Tensor:
        if positions.ndim == 2:
            return positions.unsqueeze(0).expand(batch, -1, -1)
        if positions.ndim == 3 and positions.shape[0] == batch:
            return positions
        raise ValueError("positions must be [P,3] or [B,P,3]")

    def forward(
        self,
        latent_field: Tensor,
        context_xyz_mm: Tensor,
        context_coverage: Tensor,
        query_xyz_mm: Tensor,
        query_valid: Tensor,
        target_condition: Tensor,
        relation_condition: Tensor,
    ) -> Tensor:
        if latent_field.ndim != 3 or latent_field.shape[-1] != self.latent_dim:
            raise ValueError(f"latent_field must be [B,N,{self.latent_dim}]")
        batch, nodes = latent_field.shape[:2]
        if context_coverage.shape != (batch, nodes):
            raise ValueError("context_coverage must be [B,N]")
        query_valid = torch.as_tensor(query_valid, device=latent_field.device, dtype=torch.bool)
        if query_valid.ndim != 2 or query_valid.shape[0] != batch:
            raise ValueError("query_valid must be [B,Q]")
        queries = int(query_valid.shape[1])
        if queries < 1 or queries > self.max_queries:
            raise ValueError(f"Query count must lie in [1,{self.max_queries}]")
        if torch.any(query_valid.sum(dim=1) < 1):
            raise ValueError("Every item needs at least one valid query")
        context_valid = torch.as_tensor(
            context_coverage > 0, device=latent_field.device, dtype=torch.bool
        )
        if torch.any(context_valid.sum(dim=1) < 1):
            raise ValueError("Every item needs at least one supported GeoMC context node")
        if target_condition.shape != (batch, self.condition_dim):
            raise ValueError(f"target_condition must be [B,{self.condition_dim}]")
        if relation_condition.shape != (batch, self.relation_dim):
            raise ValueError(f"relation_condition must be [B,{self.relation_dim}]")
        context_positions = self._expand_positions(context_xyz_mm, batch)
        query_positions = self._expand_positions(query_xyz_mm, batch)
        if context_positions.shape != (batch, nodes, 3):
            raise ValueError("context_xyz_mm count differs from GeoMC nodes")
        if query_positions.shape != (batch, queries, 3):
            raise ValueError("query_xyz_mm count differs from packed queries")

        context_tokens = self.context_projection(latent_field)
        context_tokens = context_tokens + self.position_encoding(context_positions)
        condition_token = self.target_condition_projection(target_condition)
        relation_token = self.relation_projection(relation_condition)
        query_tokens = self.query_token.expand(batch, queries, -1)
        query_tokens = query_tokens + self.position_encoding(query_positions)
        query_tokens = query_tokens + condition_token[:, None, :] + relation_token[:, None, :]
        tokens = torch.cat((context_tokens, query_tokens), dim=1)
        key_padding_mask = ~torch.cat((context_valid, query_valid), dim=1)
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)
        query_output = tokens[:, nodes:, :]
        prediction = self.output_projection(self.output_norm(query_output))
        prediction = torch.where(
            query_valid.unsqueeze(-1), prediction, torch.zeros_like(prediction)
        )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("3D JEPA predictor produced non-finite values")
        return prediction


__all__ = [
    "JEPA3DBlock",
    "PhysicalPositionEncoding3D",
    "QueryAwareJEPA3DPredictor",
]
