"""Geometric mode composition on a shared latent FEM-node field.

The core decomposes the input field with a complete Parseval resolvent frame,
predicts a signed nodewise scale-mixing matrix, applies one shared nonlinear
scale update, and synthesizes the result back to the same node field. The exact
orthogonal complement preserves information outside the retained eigenmodes.
"""

from __future__ import annotations

from dataclasses import dataclass


import torch
from torch import Tensor, nn

from ..geometry.frame import ParsevalResolventFrame







@dataclass(frozen=True)
class GeoMCBlockDiagnostics:
    index: int
    input_rms: float
    output_rms: float
    input_entropy_effective_rank: float
    output_entropy_effective_rank: float
    input_centered_variance: float
    output_centered_variance: float
    input_spatial_mean_energy_fraction: float
    output_spatial_mean_energy_fraction: float
    residual_relative_l2: float
    reconstruction_relative_l2: float
    additive_energy_closure_relative_error: float
    centered_additive_energy_closure_relative_error: float
    routing_matrix_mean: tuple[tuple[float, ...], ...]
    routing_matrix_abs_mean: float
    cross_scale_routing_abs_mean: float
    scale_energy_fractions: tuple[float, ...]
    centered_scale_energy_fractions: tuple[float, ...]
    frame_factorization: str
    routing_mode: str
    routing_delta_abs_mean: float
    routing_spatial_std: float

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "input_rms": self.input_rms,
            "output_rms": self.output_rms,
            "input_entropy_effective_rank": self.input_entropy_effective_rank,
            "output_entropy_effective_rank": self.output_entropy_effective_rank,
            "input_centered_variance": self.input_centered_variance,
            "output_centered_variance": self.output_centered_variance,
            "input_spatial_mean_energy_fraction": self.input_spatial_mean_energy_fraction,
            "output_spatial_mean_energy_fraction": self.output_spatial_mean_energy_fraction,
            "residual_relative_l2": self.residual_relative_l2,
            "reconstruction_relative_l2": self.reconstruction_relative_l2,
            "additive_energy_closure_relative_error": (
                self.additive_energy_closure_relative_error
            ),
            "centered_additive_energy_closure_relative_error": (
                self.centered_additive_energy_closure_relative_error
            ),
            "routing_matrix_mean": [list(row) for row in self.routing_matrix_mean],
            "routing_matrix_abs_mean": self.routing_matrix_abs_mean,
            "cross_scale_routing_abs_mean": self.cross_scale_routing_abs_mean,
            "scale_energy_fractions": list(self.scale_energy_fractions),
            "centered_scale_energy_fractions": list(
                self.centered_scale_energy_fractions
            ),
            "frame_factorization": self.frame_factorization,
            "routing_mode": self.routing_mode,
            "routing_delta_abs_mean": self.routing_delta_abs_mean,
            "routing_spatial_std": self.routing_spatial_std,
        }


@dataclass(frozen=True)
class GeoMCDiagnostics:
    routing_mode: str
    block_count: int
    input_rms: float
    geometry_embedding_relative_rms: float
    output_rms: float
    output_channel_variance_mean: float
    output_hard_threshold_rank: int
    blocks: tuple[GeoMCBlockDiagnostics, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "routing_mode": self.routing_mode,
            "block_count": self.block_count,
            "input_rms": self.input_rms,
            "geometry_embedding_relative_rms": self.geometry_embedding_relative_rms,
            "output_rms": self.output_rms,
            "output_channel_variance_mean": self.output_channel_variance_mean,
            "output_hard_threshold_rank": self.output_hard_threshold_rank,
            "blocks": [block.as_dict() for block in self.blocks],
        }


def _rms(value: Tensor) -> Tensor:
    return value.float().square().mean().sqrt()


def _hard_threshold_rank(value: Tensor, *, tolerance: float = 1.0e-4) -> int:
    flattened = value.detach().float().reshape(-1, value.shape[-1])
    flattened = flattened - flattened.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(flattened)
    if singular.numel() == 0 or float(singular.max()) == 0.0:
        return 0
    return int((singular > singular.max() * float(tolerance)).sum())


def _mass_field_statistics(value: Tensor, mass: Tensor) -> tuple[float, float, float]:
    """Return mass-weighted entropy rank, centered variance and common energy."""

    if value.ndim != 3 or mass.shape != (value.shape[1],):
        raise ValueError("mass field statistics require value [B,N,D] and mass [N]")
    probability = mass.detach().float()
    probability = probability / probability.sum().clamp_min(1.0e-12)
    ranks: list[Tensor] = []
    variances: list[Tensor] = []
    spatial_mean_energy_fractions: list[Tensor] = []
    for item in value.detach().float():
        mean = torch.sum(probability[:, None] * item, dim=0, keepdim=True)
        centered = item - mean
        variance = torch.sum(probability * centered.square().mean(dim=-1))
        total = torch.sum(probability * item.square().mean(dim=-1))
        weighted = probability.sqrt()[:, None] * centered
        energy = torch.linalg.svdvals(weighted).square()
        energy_sum = energy.sum()
        if float(energy_sum) <= 1.0e-12:
            rank = torch.zeros((), device=value.device)
        else:
            spectral_probability = energy / energy_sum
            rank = torch.exp(
                -(
                    spectral_probability
                    * spectral_probability.clamp_min(1.0e-12).log()
                ).sum()
            )
        ranks.append(rank)
        variances.append(variance)
        spatial_mean_energy_fractions.append(mean.square().mean() / total.clamp_min(1.0e-12))
    return (
        float(torch.stack(ranks).mean().cpu()),
        float(torch.stack(variances).mean().cpu()),
        float(torch.stack(spatial_mean_energy_fractions).mean().cpu()),
    )


class SharedScaleUpdate(nn.Module):
    """One shared nonlinear update for every geometry scale.

    Each scale sees its own component, a routed cross-scale update,
    the matching input-scale component, the full input field and a learned scale code.
    Keeping the full input-field input lets optimization ignore an unhelpful modal
    split without removing the geometry-composition path.
    """

    def __init__(self, d_model: int, scale_dim: int, hidden_dim: int) -> None:
        super().__init__()
        inputs = 4 * int(d_model) + int(scale_dim)
        self.normalization = nn.LayerNorm(inputs)
        self.candidate = nn.Sequential(
            nn.Linear(inputs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(inputs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        component: Tensor,
        cross_scale_update: Tensor,
        input_scale_component: Tensor,
        input_field: Tensor,
        scale_code: Tensor,
    ) -> Tensor:
        inputs = torch.cat(
            (
                component,
                cross_scale_update,
                input_scale_component,
                input_field,
                scale_code,
            ),
            dim=-1,
        )
        normalized = self.normalization(inputs)
        return self.candidate(normalized) * self.gate(normalized)


class GeoMCBlock(nn.Module):
    """One analyze--route--compose--apply--synthesize field update."""

    def __init__(
        self,
        frame: ParsevalResolventFrame,
        d_model: int,
        *,
        hidden_dim: int,
        scale_embedding_dim: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        # GeoMCCore owns the frame module. Blocks keep a non-registering
        # reference so its large FEM buffers occur only once in state_dict.
        object.__setattr__(self, "_frame", frame)
        self.d_model = int(d_model)
        self.routing_mode = "nodewise_scale_mixing"
        scales = frame.scale_count
        self.scale_embedding = nn.Parameter(
            torch.randn(scales, int(scale_embedding_dim)) * 0.02
        )
        routing_input_dim = 4 * self.d_model + 3 * scales
        self.routing_mlp = nn.Sequential(
            nn.LayerNorm(routing_input_dim),
            nn.Linear(routing_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, scales * scales),
        )
        nn.init.zeros_(self.routing_mlp[-1].weight)
        nn.init.zeros_(self.routing_mlp[-1].bias)
        self.scale_update = SharedScaleUpdate(
            self.d_model, int(scale_embedding_dim), int(hidden_dim)
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    @property
    def frame(self) -> ParsevalResolventFrame:
        return self._frame

    def _mass_pool(self, value: Tensor) -> Tensor:
        probability = self.frame.mass.to(value.device, value.dtype)
        probability = probability / probability.sum()
        return torch.sum(probability[None, :, None] * value, dim=1)

    def _scale_routing_features(
        self,
        components: Tensor,
        input_scale_components: Tensor,
    ) -> Tensor:
        """Return per-scale statistics used by the routing MLP."""

        if components.shape != input_scale_components.shape or components.ndim != 4:
            raise ValueError("scale routing features require matching [B,S,N,D]")
        probability = self.frame.mass.to(components.device, torch.float32)
        probability = probability / probability.sum().clamp_min(self.frame.eps)
        left = components.float()
        right = input_scale_components.float()
        left_energy = torch.sum(
            probability[None, None, :, None] * left.square(), dim=(2, 3)
        ) / float(left.shape[-1])
        right_energy = torch.sum(
            probability[None, None, :, None] * right.square(), dim=(2, 3)
        ) / float(right.shape[-1])
        left_rms = torch.sqrt(left_energy + self.frame.eps)
        right_rms = torch.sqrt(right_energy + self.frame.eps)
        left_reference = left_rms.square().mean(dim=1, keepdim=True).sqrt()
        right_reference = right_rms.square().mean(dim=1, keepdim=True).sqrt()
        relative_left = torch.log(
            left_rms / left_reference.clamp_min(self.frame.eps)
        )
        relative_right = torch.log(
            right_rms / right_reference.clamp_min(self.frame.eps)
        )
        numerator = torch.sum(
            probability[None, None, :, None] * left * right, dim=(2, 3)
        )
        denominator = torch.sqrt(
            torch.sum(
                probability[None, None, :, None] * left.square(), dim=(2, 3)
            )
            * torch.sum(
                probability[None, None, :, None] * right.square(), dim=(2, 3)
            )
        ).clamp_min(self.frame.eps)
        cosine = numerator / denominator
        return torch.cat((relative_left, relative_right, cosine), dim=-1).to(
            components.dtype
        )

    def _routing_matrix(
        self,
        state: Tensor,
        input_field: Tensor,
        components: Tensor,
        input_scale_components: Tensor,
    ) -> Tensor:
        batch, nodes = state.shape[:2]
        scales = self.frame.scale_count
        pooled_state = self._mass_pool(state)
        global_input = self._mass_pool(input_field)
        scale_routing_features = self._scale_routing_features(
            components, input_scale_components
        )
        routing_features = torch.cat(
            (
                state,
                input_field,
                pooled_state[:, None, :].expand(-1, nodes, -1),
                global_input[:, None, :].expand(-1, nodes, -1),
                scale_routing_features[:, None, :].expand(-1, nodes, -1),
            ),
            dim=-1,
        )
        delta = self.routing_mlp(routing_features)
        identity = torch.eye(scales, device=state.device, dtype=state.dtype)
        return identity[None, None] + delta.reshape(batch, nodes, scales, scales)

    def forward(
        self,
        state: Tensor,
        input_field: Tensor,
        input_scale_components: Tensor,
        *,
        index: int,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, GeoMCBlockDiagnostics | None]:
        analysis = self.frame.analyze(state)
        components = analysis.components
        routing_matrix = self._routing_matrix(
            state, input_field, components, input_scale_components
        )
        routed_components = torch.einsum(
            "bnij,bjnd->bind", routing_matrix, components
        )
        batch, scales, nodes, _ = components.shape
        input_field_by_scale = input_field.unsqueeze(1).expand(
            -1, scales, -1, -1
        )
        scale_code = self.scale_embedding.to(state.dtype)[None, :, None, :].expand(
            batch, -1, nodes, -1
        )
        scale_updates = self.scale_update(
            components,
            routed_components - components,
            input_scale_components,
            input_field_by_scale,
            scale_code,
        )
        filtered_scale_updates = self.frame.apply(scale_updates)
        field_update = self.frame.synthesize(filtered_scale_updates)
        output = state + self.residual_scale * field_update
        if not collect_diagnostics:
            return output, None

        with torch.no_grad():
            reconstruction = self.frame.reconstruct(analysis)
            closure = (
                (reconstruction - state).float().norm()
                / state.float().norm().clamp_min(1.0e-8)
            )
            energy = self.frame.additive_relative_energies(state)
            centered_energy = self.frame.additive_relative_energies(
                state, center=True
            )
            probability = self.frame.mass.to(routing_matrix.device, torch.float32)
            probability = probability / probability.sum().clamp_min(self.frame.eps)
            identity = torch.eye(scales, device=routing_matrix.device, dtype=torch.float32)
            routing_delta = routing_matrix.detach().float() - identity[None, None]
            mean_routing_matrix = identity + torch.sum(
                probability[None, :, None, None] * routing_delta, dim=1
            ).mean(dim=0)
            off_diagonal = ~torch.eye(
                scales, device=routing_matrix.device, dtype=torch.bool
            )
            offdiag_mean = (
                routing_matrix.detach().float()[..., off_diagonal].abs().mean()
                if off_diagonal.any()
                else torch.tensor(0.0, device=routing_matrix.device)
            )
            input_rank, input_variance, input_spatial_mean_energy = _mass_field_statistics(
                state, self.frame.mass
            )
            output_rank, output_variance, output_spatial_mean_energy = _mass_field_statistics(
                output, self.frame.mass
            )
            diagnostic = GeoMCBlockDiagnostics(
                index=int(index),
                input_rms=float(_rms(state).detach()),
                output_rms=float(_rms(output).detach()),
                input_entropy_effective_rank=input_rank,
                output_entropy_effective_rank=output_rank,
                input_centered_variance=input_variance,
                output_centered_variance=output_variance,
                input_spatial_mean_energy_fraction=input_spatial_mean_energy,
                output_spatial_mean_energy_fraction=output_spatial_mean_energy,
                residual_relative_l2=float(
                    (self.residual_scale.detach() * field_update.detach()).float().norm()
                    / state.detach().float().norm().clamp_min(1.0e-8)
                ),
                reconstruction_relative_l2=float(closure),
                additive_energy_closure_relative_error=float(
                    self.frame.additive_energy_closure_relative_error(state)
                ),
                centered_additive_energy_closure_relative_error=float(
                    self.frame.additive_energy_closure_relative_error(
                        state, center=True
                    )
                ),
                routing_matrix_mean=tuple(
                    tuple(float(value) for value in row)
                    for row in mean_routing_matrix.cpu()
                ),
                routing_matrix_abs_mean=float(routing_matrix.detach().float().abs().mean()),
                cross_scale_routing_abs_mean=float(offdiag_mean),
                scale_energy_fractions=tuple(
                    float(value) for value in energy.cpu()
                ),
                centered_scale_energy_fractions=tuple(
                    float(value) for value in centered_energy.cpu()
                ),
                frame_factorization=self.frame.factorization,
                routing_mode=self.routing_mode,
                routing_delta_abs_mean=float(
                    (routing_matrix.detach().float() - identity[None, None]).abs().mean()
                ),
                routing_spatial_std=float(
                    routing_matrix.detach().float().std(dim=1, unbiased=False).mean()
                ),
            )
        return output, diagnostic


class GeoMCCore(nn.Module):
    """Update one query-independent input field on FEM nodes."""

    def __init__(
        self,
        frame: ParsevalResolventFrame,
        d_model: int,
        *,
        input_dim: int | None = None,
        num_blocks: int = 3,
        hidden_dim: int | None = None,
        scale_embedding_dim: int = 16,
        residual_scale: float = 0.1,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        if frame.factorization != "parseval_sqrt":
            raise ValueError("GeoMCCore requires a parseval_sqrt frame")
        if int(num_blocks) < 3:
            raise ValueError("GeoMC requires at least three composition blocks")
        if int(d_model) < 1 or int(input_dim or d_model) < 1:
            raise ValueError("d_model and input_dim must be positive")

        self.frame = frame
        self.routing_mode = "nodewise_scale_mixing"
        self.d_model = int(d_model)
        self.input_dim = int(input_dim or d_model)
        hidden = int(hidden_dim or max(2 * self.d_model, 64))
        self.input_projection = (
            nn.Identity()
            if self.input_dim == self.d_model
            else nn.Linear(self.input_dim, self.d_model)
        )
        self.input_normalization = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList(
            GeoMCBlock(
                self.frame,
                self.d_model,
                hidden_dim=hidden,
                scale_embedding_dim=int(scale_embedding_dim),
                residual_scale=float(residual_scale),
            )
            for _ in range(int(num_blocks))
        )
        self.output_normalization = (
            nn.LayerNorm(self.d_model) if bool(normalize_output) else nn.Identity()
        )

    def _prepare_geometry_embedding(
        self, geometry_embedding: Tensor | None, state: Tensor
    ) -> tuple[Tensor, float]:
        if geometry_embedding is None:
            return torch.zeros_like(state), 0.0
        value = geometry_embedding
        if value.ndim == 2:
            value = value.unsqueeze(0).expand(state.shape[0], -1, -1)
        if value.shape != state.shape:
            raise ValueError(
                f"geometry_embedding must be [N,D] or {tuple(state.shape)}, got {tuple(geometry_embedding.shape)}"
            )
        value = value.to(device=state.device, dtype=state.dtype)
        ratio = float(
            _rms(value).detach() / _rms(state).detach().clamp_min(1.0e-8)
        )
        return value, ratio

    def forward(
        self,
        input_field: Tensor,
        geometry_embedding: Tensor | None = None,
        *,
        return_diagnostics: bool = True,
    ) -> Tensor | tuple[Tensor, GeoMCDiagnostics]:
        if input_field.ndim != 3 or input_field.shape[1] != self.frame.node_count:
            raise ValueError(
                f"input_field must be [B,{self.frame.node_count},{self.input_dim}]"
            )
        if input_field.shape[-1] != self.input_dim:
            raise ValueError(f"input_field last dimension must be {self.input_dim}")
        if not input_field.is_floating_point() or not torch.isfinite(input_field).all():
            raise ValueError("input_field must be finite floating point values")

        embedded = self.input_normalization(self.input_projection(input_field))
        geometry_embedding_value, geometry_embedding_ratio = (
            self._prepare_geometry_embedding(geometry_embedding, embedded)
        )
        state = embedded + geometry_embedding_value
        initial = state
        input_scale_components = self.frame.analyze(embedded).components
        block_diagnostics: list[GeoMCBlockDiagnostics] = []
        for index, block in enumerate(self.blocks):
            state, diagnostic = block(
                state,
                embedded,
                input_scale_components,
                index=index,
                collect_diagnostics=return_diagnostics,
            )
            if diagnostic is not None:
                block_diagnostics.append(diagnostic)

        output = self.output_normalization(state)
        if not return_diagnostics:
            return output
        diagnostics = GeoMCDiagnostics(
            routing_mode=self.routing_mode,
            block_count=len(self.blocks),
            input_rms=float(_rms(initial).detach()),
            geometry_embedding_relative_rms=geometry_embedding_ratio,
            output_rms=float(_rms(output).detach()),
            output_channel_variance_mean=float(
                output.detach().float().var(dim=(0, 1), unbiased=False).mean()
            ),
            output_hard_threshold_rank=_hard_threshold_rank(output),
            blocks=tuple(block_diagnostics),
        )
        return output, diagnostics


__all__ = [
    "GeoMCBlock",
    "GeoMCBlockDiagnostics",
    "GeoMCCore",
    "GeoMCDiagnostics",
    "SharedScaleUpdate",
]
