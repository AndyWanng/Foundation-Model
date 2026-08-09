"""Full-space Parseval resolvent frame on a volumetric FEM domain.

The frame is deliberately an analysis/synthesis coordinate system, not a
low-frequency bottleneck. Retained Laplace--Beltrami modes are partitioned by
non-negative resolvent windows and the exact orthogonal complement is carried
as an additional scale. The canonical tight factors ``A_b=S_b^{1/2}`` satisfy
``sum_b A_b^* A_b=I`` under the FEM mass inner product, so
``reconstruct(analyze(x)) == x`` up to floating-point error for every FEM-node
field, including fields outside the retained eigenspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FrameAnalysis:
    """Modal coefficients and active-factor FEM-node components."""

    components: Tensor
    coefficients: Tensor
    scale_names: tuple[str, ...]

    def validate(self) -> None:
        if self.components.ndim != 4:
            raise ValueError("components must be [B,S,N,D].")
        if self.coefficients.ndim != 3:
            raise ValueError("coefficients must be [B,K,D].")
        if self.components.shape[1] != len(self.scale_names):
            raise ValueError("scale_names does not match the component axis.")
        if self.components.shape[0] != self.coefficients.shape[0]:
            raise ValueError("components and coefficients use different batches.")
        if self.components.shape[-1] != self.coefficients.shape[-1]:
            raise ValueError("components and coefficients use different channels.")


@dataclass(frozen=True)
class FrameDiagnostics:
    factorization: str
    partition_max_abs_error: float
    mass_orthogonality_max_abs_error: float
    reconstruction_max_abs_error: float
    reconstruction_relative_l2_error: float
    dc_energy_fraction: float
    complement_energy_fraction: float
    scale_energy_fractions: tuple[float, ...]
    centered_scale_energy_fractions: tuple[float, ...]
    additive_energy_closure_relative_error: float
    centered_additive_energy_closure_relative_error: float
    analysis_component_euclidean_energy_fractions: tuple[float, ...]

    def as_dict(self) -> dict[str, float | str | list[float]]:
        return {
            "factorization": self.factorization,
            "partition_max_abs_error": self.partition_max_abs_error,
            "mass_orthogonality_max_abs_error": self.mass_orthogonality_max_abs_error,
            "reconstruction_max_abs_error": self.reconstruction_max_abs_error,
            "reconstruction_relative_l2_error": self.reconstruction_relative_l2_error,
            "dc_energy_fraction": self.dc_energy_fraction,
            "complement_energy_fraction": self.complement_energy_fraction,
            "scale_energy_fractions": list(self.scale_energy_fractions),
            "centered_scale_energy_fractions": list(
                self.centered_scale_energy_fractions
            ),
            "additive_energy_closure_relative_error": (
                self.additive_energy_closure_relative_error
            ),
            "centered_additive_energy_closure_relative_error": (
                self.centered_additive_energy_closure_relative_error
            ),
            "analysis_component_euclidean_energy_fractions": list(
                self.analysis_component_euclidean_energy_fractions
            ),
        }


def fixed_spatial_permutation(node_count: int, *, seed: int = 1729) -> Tensor:
    """Return a reproducible non-identity permutation for a spatial null.

    A fixed permutation is important: regenerating a null each epoch changes
    the learning problem and confounds geometry with stochastic augmentation.
    """

    count = int(node_count)
    if count < 2:
        raise ValueError("A spatial permutation null requires at least two nodes.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(count, generator=generator)
    if torch.equal(permutation, torch.arange(count)):
        permutation = torch.roll(permutation, shifts=1)
    return permutation


class ParsevalResolventFrame(nn.Module):
    """A basis-invariant soft spectral frame with an exact complement.

    Let ``Phi.T @ M @ Phi = I``.  For increasing radii ``r_j`` the retained
    eigenspace is partitioned as

    ``1-R(r_1), R(r_1)-R(r_2), ..., R(r_J)``

    where ``R(r, lambda)=1/(1+r^2 lambda)``.  The additional component
    ``S_perp = I - Phi Phi.T M`` carries everything outside the retained
    eigenspace.  Windows are functions of eigenvalues, so sign changes and
    rotations inside exactly degenerate eigenspaces leave every operator
    unchanged.
    """

    def __init__(
        self,
        eigenvalues: Tensor,
        eigenvectors: Tensor,
        mass: Tensor,
        *,
        radii: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        normalize_spectrum: bool = False,
        factorization: str = "parseval_sqrt",
        orthogonality_tolerance: float = 5.0e-4,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(eigenvalues).detach().clone()
        basis = torch.as_tensor(eigenvectors).detach().clone()
        node_mass = torch.as_tensor(mass).detach().clone()
        if not values.is_floating_point():
            values = values.float()
        if not basis.is_floating_point():
            basis = basis.float()
        if not node_mass.is_floating_point():
            node_mass = node_mass.float()
        values = values.to(dtype=torch.float64, device="cpu")
        basis = basis.to(dtype=torch.float64, device="cpu")
        node_mass = node_mass.to(dtype=torch.float64, device="cpu")
        if values.ndim != 1 or values.numel() < 1:
            raise ValueError("eigenvalues must be non-empty [K].")
        if basis.ndim != 2 or basis.shape[1] != values.numel():
            raise ValueError("eigenvectors must be [N,K].")
        if node_mass.shape != (basis.shape[0],):
            raise ValueError("mass must be [N].")
        if not torch.isfinite(values).all() or torch.any(values < -1.0e-8):
            raise ValueError("eigenvalues must be finite and non-negative.")
        if torch.any(torch.diff(values) < -1.0e-8):
            raise ValueError("eigenvalues must be sorted.")
        if not torch.isfinite(basis).all():
            raise ValueError("eigenvectors must be finite.")
        if not torch.isfinite(node_mass).all() or torch.any(node_mass <= 0):
            raise ValueError("mass must be finite and strictly positive.")
        radius_values = tuple(float(value) for value in radii)
        if not radius_values or any(value <= 0 for value in radius_values):
            raise ValueError("radii must contain positive values.")
        if any(right <= left for left, right in zip(radius_values, radius_values[1:])):
            raise ValueError("radii must be strictly increasing.")
        gram = basis.T @ (node_mass[:, None] * basis)
        orthogonality_error = float(
            (gram - torch.eye(values.numel(), dtype=torch.float64)).abs().max()
        )
        if orthogonality_error > float(orthogonality_tolerance):
            raise ValueError(
                "eigenvectors must be M-orthonormal; "
                f"maximum error={orthogonality_error:.6g}."
            )
        values = values.clamp_min(0.0)
        positive = values[values > float(eps)]
        spectral_scale = (
            positive.median()
            if bool(normalize_spectrum) and positive.numel()
            else torch.tensor(1.0, dtype=torch.float64)
        )
        scaled = values / spectral_scale.clamp_min(float(eps))
        radius_tensor = torch.tensor(radius_values, dtype=torch.float64)
        resolvents = 1.0 / (
            1.0 + radius_tensor[:, None].square() * scaled[None, :]
        )
        windows = [1.0 - resolvents[0]]
        windows.extend(
            resolvents[index] - resolvents[index + 1]
            for index in range(len(radius_values) - 1)
        )
        windows.append(resolvents[-1])
        partition = torch.stack(windows, dim=0).clamp_min(0.0)
        # This normalization only removes round-off and makes the closure
        # algebra explicit in the saved artifact.
        partition = partition / partition.sum(dim=0, keepdim=True).clamp_min(float(eps))

        self.register_buffer("eigenvalues", values, persistent=True)
        self.register_buffer("eigenvectors", basis, persistent=True)
        self.register_buffer("mass", node_mass, persistent=True)
        self.register_buffer("radii", radius_tensor, persistent=True)
        self.register_buffer("spectral_scale", spectral_scale.reshape(()), persistent=True)
        self.register_buffer("band_weights", partition, persistent=True)
        if factorization != "parseval_sqrt":
            raise ValueError("GeoMC uses the canonical 'parseval_sqrt' factorization.")
        self.factorization = "parseval_sqrt"
        self.normalize_spectrum = bool(normalize_spectrum)
        self.orthogonality_tolerance = float(orthogonality_tolerance)
        self.eps = float(eps)
        if len(radius_values) == 1:
            band_names = ("fine", "global")
        else:
            band_names = (
                "fine",
                *(f"resolvent_{index}" for index in range(1, len(radius_values))),
                "global",
            )
        self.scale_names = ("complement", *band_names)

    @property
    def node_count(self) -> int:
        return int(self.mass.numel())

    @property
    def mode_count(self) -> int:
        return int(self.eigenvalues.numel())

    @property
    def band_count(self) -> int:
        return int(self.band_weights.shape[0])

    @property
    def scale_count(self) -> int:
        return 1 + self.band_count

    @property
    def partition_max_abs_error(self) -> float:
        return float((self.band_weights.sum(dim=0) - 1.0).abs().max())

    @property
    def mass_orthogonality_max_abs_error(self) -> float:
        basis, mass = self.eigenvectors.double(), self.mass.double()
        gram = basis.T @ (mass[:, None] * basis)
        return float((gram - torch.eye(self.mode_count, dtype=torch.float64)).abs().max())

    def _validate_field(self, field: Tensor) -> None:
        if field.ndim != 3 or field.shape[1] != self.node_count:
            raise ValueError(
                f"FEM-node field must be [B,{self.node_count},D], got {tuple(field.shape)}."
            )
        if not field.is_floating_point() or not torch.isfinite(field).all():
            raise ValueError("FEM-node field must be finite floating point values.")

    def coefficients(self, field: Tensor) -> Tensor:
        """Mass-weighted modal coefficients ``Phi.T M field``."""

        self._validate_field(field)
        # Never execute the FEM projection in fp16/bf16.  The server uses bf16
        # autocast for learned layers, but orthogonal complements involve a
        # cancellation (x-Px) whose error is materially larger at low precision.
        compute_dtype = (
            torch.float32
            if field.dtype in {torch.float16, torch.bfloat16}
            else field.dtype
        )
        basis = self.eigenvectors.to(field.device, compute_dtype)
        mass = self.mass.to(field.device, compute_dtype)
        value = field.to(compute_dtype)
        with torch.autocast(device_type=field.device.type, enabled=False):
            return torch.einsum("nk,n,bnd->bkd", basis, mass, value)

    @property
    def factor_band_weights(self) -> Tensor:
        """Canonical tight factors ``A_b=sqrt(S_b)``."""

        return self.band_weights.clamp_min(0.0).sqrt()

    def _retained_synthesis(self, coefficients: Tensor, weights: Tensor) -> Tensor:
        basis = self.eigenvectors.to(coefficients.device, coefficients.dtype)
        spectral = weights.to(coefficients.device, coefficients.dtype)
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            return torch.einsum("nk,sk,bkd->bsnd", basis, spectral, coefficients)

    def analyze(self, field: Tensor) -> FrameAnalysis:
        """Apply every configured analysis factor and retain coefficients."""

        coefficients = self.coefficients(field)
        retained = self._retained_synthesis(coefficients, self.factor_band_weights)
        ones = torch.ones(
            1,
            self.mode_count,
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        projection = self._retained_synthesis(coefficients, ones).squeeze(1)
        complement = field.to(projection.dtype) - projection
        analysis = FrameAnalysis(
            components=torch.cat((complement.unsqueeze(1), retained), dim=1),
            coefficients=coefficients,
            scale_names=self.scale_names,
        )
        analysis.validate()
        return analysis

    def apply_scale(self, scale: int, field: Tensor) -> Tensor:
        """Apply one configured synthesis factor to an arbitrary node field."""

        self._validate_field(field)
        index = int(scale)
        if not 0 <= index < self.scale_count:
            raise IndexError(f"scale must lie in [0,{self.scale_count}).")
        coefficients = self.coefficients(field)
        if index == 0:
            ones = torch.ones(
                1, self.mode_count, device=field.device, dtype=field.dtype
            )
            projection = self._retained_synthesis(coefficients, ones).squeeze(1)
            return field.to(projection.dtype) - projection
        weights = self.factor_band_weights[index - 1 : index]
        return self._retained_synthesis(coefficients, weights).squeeze(1)

    def apply(self, scale_fields: Tensor) -> Tensor:
        """Apply scale ``i`` to ``scale_fields[:,i]`` before synthesis.

        Nonlinear cells generally leave their input eigensubspace.  Reapplying
        each frame operator is therefore essential; simply summing cell outputs
        would not implement geometry-grounded composition.
        """

        if scale_fields.ndim != 4 or scale_fields.shape[1] != self.scale_count:
            raise ValueError(
                "scale_fields must be [B,S,N,D] with S equal to frame.scale_count."
            )
        if scale_fields.shape[2] != self.node_count:
            raise ValueError("scale_fields uses the wrong FEM node count.")
        return torch.stack(
            [self.apply_scale(index, scale_fields[:, index]) for index in range(self.scale_count)],
            dim=1,
        )

    def synthesize(self, components: Tensor | FrameAnalysis) -> Tensor:
        """Sum already frame-aligned components into one FEM-node field."""

        values = components.components if isinstance(components, FrameAnalysis) else components
        if values.ndim != 4 or values.shape[1] != self.scale_count:
            raise ValueError("components must be [B,S,N,D].")
        if values.shape[2] != self.node_count:
            raise ValueError("components uses the wrong FEM node count.")
        return values.sum(dim=1)

    def reconstruct(self, analysis: FrameAnalysis) -> Tensor:
        """Close the tight analysis/synthesis pair on a field."""

        analysis.validate()
        return self.synthesize(self.apply(analysis.components))

    def operator_matrix(self, scale: int) -> Tensor:
        """Materialize the partition operator ``S_b``.

        This remains independent of factorization, so its sum is always the
        identity when the exact complement is included.
        """

        index = int(scale)
        if not 0 <= index < self.scale_count:
            raise IndexError(f"scale must lie in [0,{self.scale_count}).")
        basis, mass = self.eigenvectors, self.mass
        projection = (basis @ basis.T) * mass[None, :]
        if index == 0:
            return torch.eye(self.node_count, dtype=basis.dtype) - projection
        weighted = basis * self.band_weights[index - 1][None, :]
        return (weighted @ basis.T) * mass[None, :]

    def factor_operator_matrix(self, scale: int) -> Tensor:
        """Materialize the configured analysis/synthesis factor ``A_b``."""

        index = int(scale)
        if not 0 <= index < self.scale_count:
            raise IndexError(f"scale must lie in [0,{self.scale_count}).")
        basis, mass = self.eigenvectors, self.mass
        projection = (basis @ basis.T) * mass[None, :]
        if index == 0:
            return torch.eye(self.node_count, dtype=basis.dtype) - projection
        weighted = basis * self.factor_band_weights[index - 1][None, :]
        return (weighted @ basis.T) * mass[None, :]

    def additive_components(self, field: Tensor, *, center: bool = False) -> Tensor:
        """Return tight-frame components used for additive FEM-mass energy."""

        self._validate_field(field)
        compute_dtype = torch.float64 if field.dtype == torch.float64 else torch.float32
        value = field.to(compute_dtype)
        if center:
            mass = self.mass.to(field.device, compute_dtype)
            probability = mass / mass.sum().clamp_min(self.eps)
            mean = torch.sum(probability[None, :, None] * value, dim=1, keepdim=True)
            value = value - mean
        coefficients = self.coefficients(value)
        retained = self._retained_synthesis(
            coefficients, self.band_weights.clamp_min(0.0).sqrt()
        )
        ones = torch.ones(
            1,
            self.mode_count,
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        projection = self._retained_synthesis(coefficients, ones).squeeze(1)
        complement = value.to(projection.dtype) - projection
        return torch.cat((complement.unsqueeze(1), retained), dim=1)

    def _mass_relative_energies(self, components: Tensor) -> Tensor:
        mass = self.mass.to(components.device, torch.float64)
        energy = (
            components.to(torch.float64).square()
            * mass[None, None, :, None]
        ).sum(dim=(0, 2, 3))
        return energy / energy.sum().clamp_min(self.eps)

    def additive_relative_energies(
        self, field: Tensor, *, center: bool = False
    ) -> Tensor:
        """Return positive, additive Parseval energy shares for one field."""

        return self._mass_relative_energies(
            self.additive_components(field, center=center)
        )

    def mass_energy(self, field: Tensor, *, center: bool = False) -> Tensor:
        """Return the raw FEM-mass energy of a field as a float64 scalar."""

        self._validate_field(field)
        value = field.to(torch.float64)
        mass = self.mass.to(field.device, torch.float64)
        if center:
            probability = mass / mass.sum().clamp_min(self.eps)
            mean = torch.sum(probability[None, :, None] * value, dim=1, keepdim=True)
            value = value - mean
        return (value.square() * mass[None, :, None]).sum()

    def additive_mass_energies(
        self, field: Tensor, *, center: bool = False
    ) -> Tensor:
        """Return each Parseval component's *unnormalized* FEM-mass energy."""

        components = self.additive_components(field, center=center).to(torch.float64)
        mass = self.mass.to(field.device, torch.float64)
        return (components.square() * mass[None, None, :, None]).sum(dim=(0, 2, 3))

    def additive_energy_closure_relative_error(
        self, field: Tensor, *, center: bool = False
    ) -> Tensor:
        """Test Parseval energy closure without normalizing away the error."""

        reference = self.mass_energy(field, center=center)
        additive = self.additive_mass_energies(field, center=center).sum()
        return (additive - reference).abs() / reference.clamp_min(self.eps)

    def diagnostics(self, field: Tensor) -> FrameDiagnostics:
        analysis = self.analyze(field)
        reconstructed = self.reconstruct(analysis)
        difference = reconstructed - field
        denominator = field.double().square().sum().sqrt().clamp_min(self.eps)
        relative = self.additive_relative_energies(field)
        centered_relative = self.additive_relative_energies(field, center=True)
        active_energy = analysis.components.double().square().sum(dim=(0, 2, 3))
        active_relative = active_energy / active_energy.sum().clamp_min(self.eps)
        coefficients = self.coefficients(field).double()
        dc_mask = self.eigenvalues.to(coefficients.device) <= self.eps
        dc_energy = coefficients[:, dc_mask].square().sum()
        mass = self.mass.to(field.device, torch.float64)
        total_mass_energy = (
            field.to(torch.float64).square() * mass[None, :, None]
        ).sum().clamp_min(self.eps)
        return FrameDiagnostics(
            factorization=self.factorization,
            partition_max_abs_error=self.partition_max_abs_error,
            mass_orthogonality_max_abs_error=self.mass_orthogonality_max_abs_error,
            reconstruction_max_abs_error=float(difference.abs().max()),
            reconstruction_relative_l2_error=float(
                difference.double().square().sum().sqrt() / denominator
            ),
            dc_energy_fraction=float(dc_energy / total_mass_energy),
            complement_energy_fraction=float(relative[0]),
            scale_energy_fractions=tuple(float(value) for value in relative),
            centered_scale_energy_fractions=tuple(
                float(value) for value in centered_relative
            ),
            additive_energy_closure_relative_error=float(
                self.additive_energy_closure_relative_error(field)
            ),
            centered_additive_energy_closure_relative_error=float(
                self.additive_energy_closure_relative_error(field, center=True)
            ),
            analysis_component_euclidean_energy_fractions=tuple(
                float(value) for value in active_relative
            ),
        )

    def spatial_permutation_null(
        self, *, seed: int = 1729
    ) -> tuple["ParsevalResolventFrame", Tensor]:
        """Create a deterministic mass-orthonormal spatial-permutation null.

        Rows are permuted in mass-whitened coordinates and then mapped back to
        the original FEM mass.  This breaks the anatomy-to-mode assignment while
        preserving the spectrum, exact M-orthonormality, parameter count and
        frame closure.  The returned permutation should be persisted with the
        experiment record.
        """

        permutation = fixed_spatial_permutation(self.node_count, seed=seed)
        mass = self.mass.double()
        whitened = torch.sqrt(mass)[:, None] * self.eigenvectors.double()
        null_basis = whitened[permutation] / torch.sqrt(mass)[:, None]
        null = ParsevalResolventFrame(
            self.eigenvalues,
            null_basis,
            mass,
            radii=tuple(float(value) for value in self.radii),
            normalize_spectrum=self.normalize_spectrum,
            factorization=self.factorization,
            orthogonality_tolerance=self.orthogonality_tolerance,
            eps=self.eps,
        )
        return null, permutation
