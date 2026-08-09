from __future__ import annotations

import torch

from mri_pet_geomc.field.geometry_embedding import basis_invariant_geometry_features
from mri_pet_geomc.geometry.frame import ParsevalResolventFrame, fixed_spatial_permutation


def _frame(
    *,
    rotate_degenerate: bool = False,
    factorization: str = "parseval_sqrt",
) -> ParsevalResolventFrame:
    generator = torch.Generator().manual_seed(7)
    nodes, modes = 13, 5
    mass = torch.linspace(0.7, 1.9, nodes, dtype=torch.float64)
    whitened, _ = torch.linalg.qr(
        torch.randn(nodes, modes, generator=generator, dtype=torch.float64)
    )
    basis = whitened / torch.sqrt(mass)[:, None]
    if rotate_degenerate:
        angle = torch.tensor(0.61, dtype=torch.float64)
        rotation = torch.tensor(
            [[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]],
            dtype=torch.float64,
        )
        basis[:, 1:3] = basis[:, 1:3] @ rotation
    eigenvalues = torch.tensor([0.0, 0.8, 0.8, 2.5, 9.0], dtype=torch.float64)
    return ParsevalResolventFrame(
        eigenvalues,
        basis,
        mass,
        radii=(0.4, 1.0, 2.5),
        factorization=factorization,
    )


def test_resolvent_partition_and_exact_complement_reconstruct_every_field() -> None:
    frame = _frame()
    assert torch.all(frame.band_weights >= 0)
    assert torch.allclose(
        frame.band_weights.sum(dim=0),
        torch.ones(frame.mode_count, dtype=torch.float64),
        atol=1.0e-14,
        rtol=0,
    )
    operator_sum = sum(
        (frame.operator_matrix(index) for index in range(frame.scale_count)),
        start=torch.zeros(frame.node_count, frame.node_count, dtype=torch.float64),
    )
    assert torch.allclose(
        operator_sum,
        torch.eye(frame.node_count, dtype=torch.float64),
        atol=2.0e-12,
        rtol=0,
    )
    field = torch.randn(3, frame.node_count, 7, dtype=torch.float64)
    analysis = frame.analyze(field)
    reconstructed = frame.reconstruct(analysis)
    assert analysis.components.shape == (3, frame.scale_count, frame.node_count, 7)
    assert torch.allclose(reconstructed, field, atol=2.0e-12, rtol=1.0e-12)
    diagnostics = frame.diagnostics(field)
    assert diagnostics.reconstruction_relative_l2_error < 1.0e-12
    # A random field has energy outside the five retained modes: the test would
    # fail if S_perp were silently dropped.
    assert diagnostics.complement_energy_fraction > 0.0


def test_apply_reprojects_arbitrary_cell_outputs_to_each_scale() -> None:
    frame = _frame()
    fields = torch.randn(2, frame.scale_count, frame.node_count, 3, dtype=torch.float64)
    applied = frame.apply(fields)
    expected = torch.stack(
        [frame.apply_scale(index, fields[:, index]) for index in range(frame.scale_count)],
        dim=1,
    )
    assert torch.allclose(applied, expected, atol=1.0e-12, rtol=0)
    assert frame.synthesize(applied).shape == (2, frame.node_count, 3)


def test_low_precision_input_uses_stable_frame_arithmetic() -> None:
    frame = _frame()
    field = torch.randn(2, frame.node_count, 3).to(torch.bfloat16)
    reconstructed = frame.reconstruct(frame.analyze(field))
    assert reconstructed.dtype == torch.float32
    assert torch.allclose(reconstructed, field.float(), atol=3.0e-6, rtol=1.0e-6)


def test_parseval_sqrt_factors_are_tight_and_energy_additive() -> None:
    frame = _frame(factorization="parseval_sqrt")
    identity = torch.eye(frame.node_count, dtype=torch.float64)
    factor_square_sum = sum(
        (
            frame.factor_operator_matrix(index)
            @ frame.factor_operator_matrix(index)
            for index in range(frame.scale_count)
        ),
        start=torch.zeros_like(identity),
    )
    assert torch.allclose(factor_square_sum, identity, atol=3.0e-12, rtol=1.0e-12)

    field = torch.randn(3, frame.node_count, 7, dtype=torch.float64)
    analysis = frame.analyze(field)
    # A direct sum is deliberately not the Parseval synthesis.  The matching
    # factor must be applied once more before summation.
    assert not torch.allclose(frame.synthesize(analysis), field, atol=1.0e-8)
    assert torch.allclose(frame.reconstruct(analysis), field, atol=3.0e-12, rtol=1.0e-12)
    energy = frame.additive_relative_energies(field)
    centered_energy = frame.additive_relative_energies(field, center=True)
    component_energy = frame.additive_mass_energies(field)
    centered_component_energy = frame.additive_mass_energies(field, center=True)
    assert torch.all(energy >= 0) and torch.all(centered_energy >= 0)
    assert torch.allclose(energy.sum(), torch.tensor(1.0, dtype=energy.dtype), atol=1.0e-12)
    assert torch.allclose(
        centered_energy.sum(),
        torch.tensor(1.0, dtype=centered_energy.dtype),
        atol=1.0e-12,
    )
    assert torch.allclose(
        component_energy.sum(), frame.mass_energy(field), atol=2.0e-10, rtol=1.0e-12
    )
    assert torch.allclose(
        centered_component_energy.sum(),
        frame.mass_energy(field, center=True),
        atol=2.0e-10,
        rtol=1.0e-12,
    )
    assert float(frame.additive_energy_closure_relative_error(field)) < 1.0e-12
    assert (
        float(frame.additive_energy_closure_relative_error(field, center=True))
        < 1.0e-12
    )


def test_raw_energy_closure_diagnostic_cannot_be_normalized_away() -> None:
    base = _frame(factorization="parseval_sqrt")
    perturbed_basis = base.eigenvectors.clone()
    perturbed_basis[:, 0] *= 1.0001
    frame = ParsevalResolventFrame(
        base.eigenvalues,
        perturbed_basis,
        base.mass,
        radii=(0.4, 1.0, 2.5),
        factorization="parseval_sqrt",
    )
    field = torch.randn(2, frame.node_count, 3, dtype=torch.float64)
    # Shares always sum to one by definition, but the independent raw-energy
    # comparison must expose this accepted small basis perturbation.
    assert torch.allclose(
        frame.additive_relative_energies(field).sum(),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1.0e-12,
    )
    assert float(frame.additive_energy_closure_relative_error(field)) > 1.0e-8


def test_parseval_roundtrip_and_band_factors_are_differentiable() -> None:
    frame = _frame(factorization="parseval_sqrt")
    field = torch.randn(
        2, frame.node_count, 4, dtype=torch.float64, requires_grad=True
    )
    reconstructed = frame.reconstruct(frame.analyze(field))
    loss = reconstructed.square().mean()
    loss.backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
    assert float(field.grad.abs().sum()) > 0.0


def test_frame_and_geometry_features_are_invariant_to_degenerate_basis_rotation() -> None:
    original = _frame()
    rotated = _frame(rotate_degenerate=True)
    field = torch.randn(2, original.node_count, 4, dtype=torch.float64)
    assert torch.allclose(
        original.analyze(field).components,
        rotated.analyze(field).components,
        atol=2.0e-12,
        rtol=1.0e-12,
    )
    xyz = torch.arange(original.node_count * 3, dtype=torch.float64).reshape(-1, 3)
    first = basis_invariant_geometry_features(original, node_xyz_mm=xyz)
    second = basis_invariant_geometry_features(rotated, node_xyz_mm=xyz)
    assert torch.allclose(first, second, atol=2.0e-12, rtol=1.0e-12)


def test_fixed_spatial_permutation_null_is_reproducible_and_reconstructs_full_fields() -> None:
    frame = _frame()
    assert torch.equal(
        fixed_spatial_permutation(frame.node_count, seed=19),
        fixed_spatial_permutation(frame.node_count, seed=19),
    )
    first, first_permutation = frame.spatial_permutation_null(seed=19)
    second, second_permutation = frame.spatial_permutation_null(seed=19)
    assert torch.equal(first_permutation, second_permutation)
    assert not torch.equal(first_permutation, torch.arange(frame.node_count))
    assert torch.allclose(first.eigenvectors, second.eigenvectors)
    assert not torch.allclose(first.operator_matrix(1), frame.operator_matrix(1))
    field = torch.randn(2, frame.node_count, 2, dtype=torch.float64)
    assert torch.allclose(first.reconstruct(first.analyze(field)), field, atol=2.0e-12)


def test_spatial_permutation_null_preserves_parseval_factorization() -> None:
    frame = _frame(factorization="parseval_sqrt")
    null, _ = frame.spatial_permutation_null(seed=29)
    assert null.factorization == "parseval_sqrt"
    field = torch.randn(2, frame.node_count, 3, dtype=torch.float64)
    assert torch.allclose(
        null.reconstruct(null.analyze(field)), field, atol=3.0e-12, rtol=1.0e-12
    )
