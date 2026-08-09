from __future__ import annotations

import pytest
import torch

from mri_pet_geomc.geometry.frame import ParsevalResolventFrame
from mri_pet_geomc.model.geomc import GeoMCBlock, GeoMCCore, SharedScaleUpdate


def _frame(*, factorization: str = "parseval_sqrt") -> ParsevalResolventFrame:
    return ParsevalResolventFrame(
        torch.tensor([0.0, 0.2, 0.7, 1.6], dtype=torch.float64),
        torch.eye(8, dtype=torch.float64)[:, :4],
        torch.ones(8, dtype=torch.float64),
        radii=(1.0, 2.0, 4.0),
        normalize_spectrum=False,
        factorization=factorization,  # type: ignore[arg-type]
    )


def _core() -> GeoMCCore:
    return GeoMCCore(
        _frame(),
        6,
        num_blocks=3,
        hidden_dim=18,
        scale_embedding_dim=5,
        residual_scale=0.1,
    )


def test_core_exposes_one_clear_model_setup() -> None:
    core = _core()
    assert core.routing_mode == "nodewise_scale_mixing"
    assert core.frame.factorization == "parseval_sqrt"
    assert core.input_dim == 6
    assert len(core.blocks) == 3
    assert all(isinstance(block, GeoMCBlock) for block in core.blocks)
    assert all(isinstance(block.scale_update, SharedScaleUpdate) for block in core.blocks)


def test_core_rejects_invalid_frame_or_too_few_blocks() -> None:
    with pytest.raises(ValueError, match="parseval_sqrt"):
        GeoMCCore(_frame(factorization="non_tight"), 6)
    with pytest.raises(ValueError, match="at least three"):
        GeoMCCore(_frame(), 6, num_blocks=2)


def test_routing_starts_as_identity_and_can_learn_signed_local_mixing() -> None:
    torch.manual_seed(7)
    core = _core()
    block = core.blocks[0]
    input_field = torch.randn(2, core.frame.node_count, core.d_model)
    state = torch.randn_like(input_field)
    components = core.frame.analyze(state).components
    input_scale_components = core.frame.analyze(input_field).components

    initial = block._routing_matrix(
        state, input_field, components, input_scale_components
    )
    identity = torch.eye(core.frame.scale_count)[None, None].expand_as(initial)
    assert torch.equal(initial, identity)

    with torch.no_grad():
        final = block.routing_mlp[-1]
        assert isinstance(final, torch.nn.Linear)
        final.bias.copy_(torch.linspace(-0.8, 0.8, core.frame.scale_count**2))
        final.weight[0, 0] = 0.5
    routed = block._routing_matrix(
        state, input_field, components, input_scale_components
    )
    assert torch.any(routed < 0)
    assert torch.any(routed != identity)
    assert float(routed.detach().std(dim=1, unbiased=False).mean()) > 0.0


def test_core_forward_is_finite_differentiable_and_reports_frame_reconstruction() -> None:
    torch.manual_seed(11)
    core = _core()
    input_field = torch.randn(
        2, core.frame.node_count, core.input_dim, requires_grad=True
    )
    geometry_embedding = torch.randn(core.frame.node_count, core.d_model) * 0.03
    output, diagnostics = core(input_field, geometry_embedding)
    assert output.shape == input_field.shape
    assert torch.isfinite(output).all()
    assert diagnostics.routing_mode == "nodewise_scale_mixing"
    assert diagnostics.block_count == 3
    assert len(diagnostics.blocks) == 3
    assert all(item.frame_factorization == "parseval_sqrt" for item in diagnostics.blocks)
    assert all(item.reconstruction_relative_l2 < 1.0e-5 for item in diagnostics.blocks)
    output.square().mean().backward()
    assert input_field.grad is not None and torch.isfinite(input_field.grad).all()
    assert float(input_field.grad.abs().sum()) > 0.0
    routing_gradients = [
        parameter.grad
        for block in core.blocks
        for parameter in block.routing_mlp.parameters()
        if parameter.grad is not None
    ]
    assert routing_gradients
    assert all(torch.isfinite(value).all() for value in routing_gradients)


def test_shared_scale_update_keeps_a_direct_trainable_input_field_path() -> None:
    torch.manual_seed(17)
    update = SharedScaleUpdate(d_model=4, scale_dim=3, hidden_dim=12)
    shape = (2, 5, 7, 4)
    input_field = torch.randn(2, 1, 7, 4).expand(-1, 5, -1, -1).clone()
    input_field.requires_grad_(True)
    output = update(
        torch.zeros(shape),
        torch.zeros(shape),
        torch.zeros(shape),
        input_field,
        torch.zeros(2, 5, 7, 3),
    )
    output.square().mean().backward()
    assert input_field.grad is not None
    assert torch.isfinite(input_field.grad).all()
    assert float(input_field.grad.abs().sum()) > 0.0


def test_state_dict_uses_clear_module_names() -> None:
    keys = set(_core().state_dict())
    required = {
        "input_normalization.weight",
        "input_normalization.bias",
        "blocks.0.scale_embedding",
        "blocks.0.routing_mlp.0.weight",
        "blocks.0.routing_mlp.1.weight",
        "blocks.0.routing_mlp.3.weight",
        "blocks.0.scale_update.normalization.weight",
        "blocks.0.scale_update.candidate.0.weight",
        "blocks.0.scale_update.gate.0.weight",
        "blocks.0.residual_scale",
    }
    assert required <= keys
    unclear_fragments = ("gamma", "innovation", "evidence", "typed")
    assert not any(fragment in key for key in keys for fragment in unclear_fragments)