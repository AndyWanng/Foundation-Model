from __future__ import annotations

import inspect

import torch

from mri_pet_geomc.field.geometry_embedding import GeometryEmbedding
from mri_pet_geomc.geometry.frame import ParsevalResolventFrame
from mri_pet_geomc.geometry.interpolation import (
    FixedSpatialInterpolator,
    InterpolationPlan,
)
from mri_pet_geomc.model.foundation import (
    GeoMCFoundationModel,
    ObservationBatch,
    QueryDecoder,
)
from mri_pet_geomc.model.geomc import GeoMCCore
from mri_pet_geomc.model.trainable_sat3d import TrainableSAT3DEncoder


def _identity_interpolator(count: int) -> FixedSpatialInterpolator:
    return FixedSpatialInterpolator(
        InterpolationPlan(
            source_indices=torch.arange(count).reshape(count, 1),
            kernel_weights=torch.ones(count, 1),
            source_cell_volume=torch.ones(count),
            full_weight_sum=torch.ones(count),
        )
    )


def _model(*, seed: int = 3) -> GeoMCFoundationModel:
    encoder = TrainableSAT3DEncoder.tiny(
        input_size=8,
        hidden_channels=4,
        output_channels=8,
        output_grid=2,
        trainability="all",
        seed=seed,
    )
    frame = ParsevalResolventFrame(
        torch.tensor([0.0, 0.1, 0.4, 1.0], dtype=torch.float64),
        torch.eye(8, dtype=torch.float64)[:, :4],
        torch.ones(8, dtype=torch.float64),
        radii=(1.0, 2.0),
        normalize_spectrum=False,
        factorization="parseval_sqrt",
    )
    core = GeoMCCore(
        frame,
        8,
        num_blocks=3,
        hidden_dim=16,
        scale_embedding_dim=4,
    )
    geometry_embedding = GeometryEmbedding(core.frame, 8, feature_type="none")
    return GeoMCFoundationModel(
        online_encoder=encoder,
        encoder_output_dim=8,
        latent_dim=8,
        metadata_dim=4,
        token_to_node=_identity_interpolator(8),
        node_to_token=_identity_interpolator(8),
        core=core,
        geometry_embedding=geometry_embedding,
    )


def _observation(
    volume: torch.Tensor,
    metadata: torch.Tensor,
    observation_uid: str,
) -> ObservationBatch:
    return ObservationBatch(
        volume=volume,
        token_coverage=torch.ones(1, 8),
        token_reliability=torch.ones(1, 8),
        metadata=metadata,
        observation_uid=observation_uid,
    )


def _query_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[True, True, True, True, False, False, False, False]]),
        torch.ones(1, 8),
    )


def test_latent_field_is_independent_of_observation_order_and_target_values() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    mri = torch.randn(1, 1, 8, 8, 8)
    pet = torch.randn(1, 1, 8, 8, 8)
    target_a = torch.randn(1, 1, 8, 8, 8)
    target_b = torch.randn(1, 1, 8, 8, 8)
    mri_metadata = torch.tensor([1.0, 0.0, 1.0, 1.0])
    pet_metadata = torch.tensor([0.0, 1.0, 1.0, 1.0])
    query, support = _query_inputs()
    observations = [
        _observation(mri, mri_metadata, "mri"),
        _observation(pet, pet_metadata, "pet"),
    ]

    first = model(
        observations,
        target_a,
        target_metadata=pet_metadata,
        query_mask=query,
        target_support=support,
    )
    changed_target = model(
        observations,
        target_b,
        target_metadata=pet_metadata,
        query_mask=query,
        target_support=support,
    )
    reversed_observations = model(
        list(reversed(observations)),
        target_a,
        target_metadata=pet_metadata,
        query_mask=query,
        target_support=support,
    )
    changed_target_type = model(
        observations,
        target_a,
        target_metadata=mri_metadata,
        query_mask=query,
        target_support=support,
    )

    assert torch.allclose(first.latent_field, changed_target.latent_field, atol=1.0e-6)
    assert torch.allclose(first.predicted_latents, changed_target.predicted_latents, atol=1.0e-6)
    assert not torch.allclose(first.target_latents, changed_target.target_latents, atol=1.0e-6)
    assert torch.allclose(first.latent_field, reversed_observations.latent_field, atol=1.0e-6)
    assert torch.allclose(first.latent_field, changed_target_type.latent_field, atol=1.0e-6)
    assert first.core_diagnostics is not None
    assert first.decoder_diagnostics is not None


def test_prediction_gradient_reaches_shared_encoder_through_node_features_and_core() -> None:
    torch.manual_seed(11)
    model = _model(seed=5).train()
    volume = torch.randn(1, 1, 8, 8, 8)
    metadata = torch.tensor([1.0, 0.0, 1.0, 1.0])
    output = model(
        [_observation(volume, metadata, "mri")],
        torch.randn_like(volume),
        target_metadata=metadata,
        query_mask=torch.ones(1, 8, dtype=torch.bool),
        target_support=torch.ones(1, 8),
        return_diagnostics=False,
    )
    output.predicted_latents.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.online_encoder.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert sum(float(value.abs().sum()) for value in gradients) > 0.0


def test_ema_target_is_frozen_and_updates_only_through_explicit_ema() -> None:
    model = _model(seed=13).train()
    assert not any(parameter.requires_grad for parameter in model.target_encoder.parameters())
    assert not model.target_encoder.training
    target_before = {
        name: value.detach().clone()
        for name, value in model.target_encoder.state_dict().items()
    }
    with torch.no_grad():
        first_online = next(model.online_encoder.parameters())
        first_online.add_(0.25)
    model.update_target(0.5)
    target_after = model.target_encoder.state_dict()
    assert any(
        not torch.equal(target_before[name], target_after[name])
        for name in target_before
        if target_before[name].dtype.is_floating_point
    )
    assert not model.target_encoder.training


def test_model_specification_uses_clear_current_terms() -> None:
    specification = _model().model_specification()
    assert specification["name"] == "shared_sat3d_geomc"
    assert specification["routing_mode"] == "nodewise_scale_mixing"
    assert specification["observation_aggregation"] == "reliability_weighted_mean"
    assert specification["decoder_type"] == "mlp"
    assert specification["frame_factorization"] == "parseval_sqrt"
    assert specification["latent_domain"] == "fem_nodes"
    assert specification["latent_field_uses_observation_metadata"] is True
    assert specification["latent_field_uses_target_metadata"] is False
    assert specification["decoder_uses_target_metadata"] is True
    assert specification["target_trainable_parameters"] == 0


def test_query_decoder_is_an_mlp_with_clear_diagnostics() -> None:
    decoder = QueryDecoder(
        _identity_interpolator(8),
        latent_dim=8,
        metadata_dim=4,
    )
    result, diagnostics = decoder(
        torch.randn(2, 8, 8),
        torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]),
        return_diagnostics=True,
    )
    assert result.shape == (2, 8, 8)
    assert diagnostics.input_token_rms > 0.0
    assert diagnostics.output_token_rms > 0.0
    assert "residual_scale" not in decoder.state_dict()


def test_forward_exposes_only_plain_observation_and_target_inputs() -> None:
    parameters = tuple(inspect.signature(GeoMCFoundationModel.forward).parameters)
    assert parameters == (
        "self",
        "observations",
        "target_volume",
        "target_metadata",
        "query_mask",
        "target_support",
        "return_diagnostics",
    )