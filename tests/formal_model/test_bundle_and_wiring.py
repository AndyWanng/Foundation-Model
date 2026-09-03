from __future__ import annotations

import torch
from torch import nn

from mri_pet_geomc.field.geometry_embedding import GeometryEmbedding
from mri_pet_geomc.formal.model import (
    AcquisitionConditionEncoder,
    AnchorCentricFusion,
    ConditionedFeatureAdapter3D,
    ContentGatedTokenProjector,
    FormalEncoderBundle,
    FormalGeoMCJEPAModel,
    FormalObservationBatch,
    MRIPETResidualStems,
    MetadataVocabulary,
    OnlineEMABundle,
    PackedCompanionBatch,
    QueryAwareJEPA3DPredictor,
)
from mri_pet_geomc.geometry.frame import ParsevalResolventFrame
from mri_pet_geomc.geometry.interpolation import FixedSpatialInterpolator, InterpolationPlan
from mri_pet_geomc.model.geomc import GeoMCCore


class TinyFeatureEncoder(nn.Module):
    """Test-only 8^3 -> 2^3 feature encoder; never instantiates SAT3D."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((2, 2, 2))

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.nn.functional.silu(self.conv(volume)))


def _identity_interpolator(count: int) -> FixedSpatialInterpolator:
    return FixedSpatialInterpolator(
        InterpolationPlan(
            source_indices=torch.arange(count).reshape(count, 1),
            kernel_weights=torch.ones(count, 1),
            source_cell_volume=torch.ones(count),
            full_weight_sum=torch.ones(count),
        )
    )


def _records() -> list[dict[str, object]]:
    return [
        {
            "modality": "MRI",
            "mri_sequence": "T1w",
            "mri_product": "native",
            "native_spacing_x": 1.0,
            "native_spacing_y": 1.0,
            "native_spacing_z": 1.0,
        },
        {
            "modality": "PET",
            "pet_family": "FDG",
            "exact_tracer": "FDG",
            "native_spacing_x": 2.0,
            "native_spacing_y": 2.0,
            "native_spacing_z": 2.0,
        },
    ]


def _model_and_condition() -> tuple[FormalGeoMCJEPAModel, object]:
    vocabulary = MetadataVocabulary.from_records(_records())
    condition_encoder = AcquisitionConditionEncoder(
        vocabulary, output_dim=8, categorical_embedding_dim=2
    )
    online = FormalEncoderBundle(
        stems=MRIPETResidualStems(hidden_channels=2),
        condition_encoder=condition_encoder,
        encoder=TinyFeatureEncoder(),
        feature_adapter=ConditionedFeatureAdapter3D(4, 8, bottleneck_dim=2),
        projector=ContentGatedTokenProjector(4, 8, 8),
    )
    pair = OnlineEMABundle(online)
    frame = ParsevalResolventFrame(
        torch.tensor([0.0, 0.1, 0.4, 1.0], dtype=torch.float64),
        torch.eye(8, dtype=torch.float64)[:, :4],
        torch.ones(8, dtype=torch.float64),
        radii=(1.0, 2.0),
        normalize_spectrum=False,
        factorization="parseval_sqrt",
    )
    core = GeoMCCore(frame, 8, num_blocks=3, hidden_dim=16, scale_embedding_dim=4)
    interpolator = _identity_interpolator(8)
    fusion = AnchorCentricFusion(
        interpolator,
        feature_dim=8,
        condition_dim=8,
        relation_dim=4,
        output_dim=8,
        hidden_dim=12,
    )
    predictor = QueryAwareJEPA3DPredictor(
        latent_dim=8,
        condition_dim=8,
        relation_dim=4,
        d_model=32,
        depth=1,
        num_heads=4,
        drop_path_rate=0.0,
        max_queries=4,
    )
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
            [2.0, 2.0, 2.0],
        ]
    )
    model = FormalGeoMCJEPAModel(
        encoder_bundle=pair,
        fusion=fusion,
        core=core,
        geometry_embedding=GeometryEmbedding(frame, 8, feature_type="none"),
        predictor=predictor,
        token_xyz_mm=coordinates,
        node_xyz_mm=coordinates,
        relation_count=4,
    )
    return model, vocabulary.encode(_records())


def test_formal_wiring_supports_b2_optional_companion_and_query_packing() -> None:
    torch.manual_seed(13)
    model, condition = _model_and_condition()
    anchor_volume = torch.randn(2, 1, 8, 8, 8)
    anchor = FormalObservationBatch(
        volume=anchor_volume,
        token_coverage=torch.ones(2, 8),
        condition=condition,
        modality_kind=torch.tensor([0, 1]),
        observation_uid=("mri-anchor", "pet-anchor"),
    )
    companion = PackedCompanionBatch(
        observation=FormalObservationBatch(
            volume=torch.randn(1, 1, 8, 8, 8),
            token_coverage=torch.ones(1, 8),
            condition=condition.index_select(torch.tensor([0])),
            modality_kind=torch.tensor([0]),
            observation_uid=("mri-companion",),
        ),
        anchor_indices=torch.tensor([0], dtype=torch.long),
    )
    output = model(
        anchor,
        torch.randn_like(anchor_volume),
        target_support=torch.ones(2, 8),
        query_indices=torch.tensor([[0, 1, -1], [2, 3, 4]]),
        query_valid=torch.tensor([[True, True, False], [True, True, True]]),
        relation_codes=torch.tensor([1, 0]),
        companion=companion,
        return_diagnostics=False,
    )
    assert output.predicted_latents.shape == (2, 3, 8)
    assert output.target_latents.shape == (2, 3, 8)
    assert torch.all(output.predicted_latents[0, 2] == 0)
    assert torch.all(output.node_field.companion_gate[1] == 0)
    output.predicted_latents.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.encoder_bundle.online.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert not any(parameter.requires_grad for parameter in model.encoder_bundle.target.parameters())


def test_ema_scope_is_explicit_and_excludes_geomc_and_predictor() -> None:
    model, _ = _model_and_condition()
    contract = model.encoder_bundle.ema_contract()
    assert contract["target_trainable_parameters"] == 0
    assert "encoder" in contract["included"]
    assert "predictor" in contract["excluded"]
    assert "geomc" in contract["excluded"]
    target_before = {
        name: value.detach().clone()
        for name, value in model.encoder_bundle.target.state_dict().items()
    }
    with torch.no_grad():
        next(model.encoder_bundle.online.parameters()).add_(0.5)
    model.update_target(0.5)
    assert any(
        not torch.equal(target_before[name], value)
        for name, value in model.encoder_bundle.target.state_dict().items()
        if value.dtype.is_floating_point
    )
    specification = model.model_specification()
    assert specification["quality_or_reliability_input"] is False
    assert specification["predictor"]["raw_encoder_skip"] is False
    assert specification["companions_per_anchor_max"] == 1
