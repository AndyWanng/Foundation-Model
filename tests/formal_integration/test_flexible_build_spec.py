"""Configurable formal widths, without loading SAT3D assets or allocating CUDA."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from mri_pet_geomc.formal.integration import (
    FormalModelBuildSpec,
    _formal_model_from_parts,
    _tiny_geometry,
)
from mri_pet_geomc.formal.model import FormalObservationBatch, MetadataVocabulary
from mri_pet_geomc.training.jepa import masked_huber_objective


def _spec(**overrides: object) -> FormalModelBuildSpec:
    values: dict[str, object] = {
        "sat3d_code_root": Path("not-loaded-sat3d"),
        "sat3d_checkpoint": Path("not-loaded-sat3d.pt"),
        "fem_geometry_path": Path("not-loaded-fem.pt"),
        "reference_shape": (128, 128, 128),
        "reference_affine": (
            (2.0, 0.0, 0.0, 0.0),
            (0.0, 2.0, 0.0, 0.0),
            (0.0, 0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        "expected_checkpoint_sha256": "0" * 64,
        "expected_source_tree_sha256": "1" * 64,
    }
    values.update(overrides)
    return FormalModelBuildSpec(**values)


def test_existing_default_build_spec_remains_unchanged() -> None:
    spec = _spec()
    assert (spec.latent_dim, spec.condition_dim, spec.relation_dim) == (128, 128, 32)
    assert (spec.predictor_dim, spec.predictor_depth, spec.predictor_heads) == (256, 6, 8)
    assert spec.max_queries == 128


@pytest.mark.parametrize(
    "overrides",
    [
        {"latent_dim": 384, "geomc_hidden_dim": 768, "fusion_hidden_dim": 384},
        {
            "latent_dim": 96,
            "condition_dim": 48,
            "relation_dim": 12,
            "predictor_dim": 192,
            "predictor_depth": 2,
            "predictor_heads": 6,
            "max_queries": 256,
        },
        {"predictor_dim": 6, "predictor_heads": 2, "predictor_depth": 1, "max_queries": 1},
        {"max_queries": 512, "predictor_drop_path": 0.0, "warmup_coverages": 0},
        {"geometry_embedding_initial_scale": 0.0, "geomc_residual_scale": 0.0},
        {"geometry_embedding_initial_scale": -0.05, "geomc_residual_scale": -0.1},
    ],
)
def test_build_spec_accepts_independent_widths_and_valid_boundaries(overrides: dict) -> None:
    spec = _spec(**overrides)
    for name, expected in overrides.items():
        assert getattr(spec, name) == expected


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"latent_dim": 0}, "latent_dim"),
        ({"latent_dim": 128.5}, "latent_dim"),
        ({"condition_dim": True}, "condition_dim"),
        ({"condition_dim": -1}, "condition_dim"),
        ({"relation_dim": 0}, "relation_dim"),
        ({"fusion_hidden_dim": 0}, "fusion_hidden_dim"),
        ({"geomc_hidden_dim": 0}, "geomc_hidden_dim"),
        ({"geomc_blocks": 2}, "at least 3"),
        ({"predictor_heads": 0}, "predictor_heads"),
        ({"predictor_heads": 7}, "divisible"),
        ({"predictor_dim": 4, "predictor_heads": 2}, "at least 6"),
        ({"predictor_depth": 0}, "predictor_depth"),
        ({"max_queries": 0}, "max_queries"),
        ({"max_queries": 513}, "max_queries"),
        ({"predictor_drop_path": 1.0}, "predictor_drop_path"),
        ({"predictor_drop_path": float("nan")}, "predictor_drop_path"),
        ({"predictor_mlp_ratio": float("inf")}, "predictor_mlp_ratio"),
        ({"predictor_mlp_ratio": 0.00001}, "hidden dimension"),
        ({"adapter_bottleneck_ratio": float("nan")}, "adapter_bottleneck_ratio"),
        ({"mask_hidden_fraction": float("nan")}, "mask_hidden_fraction"),
        ({"interpolation_sigma_mm": float("nan")}, "interpolation_sigma_mm"),
        ({"geometry_embedding_initial_scale": float("nan")}, "initial_scale"),
        ({"geomc_residual_scale": float("inf")}, "residual_scale"),
        ({"resolvent_radii": ()}, "resolvent_radii"),
        ({"resolvent_radii": (3.0, 3.0)}, "resolvent_radii"),
        ({"resolvent_radii": (3.0, float("inf"))}, "resolvent_radii"),
        ({"sat3d_stage_multipliers": (0.5, 1.0)}, "four"),
        ({"sat3d_stage_multipliers": (0.5, 0.7, float("nan"), 1.0)}, "multipliers"),
        ({"sat3d_patch_embed_multiplier": 0.0}, "patch_embed_multiplier"),
        ({"weight_decay": float("nan")}, "decay/scheduler"),
        ({"final_lr_ratio": float("inf")}, "decay/scheduler"),
        ({"warmup_coverages": -1}, "warmup_coverages"),
        ({"ema_initial_momentum": float("nan")}, "EMA"),
        ({"reference_shape": (64, 64, 64)}, "128\\^3"),
        ({"reference_affine": ((1.0, 0.0), (0.0, 1.0))}, "finite 4x4"),
        ({"activation_checkpointing": "yes"}, "activation_checkpointing"),
        ({"geometry_embedding_trainable_scale": 1}, "trainable_scale"),
        ({"trainability": "last_stage"}, "trainability"),
    ],
)
def test_build_spec_rejects_incompatible_or_nonfinite_values(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _spec(**overrides)


def test_from_config_propagates_nondefault_widths_without_rewriting_them() -> None:
    path = Path(__file__).parents[2] / "configs" / "workstation_formal.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["model"]["latent_dim"] = 384
    config["model"]["projector"]["output_dim"] = 384
    config["model"]["geometry"]["hidden_dim"] = 768
    config["model"]["fusion"]["hidden_dim"] = 384
    config["model"]["fusion"]["relation_embedding_dim"] = 12
    config["metadata_conditioning"]["dimension"] = 48
    config["model"]["predictor"].update(
        {"embed_dim": 192, "heads": 6, "depth": 2, "max_queries": 256}
    )
    config["masking"]["query_tokens"] = 256
    spec = FormalModelBuildSpec.from_config(
        config,
        sat3d_code_root="not-loaded-sat3d",
        sat3d_checkpoint="not-loaded-sat3d.pt",
        fem_geometry_path="not-loaded-fem.pt",
        reference_affine=_spec().reference_affine,
    )
    assert spec.latent_dim == 384
    assert spec.geomc_hidden_dim == 768
    assert spec.fusion_hidden_dim == 384
    assert spec.condition_dim == 48
    assert spec.relation_dim == 12
    assert (spec.predictor_dim, spec.predictor_heads, spec.predictor_depth) == (192, 6, 2)
    assert spec.max_queries == 256


class _EightVoxelTestEncoder(nn.Module):
    """8^3 input -> eight 384-channel tokens; no real SAT3D modules/assets."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.projection = nn.Conv3d(1, 384, kernel_size=1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(volume))


@pytest.mark.parametrize(
    "latent_dim,condition_dim,relation_dim,hidden_dim,predictor_dim,predictor_heads",
    [(384, 48, 12, 768, 256, 8), (48, 20, 8, 64, 96, 3)],
)
def test_configurable_widths_forward_backward_on_eight_node_cpu_field(
    latent_dim: int,
    condition_dim: int,
    relation_dim: int,
    hidden_dim: int,
    predictor_dim: int,
    predictor_heads: int,
) -> None:
    # The largest case has production latent width but only eight input tokens,
    # eight synthetic nodes and two queries. Bound CPU parallelism explicitly.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            spec = _spec(
                latent_dim=latent_dim,
                condition_dim=condition_dim,
                relation_dim=relation_dim,
                geomc_hidden_dim=hidden_dim,
                fusion_hidden_dim=latent_dim,
                predictor_dim=predictor_dim,
                predictor_heads=predictor_heads,
            )
            # Preserve the production build validation, but deliberately reduce
            # depth/geometry for a bounded shape-and-gradient test.
            spec = replace(
                spec,
                geomc_blocks=3,
                predictor_depth=1,
                predictor_drop_path=0.0,
                max_queries=4,
                condition_categorical_embedding_dim=2,
                stem_hidden_channels=2,
                geometry_embedding_hidden_dim=8,
                geomc_scale_embedding_dim=4,
                resolvent_radii=(1.0, 2.0),
                interpolation_neighbors=1,
                interpolation_sigma_mm=1.0,
            )
            records = [{"modality": "MRI", "mri_sequence": "T1w"}]
            vocabulary = MetadataVocabulary.from_records(records)
            geometry = _tiny_geometry()
            names = (
                "latent_dim", "condition_dim", "relation_dim",
                "condition_categorical_embedding_dim", "stem_hidden_channels",
                "adapter_bottleneck_ratio", "resolvent_radii", "geomc_blocks",
                "geomc_hidden_dim", "geomc_scale_embedding_dim", "geomc_residual_scale",
                "fusion_hidden_dim", "geometry_embedding_hidden_dim",
                "geometry_embedding_initial_scale", "geometry_embedding_trainable_scale",
                "predictor_dim", "predictor_depth", "predictor_heads", "predictor_mlp_ratio",
                "predictor_drop_path", "max_queries", "interpolation_neighbors",
                "interpolation_sigma_mm", "geometry_feature_type",
            )
            model = _formal_model_from_parts(
                encoder=_EightVoxelTestEncoder(),
                encoder_channels=384,
                token_xyz_mm=geometry.node_xyz_mm,
                geometry=geometry,
                vocabulary=vocabulary,
                **{name: getattr(spec, name) for name in names},
            )
            full_volume = torch.randn(1, 1, 8, 8, 8)
            masked_volume = full_volume.clone()
            masked_volume[:, :, :4, :4, :] = 0.0
            coverage = torch.ones(1, 8)
            coverage[:, :2] = 0.0
            anchor = FormalObservationBatch(
                volume=masked_volume,
                token_coverage=coverage,
                condition=vocabulary.encode(records),
                modality_kind=torch.zeros(1, dtype=torch.long),
                observation_uid=("synthetic-width-contract-only",),
            )
            output = model(
                anchor,
                full_volume,
                target_support=torch.ones(1, 8),
                query_indices=torch.tensor([[0, 1]]),
                query_valid=torch.ones(1, 2, dtype=torch.bool),
                relation_codes=torch.zeros(1, dtype=torch.long),
                return_diagnostics=False,
            )
            assert output.latent_field.shape == (1, 8, latent_dim)
            assert output.predicted_latents.shape == output.target_latents.shape == (
                1, 2, latent_dim
            )
            assert model.predictor.context_projection.in_features == latent_dim
            assert model.predictor.target_condition_projection.in_features == condition_dim
            assert model.predictor.relation_projection.in_features == relation_dim
            loss = masked_huber_objective(
                output.predicted_latents, output.target_latents, output.query_valid
            ).loss
            assert torch.isfinite(loss)
            loss.backward()
            for module in (model.encoder_bundle.online, model.fusion, model.core, model.predictor):
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
                assert any(bool(torch.any(gradient != 0)) for gradient in gradients)
            assert not output.target_latents.requires_grad
            assert all(p.grad is None for p in model.encoder_bundle.target.parameters())
            assert "geomc" in model.encoder_bundle.ema_contract()["excluded"]
    finally:
        torch.set_num_threads(previous_threads)
