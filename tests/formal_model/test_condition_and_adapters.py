from __future__ import annotations

import pytest
import torch

from mri_pet_geomc.formal.model import (
    AcquisitionConditionEncoder,
    ConditionedFeatureAdapter3D,
    MRIPETResidualStems,
    MetadataVocabulary,
    NearIdentityResidualStem,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "modality": "MRI",
            "mri_sequence": "T1w",
            "mri_product": "native",
            "derived_flag": "false",
            "native_spacing_x": 1.0,
            "native_spacing_y": 1.0,
            "native_spacing_z": 1.2,
        },
        {
            "modality": "PET",
            "pet_family": "amyloid",
            "exact_tracer": "AV45",
            "derived_flag": "false",
            "native_spacing_x": 2.0,
            "native_spacing_y": 2.0,
            "native_spacing_z": 2.0,
        },
    ]


def test_vocabulary_is_deterministic_roundtrippable_and_acquisition_only() -> None:
    first = MetadataVocabulary.from_records(_records())
    second = MetadataVocabulary.from_records(list(reversed(_records())))
    assert first.as_dict() == second.as_dict()
    restored = MetadataVocabulary.from_dict(first.as_dict())
    batch = restored.encode(_records())
    assert batch.categorical.shape == (2, 8)
    assert batch.continuous.shape == (2, 6)
    encoder = AcquisitionConditionEncoder(restored, output_dim=16, categorical_embedding_dim=4)
    output = encoder(batch)
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="Forbidden"):
        MetadataVocabulary.from_records([{"modality": "MRI", "quality_score": 0.9}])
    with pytest.raises(ValueError, match="Forbidden"):
        first.encode([{"modality": "MRI", "site": "A"}])
    mp2rage = MetadataVocabulary.from_records(
        [{"modality": "MRI", "mp2rage_component": "UNI"}]
    )
    assert mp2rage.encode(
        [{"modality": "MRI", "mp2rage_component": "UNI"}]
    ).categorical.shape == (1, 8)
    with pytest.raises(ValueError, match="Unsupported"):
        first.encode([{"modality": "MRI", "scanner_vendor": "unknown"}])


def test_stems_and_conditioned_feature_adapter_start_exactly_as_identity() -> None:
    torch.manual_seed(3)
    volume = torch.randn(2, 1, 4, 4, 4)
    stem = NearIdentityResidualStem(hidden_channels=2)
    assert torch.equal(stem(volume), volume)

    stems = MRIPETResidualStems(hidden_channels=2)
    mixed = stems(volume, torch.tensor([stems.MRI, stems.PET]))
    assert torch.equal(mixed, volume)

    feature = torch.randn(2, 4, 2, 2, 2)
    adapter = ConditionedFeatureAdapter3D(4, 8, bottleneck_dim=2)
    adapted = adapter(feature, torch.randn(2, 8))
    assert torch.equal(adapted, feature)


def test_modality_stems_keep_separate_trainable_branches() -> None:
    stems = MRIPETResidualStems(hidden_channels=2)
    with torch.no_grad():
        stems.mri.residual[-1].weight.fill_(0.1)
        stems.pet.residual[-1].weight.fill_(-0.1)
    volume = torch.ones(2, 1, 4, 4, 4)
    result = stems(volume, torch.tensor([stems.MRI, stems.PET]))
    assert not torch.equal(result[0], result[1])
