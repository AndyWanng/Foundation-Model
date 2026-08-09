from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mri_pet_geomc.data.observations import (
    CrossModalLink,
    ObservationVolume,
    Observation,
    ObservationValidationError,
    ObservationRelationSampler,
    RelationExample,
    build_observation_tables,
)
from mri_pet_geomc.training.engine import (
    _observation_metadata,
    _make_sampler,
    prepare_example,
)


def _paired_tables(subject_count: int = 4, *, split: str = "train"):
    observations = []
    links = []
    splits = {}
    for index in range(subject_count):
        subject = f"S{index:02d}"
        splits[subject] = split
        mri_uid = f"{subject}-mri"
        pet_uid = f"{subject}-pet"
        for uid, modality in ((mri_uid, "mri"), (pet_uid, "pet")):
            observations.append(
                Observation(
                    observation_uid=uid,
                    subject_id=subject,
                    session_id="ses-1",
                    split=split,
                    modality=modality,
                    volume_path=Path(f"{uid}.nii.gz"),
                    brain_mask_path=Path(f"{uid}-mask.nii.gz"),
                )
            )
        links.append(
            CrossModalLink(
                link_uid=f"{subject}-link",
                subject_id=subject,
                split=split,
                mri_observation_uid=mri_uid,
                pet_observation_uid=pet_uid,
                verified=True,
                verification_method="unit_test_identity",
            )
        )
    return build_observation_tables(observations, links, splits)


def test_verified_unpaired_and_mismatched_relations_are_distinct() -> None:
    tables = _paired_tables()
    common = dict(split="train", relation_cycle_length=20, seed=11)
    verified = ObservationRelationSampler(tables, pairing_mode="verified", **common)
    unpaired = ObservationRelationSampler(
        tables, pairing_mode="unpaired", verified_pair_retention_fraction=0.0, **common
    )
    mismatched = ObservationRelationSampler(tables, pairing_mode="wrong_subject", **common)

    verified_cross = [item for item in verified.supported_examples if item.is_cross_modal]
    mismatched_cross = [item for item in mismatched.supported_examples if item.is_cross_modal]
    assert verified_cross
    assert all(item.subject_id == item.target_subject_id for item in verified_cross)
    assert all(item.pairing_type == "verified_pair" for item in verified_cross)
    assert all(item.link_uid is not None and not item.is_pairing_control for item in verified_cross)

    assert set(unpaired.active_relations) == {"mri->mri", "pet->pet"}
    assert not any(item.is_cross_modal for item in unpaired.supported_examples)
    assert unpaired.sampling_summary()["retained_verified_link_count"] == 0
    assert unpaired.sampling_summary()["supports_within_modality_singletons"] is True

    assert len(mismatched_cross) == len(verified_cross)
    assert all(item.subject_id != item.target_subject_id for item in mismatched_cross)
    assert all(item.pairing_type == "mismatched_subject_pair" for item in mismatched_cross)
    assert all(item.link_uid is None and item.is_pairing_control for item in mismatched_cross)
    assert all(item.source_link_uid != item.target_link_uid for item in mismatched_cross)
    assert mismatched.sampling_summary()["mismatched_pair_generation"] == (
        "deterministic_cyclic_subject_derangement"
    )


def test_validation_selection_uses_the_configured_pairing_policy() -> None:
    tables = _paired_tables(split="validation")
    config = {
        "training": {
            "relation_cycle_length": 20,
            "relation_sampling_weights": {
                "mri_to_mri": 0.35,
                "pet_to_pet": 0.35,
                "mri_to_pet": 0.15,
                "pet_to_mri": 0.15,
            },
        },
        "data": {},
    }
    unpaired_experiment = {"verified_pair_retention_fraction": 0.0, "pairing_mode": "unpaired"}
    selection = _make_sampler(
        tables,
        config,
        unpaired_experiment,
        split="validation",
        seed=31,
        use_configured_pairing=True,
    )
    verified_reference = _make_sampler(
        tables,
        config,
        unpaired_experiment,
        split="validation",
        seed=31,
        use_configured_pairing=False,
    )

    assert set(selection.active_relations) == {"mri->mri", "pet->pet"}
    assert set(verified_reference.active_relations) == {
        "mri->mri",
        "pet->pet",
        "mri->pet",
        "pet->mri",
    }


class _ObservationCache:
    def __init__(self, values: dict[str, ObservationVolume]) -> None:
        self.values = values

    def get(self, uid: str) -> ObservationVolume:
        return self.values[uid]


def _observation_volume(
    uid: str,
    subject: str,
    modality: str,
    values: torch.Tensor,
    *,
    metadata: dict | None = None,
) -> ObservationVolume:
    observation = Observation(
        observation_uid=uid,
        subject_id=subject,
        session_id="ses-1",
        split="train",
        modality=modality,
        volume_path=Path(f"{uid}.nii.gz"),
        brain_mask_path=Path(f"{uid}-mask.nii.gz"),
        metadata=metadata or {},
    )
    support = torch.ones_like(values)
    return ObservationVolume(
        observation=observation,
        volume=values,
        brain_mask=support,
        visible_mask=support.clone(),
        reliability=support.clone(),
        normalization_mean=0.0,
        normalization_std=1.0,
        voxel_volume_mm3=1.0,
    )


def test_observation_metadata_does_not_confuse_audit_metadata_with_acquisition_values() -> None:
    values = torch.ones(1, 2, 2, 2)
    audit_only = _observation_volume(
        "S00-mri",
        "S00",
        "mri",
        values,
        metadata={"legacy_pair_uid": "pair-0", "registration_qc": "pass"},
    )
    explicit = _observation_volume(
        "S01-pet",
        "S01",
        "pet",
        values,
        metadata={
            "acquisition_metadata": {"tracer": "FDG"},
            "acquisition_metadata_available": True,
        },
    )

    assert _observation_metadata(audit_only, 4)[3].item() == 0.0
    assert _observation_metadata(explicit, 4)[3].item() == 1.0


def test_acquisition_availability_cannot_be_claimed_without_values() -> None:
    observation = Observation(
        observation_uid="S00-mri",
        subject_id="S00",
        session_id="ses-1",
        split="train",
        modality="mri",
        volume_path=Path("S00-mri.nii.gz"),
        brain_mask_path=Path("S00-mask.nii.gz"),
        metadata={
            "acquisition_metadata": {},
            "acquisition_metadata_available": True,
        },
    )
    with pytest.raises(ObservationValidationError, match="provides no values"):
        build_observation_tables([observation], [], {"S00": "train"})


def test_masked_target_observation_never_exposes_query_tokens() -> None:
    mri = _observation_volume(
        "S00-mri", "S00", "mri", torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    )
    pet = _observation_volume(
        "S00-pet", "S00", "pet", torch.arange(8, 0, -1, dtype=torch.float32).reshape(1, 2, 2, 2)
    )
    cache = _ObservationCache({"S00-mri": mri, "S00-pet": pet})
    example = RelationExample(
        source_observation_uid="S00-mri",
        target_observation_uid="S00-pet",
        source_modality="mri",
        target_modality="pet",
        relation="mri->pet",
        subject_id="S00",
        target_subject_id="S00",
        split="train",
        link_uid="S00-link",
        link_reliability=1.0,
        pairing_type="verified_pair",
        is_pairing_control=False,
    )

    prepared = prepare_example(
        cache,  # type: ignore[arg-type]
        example,
        grid=(2, 2, 2),
        hidden_fraction=0.3,
        query_tokens=2,
        include_target_observation_probability=1.0,
        metadata_dim=4,
        seed=19,
    )

    assert prepared.target_observation_included is True
    assert len(prepared.observations) == 2
    assert prepared.observations[1].observation_uid == "S00-pet"
    query = prepared.query_mask[0]
    assert int(query.sum()) == 2
    assert torch.equal(
        prepared.observations[1].token_coverage[0, query],
        torch.zeros(int(query.sum())),
    )
    assert torch.equal(prepared.target_volume, pet.volume.unsqueeze(0))


def test_intra_modal_query_tokens_are_hidden_before_online_encoding() -> None:
    mri = _observation_volume(
        "S00-mri", "S00", "mri", torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    )
    cache = _ObservationCache({"S00-mri": mri})
    example = RelationExample(
        source_observation_uid="S00-mri",
        target_observation_uid="S00-mri",
        source_modality="mri",
        target_modality="mri",
        relation="mri->mri",
        subject_id="S00",
        target_subject_id="S00",
        split="train",
        link_uid=None,
        link_reliability=1.0,
        pairing_type="within_observation",
        is_pairing_control=False,
    )

    prepared = prepare_example(
        cache,  # type: ignore[arg-type]
        example,
        grid=(2, 2, 2),
        hidden_fraction=0.5,
        query_tokens=2,
        include_target_observation_probability=0.0,
        metadata_dim=4,
        seed=23,
    )

    assert len(prepared.observations) == 1
    query = prepared.query_mask[0]
    assert int(query.sum()) == 2
    assert torch.equal(
        prepared.observations[0].token_coverage[0, query],
        torch.zeros(int(query.sum())),
    )
