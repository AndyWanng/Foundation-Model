from __future__ import annotations

import json
from pathlib import Path

import pytest

from mri_pet_geomc.data.observations import Observation
from mri_pet_geomc.stages import _verify_pet128_preprocessing_record
from mri_pet_geomc.utils import sha256_file


NORMALIZATION = "eroded_brain_relative_median_after_p005_p995_clip"


def _pet_with_preprocessing_record(
    tmp_path: Path,
    *,
    spatial_resample_count: int = 1,
    output_sha256: str | None = None,
) -> Observation:
    volume = tmp_path / "pet_sat3d_input_128.nii.gz"
    volume.write_bytes(b"model-input-payload")
    digest = sha256_file(volume)
    preprocessing_record = tmp_path / "pet128_preprocessing_record.json"
    payload = {
        "schema_version": 1,
        "status": "completed",
        "spatial_resample_count": spatial_resample_count,
        "suv_claim_allowed": False,
        "plan": {"spatial_resample_count": spatial_resample_count},
        "execution": {"spatial_resample_count": spatial_resample_count},
        "relative_normalization": {
            "normalization": NORMALIZATION,
            "quantification_contract": "relative_fdg_only",
            "spatial_resample_count_added": 0,
            "suv_claim_allowed": False,
        },
        "sat3d_normalization": {
            "output_sha256": digest,
            "spatial_resample_count_added": 0,
            "suv_claim_allowed": False,
        },
        "outputs": [
            {
                "path": str(volume),
                "sha256": output_sha256 if output_sha256 is not None else digest,
            }
        ],
    }
    preprocessing_record.write_text(json.dumps(payload), encoding="utf-8")
    mask = tmp_path / "mask.nii.gz"
    mask.write_bytes(b"mask")
    return Observation(
        observation_uid="pet-1",
        subject_id="S1",
        session_id="ses-1",
        split="train",
        modality="pet",
        volume_path=volume,
        brain_mask_path=mask,
        preprocessing_record_path=preprocessing_record,
    )


def test_pet128_preprocessing_record_proves_one_resample_and_relative_fdg(tmp_path: Path) -> None:
    observation = _pet_with_preprocessing_record(tmp_path)
    result = _verify_pet128_preprocessing_record(
        observation,
        expected_relative_normalization=NORMALIZATION,
    )
    assert result["spatial_resample_count"] == 1
    assert result["quantification_contract"] == "relative_fdg_only"
    assert result["suv_claim_allowed"] is False
    assert result["model_input_sha256"] == sha256_file(observation.volume_path)


def test_pet128_preprocessing_record_rejects_more_than_one_resample(tmp_path: Path) -> None:
    observation = _pet_with_preprocessing_record(tmp_path, spatial_resample_count=2)
    with pytest.raises(RuntimeError, match="exactly one spatial resample"):
        _verify_pet128_preprocessing_record(
            observation,
            expected_relative_normalization=NORMALIZATION,
        )


def test_pet128_preprocessing_record_rejects_unbound_model_input(tmp_path: Path) -> None:
    observation = _pet_with_preprocessing_record(tmp_path, output_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="not hash-bound"):
        _verify_pet128_preprocessing_record(
            observation,
            expected_relative_normalization=NORMALIZATION,
        )
