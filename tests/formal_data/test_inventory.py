from __future__ import annotations

from pathlib import Path

from mri_pet_geomc.formal.data.inventory import PhysicalInventory


def test_hierarchy_ignores_cache_pt030_and_normalizes_duplicate_session_wrapper(
    tmp_path: Path,
) -> None:
    hierarchy = tmp_path / "tree.txt"
    hierarchy.write_text(
        "\n".join(
            [
                "FOMO_MRI/.cache/huggingface/PT001/sub-01/ses-01.zip",
                "FOMO_MRI/PT001_ClevelandCCF/sub-01/ses-01/ses-01/anat/a.nii.gz",
                "FOMO_MRI/PT001_ClevelandCCF/sub-02/ses-01.zip",
                "FOMO_MRI/PT030_HCP_1200/sub-99/ses-01/anat/x.nii.gz",
                "ADNI_PET/FDG/002_S_0001/FDG_series/2020/I123",
                "ADNI_PET/FDG/002_S_0001/FDG_series/2020/I123/a.dcm",
                "ADNI_PET/FDG/002_S_0001/FDG_series/2020/I123/b.dcm",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    inventory = PhysicalInventory.from_hierarchy(
        hierarchy,
        fomo_root=tmp_path / "FOMO_MRI",
        adni_root=tmp_path / "ADNI_PET",
    )
    extracted = inventory.fomo_file(
        "PT001_ClevelandCCF", "sub-01/ses-01/anat/a.nii.gz"
    )
    assert extracted is not None
    assert "ses-01/ses-01" in extracted.relative_path
    assert inventory.fomo("PT001_ClevelandCCF", "sub-02", "ses-01") is not None
    assert not any(key[0].startswith("PT030") for key in inventory.fomo_files)
    assert inventory.adni("123") is not None
    assert inventory.adni("123").file_count == 2  # type: ignore[union-attr]
