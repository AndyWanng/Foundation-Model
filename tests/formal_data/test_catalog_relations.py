from __future__ import annotations

from collections import Counter
from pathlib import Path

from mri_pet_geomc.formal.data.catalog import (
    FOMO_DATASETS,
    PET_MANIFESTS,
    _adni_manifest_base,
    _fomo_metadata_base,
    _sequence_product,
    assign_subject_splits,
    build_catalog,
)
from mri_pet_geomc.formal.data.inventory import InventoryEntry, PhysicalInventory
from mri_pet_geomc.formal.data.relations import build_relations
from mri_pet_geomc.formal.data.schema import CatalogObservation, SourceLocator, stable_uid


def _inventory_entry(relative: str, *, count: int = 1, **values: str) -> InventoryEntry:
    return InventoryEntry(
        relative_path=relative,
        path=f"/workstation/{relative}",
        file_count=count,
        signature=f"sig:{relative}",
        **values,
    )


def test_exact_namespaced_80_10_10_split() -> None:
    subjects = {
        **{f"FOMO:PT001:S{index:03d}": "PT001" for index in range(60)},
        **{f"ADNI:A{index:03d}": "ADNI_PET" for index in range(40)},
    }
    first = assign_subject_splits(subjects, seed=17)
    second = assign_subject_splits(dict(reversed(list(subjects.items()))), seed=17)
    assert first == second
    assert Counter(row.split for row in first) == {
        "train": 80,
        "validation": 10,
        "test": 10,
    }
    assert len({row.canonical_subject_id for row in first}) == 100


def test_original_metadata_layout_is_also_detected(tmp_path: Path) -> None:
    fomo = tmp_path / "MRI"
    for dataset_id in FOMO_DATASETS:
        root = fomo / dataset_id
        root.mkdir(parents=True)
        (root / "mapping.tsv").write_text("header\n", encoding="utf-8")
    manifests = tmp_path / "PET" / "ADNI" / "manifest"
    manifests.mkdir(parents=True)
    for filename in PET_MANIFESTS.values():
        (manifests / filename).write_text("header\n", encoding="utf-8")
    assert _fomo_metadata_base(tmp_path) == fomo
    assert _adni_manifest_base(tmp_path) == manifests


def test_dwi_bvalues_are_products_of_one_sequence() -> None:
    for b_value in (0, 700, 800, 1000, 1500, 2000):
        sequence, product, derived = _sequence_product(
            f"sub-01_ses-01_run-1_dwi_bval{b_value}.nii.gz",
            "raw/session_1/dwi",
            f"dwi_bval{b_value}.nii.gz",
        )
        assert (sequence, product, derived) == ("DWI", f"BVAL{b_value}", False)


def test_bundled_metadata_layout_and_extracted_nifti_precede_zip(tmp_path: Path) -> None:
    fomo_files = {}
    fomo_zips = {}
    for index, dataset_id in enumerate(FOMO_DATASETS, start=1):
        root = tmp_path / "fomo" / dataset_id
        root.mkdir(parents=True)
        participant = f"sub-{index:02d}"
        new_path = f"{participant}/ses-01/anat/{participant}_ses-01_T1w.nii.gz"
        (root / "mapping.tsv").write_text(
            "old_path\tnew_path\told_filename\tnew_filename\tparticipant_id\tsession_id\tmodality\n"
            f"raw/{participant}/session_1/anat/a.nii.gz\t{new_path}\ta.nii.gz\t"
            f"{participant}_ses-01_T1w.nii.gz\t{participant}\tses-01\tanat\n",
            encoding="utf-8",
        )
        (root / "mri_info.tsv").write_text(
            "participant_id\tsession_id\tfilename\tModality\tMagneticFieldStrength\n"
            f"{participant}\tses-01\t{new_path}\tMR\t3.0\n",
            encoding="utf-8",
        )
        relative = f"FOMO_MRI/{dataset_id}/{participant}/ses-01/ses-01/anat/{Path(new_path).name}"
        if index == 1:
            fomo_files[(dataset_id, new_path)] = _inventory_entry(relative)
            fomo_zips[(dataset_id, participant, "ses-01")] = _inventory_entry(
                f"FOMO_MRI/{dataset_id}/{participant}/ses-01.zip"
            )
        else:
            fomo_zips[(dataset_id, participant, "ses-01")] = _inventory_entry(
                f"FOMO_MRI/{dataset_id}/{participant}/ses-01.zip"
            )
    adni_root = tmp_path / "adni_pet"
    adni_root.mkdir()
    adni_series = {}
    for index, (family, filename) in enumerate(PET_MANIFESTS.items(), start=101):
        (adni_root / filename).write_text(
            '"image_id","subject_id","study_id","series_id","image_visit",'
            '"image_date","image_description"\n'
            f'"{index}","002_S_{index}","1","2","m12","2020-01-0{index-100}",'
            f'"{family} AV45 fully processed"\n',
            encoding="utf-8",
        )
        adni_series[str(index)] = _inventory_entry(
            f"ADNI_PET/{family}/002_S_{index}/AV45/2020/I{index}",
            count=96,
            family=family,
            subject_id=f"002_S_{index}",
            series_label="AV45",
        )
    inventory = PhysicalInventory(
        fomo_zips=fomo_zips,
        fomo_files=fomo_files,
        adni_series=adni_series,
        source="unit-test",
    )
    result = build_catalog(tmp_path, inventory, split_seed=9)
    assert len(result.observations) == 9
    extracted = next(
        row for row in result.observations if row.dataset_id == FOMO_DATASETS[0]
    )
    assert extracted.source.kind == "nifti"
    assert extracted.source.archive_member == ""
    zipped = next(row for row in result.observations if row.dataset_id == FOMO_DATASETS[1])
    assert zipped.source.kind == "zip_member"
    assert all(
        row.canonical_subject_id.startswith("FOMO:")
        for row in result.observations
        if row.modality == "mri"
    )
    assert all(
        row.canonical_subject_id.startswith("ADNI:")
        for row in result.observations
        if row.modality == "pet"
    )


def _observation(
    uid: str,
    subject: str,
    modality: str,
    session: str,
    order: float,
    *,
    sequence: str = "",
    tracer: str = "",
    product: str = "",
    derived: bool = False,
) -> CatalogObservation:
    dataset = "PT001" if modality == "mri" else "ADNI_PET"
    return CatalogObservation(
        observation_uid=uid,
        dataset_id=dataset,
        canonical_subject_id=subject,
        participant_id=subject.rsplit(":", 1)[-1],
        modality=modality,
        session_id=session,
        source_session_id=session,
        session_order=order,
        acquisition_uid=f"acq-{uid}",
        source=SourceLocator("nifti", f"/{uid}.nii.gz", f"{uid}.nii.gz"),
        split="train",
        sequence=sequence,
        product=product or sequence,
        derived=derived,
        tracer=tracer,
        tracer_family="FDG" if tracer else "",
    )


def test_relations_use_same_subject_and_past_context_only() -> None:
    observations = (
        _observation("mri-t1-old", "FOMO:PT001:S1", "mri", "s1", 1, sequence="T1w"),
        _observation("mri-t2-old", "FOMO:PT001:S1", "mri", "s1", 1, sequence="T2w"),
        _observation("mri-t1-new", "FOMO:PT001:S1", "mri", "s2", 2, sequence="T1w"),
        _observation("pet-old", "ADNI:A1", "pet", "d1", 1, tracer="FDG"),
        _observation("pet-new", "ADNI:A1", "pet", "d2", 2, tracer="FDG"),
    )
    result = build_relations(observations)
    assert result.summary["cross_modal_relation_count"] == 0
    assert all(
        next(row for row in observations if row.observation_uid == edge.anchor_uid).modality
        == next(row for row in observations if row.observation_uid == edge.companion_uid).modality
        for edge in result.relations
    )
    longitudinal = [
        edge for edge in result.relations if edge.relation_type.startswith("longitudinal")
    ]
    assert {(edge.anchor_uid, edge.companion_uid) for edge in longitudinal} == {
        ("mri-t1-new", "mri-t1-old"),
        ("pet-new", "pet-old"),
    }
    assert all(edge.direction == "past_to_current" for edge in longitudinal)


def test_complementary_products_share_bucket_but_derived_siblings_do_not() -> None:
    subject = "FOMO:PT001:S2"
    observations = (
        _observation(
            "dwi-b0-old", subject, "mri", "s1", 1, sequence="DWI", product="BVAL0"
        ),
        _observation(
            "dwi-b1000-old",
            subject,
            "mri",
            "s1",
            1,
            sequence="DWI",
            product="BVAL1000",
        ),
        _observation(
            "mp-inv1", subject, "mri", "s1", 1, sequence="MP2RAGE", product="INV1"
        ),
        _observation(
            "mp-uni", subject, "mri", "s1", 1, sequence="MP2RAGE", product="UNI"
        ),
        _observation("asl", subject, "mri", "s1", 1, sequence="ASL", product="ASL"),
        _observation(
            "cbf", subject, "mri", "s1", 1, sequence="CBF", product="CBF", derived=True
        ),
        _observation(
            "dwi-b1000-new",
            subject,
            "mri",
            "s2",
            2,
            sequence="DWI",
            product="BVAL1000",
        ),
    )
    result = build_relations(observations)
    complementary = {
        (edge.anchor_uid, edge.companion_uid)
        for edge in result.relations
        if edge.relation_type == "same_session_cross_sequence"
    }
    assert ("dwi-b0-old", "dwi-b1000-old") in complementary
    assert ("dwi-b1000-old", "dwi-b0-old") in complementary
    assert ("mp-inv1", "mp-uni") in complementary
    assert ("mp-uni", "mp-inv1") in complementary
    assert not any("cbf" in endpoints for endpoints in complementary)
    longitudinal = {
        (edge.anchor_uid, edge.companion_uid)
        for edge in result.relations
        if edge.relation_type == "longitudinal_same_sequence"
    }
    assert ("dwi-b1000-new", "dwi-b1000-old") in longitudinal
    assert ("dwi-b1000-new", "dwi-b0-old") not in longitudinal
    assert result.summary["same_session_cross_sequence_bucket_semantics"].startswith(
        "same_session_complementary_acquisition"
    )
