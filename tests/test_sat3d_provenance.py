from __future__ import annotations

from pathlib import Path

from mri_pet_geomc.model.trainable_sat3d import sat3d_source_tree_sha256


def test_sat3d_source_tree_digest_is_stable_and_source_sensitive(tmp_path: Path) -> None:
    python_root = tmp_path / "SAT3D-slicer" / "sat3D"
    package = python_root / "segment_anything_with_swin_conf"
    package.mkdir(parents=True)
    builder = package / "build_samswin3D.py"
    builder.write_text("REGISTRY = {}\n", encoding="utf-8")
    helper = package / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")

    resolved, first = sat3d_source_tree_sha256(tmp_path)
    _, repeated = sat3d_source_tree_sha256(tmp_path)
    helper.write_text("VALUE = 2\n", encoding="utf-8")
    _, changed = sat3d_source_tree_sha256(tmp_path)

    assert resolved == python_root.resolve()
    assert first == repeated
    assert changed != first
