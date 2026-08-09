from __future__ import annotations

from pathlib import Path

from mri_pet_geomc.utils import source_tree_digest


def test_source_tree_digest_ignores_generated_packaging_outputs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    baseline = source_tree_digest(project)

    (project / "build" / "lib").mkdir(parents=True)
    (project / "build" / "lib" / "copied.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    (project / "dist").mkdir()
    (project / "dist" / "metadata.txt").write_text("generated\n", encoding="utf-8")
    egg_info = project / "src" / "example.egg-info"
    egg_info.mkdir(parents=True)
    (egg_info / "SOURCES.txt").write_text("generated\n", encoding="utf-8")
    dist_info = project / "vendor" / "example.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA.txt").write_text("generated\n", encoding="utf-8")
    test_deps = project / ".codex_test_deps"
    test_deps.mkdir()
    (test_deps / "third_party.py").write_text("VALUE = 99\n", encoding="utf-8")

    assert source_tree_digest(project) == baseline


def test_source_tree_digest_still_tracks_deliverable_source(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    baseline = source_tree_digest(project)

    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert source_tree_digest(project) != baseline
