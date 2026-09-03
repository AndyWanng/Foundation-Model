from __future__ import annotations

import io
import json
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
import yaml

from mri_pet_geomc.formal.data import (
    FormalSample,
    ObservationRelation,
    RelationSchedule,
    SampleKey,
)
from mri_pet_geomc.formal.data.catalog import FOMO_DATASETS, PET_MANIFESTS
from mri_pet_geomc.formal.config import (
    FormalConfigError,
    load_formal_config,
    validate_formal_config,
)
from mri_pet_geomc.formal.integration import (
    _coverage_aligned_warmup_updates,
    DeterministicPhysicalBlockMasker,
    FormalBatchPreparer,
    FormalCoverageDataSource,
    FormalModelBuildSpec,
    acquisition_condition_record,
    build_acquisition_vocabulary,
    encode_condition_batch,
    run_final_test_once,
    run_full_training,
    run_synthetic_smoke,
)
from mri_pet_geomc.formal.model import (
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    OnlineEMABundle,
)
from mri_pet_geomc.formal.runtime import (
    AtomicCheckpointManager,
    FormalTrainingSchedule,
    Microbatch,
    ModelContractError,
    PickleSerializer,
    ResourceSnapshot,
    RunJournal,
    TelemetryRecorder,
    TerminalProgress,
    TrainingRuntimeController,
    TrainingStepResult,
)


def _metadata(modality: str) -> dict[str, Any]:
    return {
        "dataset_id": "must-not-enter-model",
        "canonical_subject_id": "must-not-enter-model",
        "site": "must-not-enter-model",
        "qc": "must-not-enter-model",
        "modality": modality,
        "sequence": "MP2RAGE" if modality == "mri" else "",
        "product": "UNI" if modality == "mri" else "",
        "tracer_family": "amyloid" if modality == "pet" else "",
        "tracer": "AV45" if modality == "pet" else "",
        "derived": False,
        "preprocessing": {"geometry": {"native_spacing": [1.0, 1.5, 2.0]}},
    }


def test_warmup_updates_follow_the_configured_50_coverage_horizon() -> None:
    assert _coverage_aligned_warmup_updates(
        total_optimizer_steps=59_900,
        total_coverages=50,
        warmup_coverages=2,
    ) == 2_396

    with pytest.raises(ValueError, match="coverage-aligned"):
        _coverage_aligned_warmup_updates(
            total_optimizer_steps=59_901,
            total_coverages=50,
            warmup_coverages=2,
        )


def test_acquisition_projection_and_deterministic_metadata_dropout_are_safe() -> None:
    records = [_metadata("mri"), _metadata("pet")]
    projected = acquisition_condition_record(records[0])
    assert set(projected) == set(CATEGORICAL_FIELDS) | set(CONTINUOUS_FIELDS)
    assert not ({"dataset_id", "canonical_subject_id", "site", "qc"} & projected.keys())
    assert projected["mp2rage_component"] == "UNI"
    vocabulary = build_acquisition_vocabulary(records)
    common = dict(
        device="cpu",
        observation_uids=["mri", "pet"],
        coverage_ids=[7, 7],
        view_ids=[0, 0],
        dropout_probability=0.90,
        dropout_seed=99,
    )
    left = encode_condition_batch(vocabulary, records, **common)
    right = encode_condition_batch(vocabulary, records, **common)
    assert torch.equal(left.categorical, right.categorical)
    assert torch.equal(left.continuous_present, right.continuous_present)
    modality_column = CATEGORICAL_FIELDS.index("modality")
    assert torch.all(left.categorical[:, modality_column] >= 2)
    assert not left.continuous_present.all()


def test_physical_mask_is_deterministic_supported_and_padded() -> None:
    xyz = torch.tensor(
        [[x, y, z] for x in (0.0, 2.0) for y in (0.0, 2.0) for z in (0.0, 2.0)]
    )
    masker = DeterministicPhysicalBlockMasker(
        xyz,
        grid_shape=(2, 2, 2),
        volume_shape=(8, 8, 8),
        hidden_fraction=0.50,
        max_queries=4,
        blocks=2,
        seed=3,
    )
    support = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
    support[:, :4, :, :] = True
    first = masker.plan(
        support, anchor_uids=["a"], coverage_ids=[1], view_ids=[0]
    )
    second = masker.plan(
        support, anchor_uids=["a"], coverage_ids=[1], view_ids=[0]
    )
    assert torch.equal(first.query_indices, second.query_indices)
    assert first.query_valid.sum().item() == 2
    assert torch.equal(first.query_indices[~first.query_valid], torch.tensor([-1, -1]))
    selected = first.query_indices[first.query_valid]
    assert torch.all(first.full_token_support[0, selected] > 0)
    assert first.hidden_voxel_mask.shape == support.shape


def _companion_fallback_fixture() -> tuple[
    FormalBatchPreparer,
    DeterministicPhysicalBlockMasker,
    list[tuple[Mapping[str, Any], ...]],
]:
    records = [_metadata("mri"), _metadata("mri")]
    vocabulary = build_acquisition_vocabulary(records)
    xyz = torch.tensor(
        [[x, y, z] for x in (0.0, 2.0) for y in (0.0, 2.0) for z in (0.0, 2.0)]
    )
    masker = DeterministicPhysicalBlockMasker(
        xyz,
        grid_shape=(2, 2, 2),
        volume_shape=(8, 8, 8),
        hidden_fraction=0.50,
        max_queries=4,
        blocks=2,
        seed=3,
    )
    events: list[tuple[Mapping[str, Any], ...]] = []
    preparer = FormalBatchPreparer(
        vocabulary=vocabulary,
        masker=masker,
        device="cpu",
        non_blocking=False,
        metadata_dropout_probability=0.0,
        companion_fallback_callback=lambda values: events.append(tuple(values)),
    )
    return preparer, masker, events


def test_empty_visible_optional_companion_falls_back_to_self_without_dropping_anchor() -> None:
    preparer, masker, events = _companion_fallback_fixture()
    anchor_support = torch.ones(1, 8, 8, 8, dtype=torch.bool)
    plan = masker.plan(
        anchor_support,
        anchor_uids=["anchor"],
        coverage_ids=[7],
        view_ids=[0],
    )
    companion_support = plan.hidden_voxel_mask.clone()
    batch = {
        "anchor_image": torch.randn(1, 1, 8, 8, 8),
        "anchor_support": anchor_support,
        "anchor_uid": ["anchor"],
        "anchor_metadata": [_metadata("mri")],
        "coverage_id": torch.tensor([7]),
        "view_id": torch.tensor([0]),
        "relation_type": ["same_session_cross_sequence"],
        "relation_uid": ["relation"],
        "temporal_delta_days": [None],
        "companion_image": torch.randn(1, 1, 8, 8, 8),
        "companion_support": companion_support,
        "companion_batch_index": torch.tensor([0]),
        "companion_uid": ["companion"],
        "companion_metadata": [_metadata("mri")],
    }

    prepared = preparer.prepare(batch, training=True)

    assert prepared.anchor_count == 1
    assert prepared.companion is None
    assert prepared.relation_types == ("same_observation",)
    assert prepared.relation_codes.tolist() == [0]
    assert prepared.companion_requested_count == 1
    assert len(prepared.companion_fallbacks) == 1
    assert prepared.companion_fallbacks[0]["anchor_uid"] == "anchor"
    assert prepared.companion_fallbacks[0]["companion_uid"] == "companion"
    assert prepared.companion_fallbacks[0]["companion_support_voxels"] > 0
    assert prepared.companion_fallbacks[0]["companion_visible_voxels"] == 0
    assert events == [prepared.companion_fallbacks]


def test_companion_fallback_filters_only_the_unusable_packed_row() -> None:
    preparer, masker, events = _companion_fallback_fixture()
    anchor_support = torch.ones(2, 8, 8, 8, dtype=torch.bool)
    plan = masker.plan(
        anchor_support,
        anchor_uids=["anchor-0", "anchor-1"],
        coverage_ids=[7, 7],
        view_ids=[0, 0],
    )
    companion_support = torch.stack(
        (plan.hidden_voxel_mask[0], torch.ones(8, 8, 8, dtype=torch.bool))
    )
    batch = {
        "anchor_image": torch.randn(2, 1, 8, 8, 8),
        "anchor_support": anchor_support,
        "anchor_uid": ["anchor-0", "anchor-1"],
        "anchor_metadata": [_metadata("mri"), _metadata("mri")],
        "coverage_id": torch.tensor([7, 7]),
        "view_id": torch.tensor([0, 0]),
        "relation_type": [
            "same_session_cross_sequence",
            "same_session_cross_sequence",
        ],
        "relation_uid": ["relation-0", "relation-1"],
        "temporal_delta_days": [None, None],
        "companion_image": torch.randn(2, 1, 8, 8, 8),
        "companion_support": companion_support,
        "companion_batch_index": torch.tensor([0, 1]),
        "companion_uid": ["companion-0", "companion-1"],
        "companion_metadata": [_metadata("mri"), _metadata("mri")],
    }

    prepared = preparer.prepare(batch, training=False)

    assert prepared.companion is not None
    assert prepared.companion.anchor_indices.tolist() == [1]
    assert prepared.companion.observation.observation_uid == ("companion-1",)
    assert prepared.relation_types == (
        "same_observation",
        "same_session_cross_sequence",
    )
    assert prepared.relation_codes.tolist() == [0, 1]
    assert prepared.companion_requested_count == 2
    assert len(prepared.companion_fallbacks) == 1
    assert len(events) == 1 and len(events[0]) == 1


def test_empty_cached_companion_support_remains_a_structural_error() -> None:
    preparer, _, _ = _companion_fallback_fixture()
    batch = {
        "anchor_image": torch.randn(1, 1, 8, 8, 8),
        "anchor_support": torch.ones(1, 8, 8, 8, dtype=torch.bool),
        "anchor_uid": ["anchor"],
        "anchor_metadata": [_metadata("mri")],
        "coverage_id": torch.tensor([7]),
        "view_id": torch.tensor([0]),
        "relation_type": ["same_session_cross_sequence"],
        "relation_uid": ["relation"],
        "temporal_delta_days": [None],
        "companion_image": torch.randn(1, 1, 8, 8, 8),
        "companion_support": torch.zeros(1, 8, 8, 8, dtype=torch.bool),
        "companion_batch_index": torch.tensor([0]),
        "companion_uid": ["empty-companion"],
        "companion_metadata": [_metadata("mri")],
    }

    with pytest.raises(
        ModelContractError,
        match="Cached companion empty-companion has empty structural support",
    ):
        preparer.prepare(batch)


class _Dataset(torch.utils.data.Dataset[FormalSample]):
    def __init__(
        self, count: int = 5, modalities: tuple[str, ...] | None = None
    ) -> None:
        self.observation_uids = tuple(f"uid-{index}" for index in range(count))
        self.modalities = modalities or tuple("mri" for _ in range(count))
        assert len(self.modalities) == count
        self.observations = tuple(
            SimpleNamespace(modality=value) for value in self.modalities
        )
        self.selector = SimpleNamespace(schedule=RelationSchedule())

    def __len__(self) -> int:
        return len(self.observation_uids)

    def __getitem__(self, index: int | SampleKey) -> FormalSample:
        key = index if isinstance(index, SampleKey) else SampleKey(index, 1, 0)
        uid = self.observation_uids[key.anchor_index]
        relation = ObservationRelation(
            relation_uid=f"self-{uid}",
            relation_type="same_observation",
            anchor_uid=uid,
            companion_uid=uid,
            canonical_subject_id=uid,
            split="train",
        )
        return FormalSample(
            anchor_uid=uid,
            anchor_image=torch.zeros(1, 8, 8, 8),
            anchor_support=torch.ones(8, 8, 8, dtype=torch.bool),
            anchor_metadata=_metadata(self.modalities[key.anchor_index]),
            coverage_id=key.coverage_id,
            view_id=key.view_id,
            relation=relation,
        )


def test_data_source_uses_one_based_coverage_and_exact_consumed_cursor() -> None:
    source = FormalCoverageDataSource(
        _Dataset(), seed=13, num_workers=0, pin_memory=False
    )
    expected = FormalTrainingSchedule().relation_mixture(3)
    assert source.assert_relation_mixture(3, expected)["same_observation"] == 0.875
    batches = list(
        source.iter_microbatches(
            coverage=3,
            start_anchor_index=2,
            micro_batch_anchors=2,
            relation_mixture=expected,
        )
    )
    assert [item.anchor_count for item in batches] == [2, 1]
    assert [item.next_anchor_index for item in batches] == [4, 5]
    assert sum(item.anchor_count for item in batches) == 3
    assert all(torch.as_tensor(item.payload["coverage_id"]).eq(3).all() for item in batches)


def test_observational_subset_is_deterministic_and_modality_balanced() -> None:
    dataset = _Dataset(8, ("mri",) * 4 + ("pet",) * 4)
    source = FormalCoverageDataSource(
        dataset, seed=13, num_workers=0, pin_memory=False
    )
    first = source.evaluation_anchor_indices(coverage=7, max_anchors=4)
    second = source.evaluation_anchor_indices(coverage=7, max_anchors=4)

    assert first == second
    assert len(first) == 4
    assert [dataset.modalities[index] for index in first].count("mri") == 2
    assert [dataset.modalities[index] for index in first].count("pet") == 2


def test_formal_spec_is_constructed_from_yaml_values() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "workstation_formal.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    spec = FormalModelBuildSpec.from_config(
        config,
        sat3d_code_root="sat3d",
        sat3d_checkpoint="sat3d.pt",
        fem_geometry_path="fem.pt",
        reference_affine=torch.eye(4).tolist(),
    )
    assert spec.interpolation_sigma_mm == 12.0
    assert spec.geometry_feature_type == "spectral"
    assert spec.stem_hidden_channels == 8
    assert spec.adapter_bottleneck_ratio == 0.25
    assert spec.geomc_scale_embedding_dim == 16
    assert spec.geomc_residual_scale == 0.10
    assert spec.geometry_embedding_initial_scale == 0.05
    assert spec.mask_hidden_fraction == 0.30
    assert spec.mask_blocks == 2
    assert spec.model_lr == 2.0e-4


def test_formal_config_has_no_quality_or_reliability_switches() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "workstation_formal.yaml"
    config = load_formal_config(config_path)
    assert "quality_control" not in config["preprocessing"]
    assert "reliability_weighting" not in config["model"]["fusion"]
    invalid = copy.deepcopy(config)
    invalid["preprocessing"]["quality_control"] = False
    with pytest.raises(FormalConfigError, match="was removed"):
        validate_formal_config(invalid)

    split = config["data"]["split"]
    assert split["group_key"] == "canonical_subject_id"
    assert split["balance_by_dataset"] is True
    assert "soft_balance_fields" not in split
    invalid = copy.deepcopy(config)
    invalid["data"]["split"]["soft_balance_fields"] = ["sequence"]
    with pytest.raises(FormalConfigError, match="unsupported"):
        validate_formal_config(invalid)

    assert tuple(config["data"]["fomo_sources"]) == FOMO_DATASETS
    assert tuple(config["data"]["pet_manifests"]) == tuple(PET_MANIFESTS.values())
    assert tuple(config["model"]["ema"]["includes"]) == OnlineEMABundle.EMA_SCOPE
    assert tuple(config["model"]["ema"]["excludes"]) == OnlineEMABundle.EMA_EXCLUDED


def test_tiny_smoke_is_finite_and_explicitly_non_scientific(tmp_path: Path) -> None:
    result = run_synthetic_smoke(work_dir=tmp_path, device="cpu")
    assert result["valid_for_scientific_results"] is False
    assert result["anchors_processed"] == 2
    assert result["optimizer_steps"] == 1
    assert result["loss_total"] > 0
    assert Path(result["artifact"]).is_file()


class _Stateful:
    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    def state_dict(self) -> Mapping[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state_dict: Mapping[str, Any], **_: Any) -> None:
        self.value = int(state_dict["value"])


class _EMA:
    def __init__(self) -> None:
        self.updates = 0

    def state_dict(self) -> Mapping[str, Any]:
        return {"updates": self.updates}

    def load_state_dict(self, state_dict: Mapping[str, Any], **_: Any) -> None:
        self.updates = int(state_dict["updates"])


@dataclass
class _Backend:
    fail_coverage: int | None = None

    def __post_init__(self) -> None:
        self.model = _Stateful()
        self.optimizer = _Stateful()
        self.ema = _EMA()
        self.scheduler = None
        self.scaler = None

    def extra_checkpoint_stateful(self) -> Mapping[str, Any]:
        return {}

    def train_accumulated_step(
        self,
        microbatches: Any,
        *,
        coverage: int,
        relation_mixture: Any,
        expected_global_batch_anchors: int,
    ) -> TrainingStepResult:
        del relation_mixture
        if coverage == self.fail_coverage:
            raise RuntimeError("simulated interruption")
        anchors = sum(item.anchor_count for item in microbatches)
        self.model.value += anchors
        self.optimizer.value += 1
        self.ema.updates += 1
        return TrainingStepResult(
            anchors_processed=anchors,
            next_anchor_index=microbatches[-1].next_anchor_index,
            elapsed_seconds=0.01,
            loader_wait_seconds=0.0,
            loss_total=1.0 / coverage,
            loss_components={"self": 1.0 / coverage},
            modality_losses={"mri": 1.0 / coverage},
            modality_counts={"mri": anchors},
            relation_losses={"same_observation": 1.0 / coverage},
            relation_counts={"same_observation": anchors},
            learning_rates={"main": 1.0e-4},
            ema_decay=0.996,
            grad_norm=1.0,
            is_coverage_tail=anchors < expected_global_batch_anchors,
        )

    def validate(self, request: Any) -> Mapping[str, Any]:
        return {"loss": 2.0 / request.coverage}


class _Source:
    def total_anchors(self, coverage: int) -> int:
        assert 1 <= coverage <= 100
        return 1

    def assert_relation_mixture(self, coverage: int, mixture: Any) -> Mapping[str, float]:
        del coverage
        return FormalCoverageDataSource._runtime_weights(mixture)

    def iter_microbatches(
        self,
        *,
        coverage: int,
        start_anchor_index: int,
        micro_batch_anchors: int,
        relation_mixture: Any,
    ) -> Any:
        del coverage, micro_batch_anchors, relation_mixture
        if start_anchor_index == 0:
            yield Microbatch(
                payload=None,
                anchor_count=1,
                next_anchor_index=1,
                loader_wait_seconds=0.0,
            )


class _Resources:
    def sample(self) -> ResourceSnapshot:
        return ResourceSnapshot(cpu_rss_gib=0.1)


def _controller(root: Path, backend: _Backend) -> tuple[TrainingRuntimeController, RunJournal]:
    del backend
    journal = RunJournal(root / "run.log", root / "metrics.jsonl", run_id="run")
    progress = TerminalProgress(
        journal=journal,
        stream=io.StringIO(),
        force_plain=True,
        plain_update_every=1000,
    )
    telemetry = TelemetryRecorder(
        journal=journal, progress=progress, resource_provider=_Resources()
    )
    return (
        TrainingRuntimeController(
            run_id="run",
            hashes={"manifest": "m", "config": "c", "split": "s"},
            journal=journal,
            progress=progress,
            telemetry=telemetry,
            checkpoints=AtomicCheckpointManager(
                root / "checkpoints",
                serializer=PickleSerializer(),
                journal=journal,
            ),
            checkpoint_every_optimizer_steps=7,
        ),
        journal,
    )


def test_exact_resume_retention_completion_and_final_test_idempotence(tmp_path: Path) -> None:
    unrelated = tmp_path / "checkpoints" / "do-not-delete.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("owned elsewhere", encoding="utf-8")
    interrupted = _Backend(fail_coverage=4)
    controller, journal = _controller(tmp_path, interrupted)
    with journal, pytest.raises(RuntimeError, match="simulated interruption"):
        run_full_training(
            controller=controller,
            data_source=_Source(),  # type: ignore[arg-type]
            backend=interrupted,  # type: ignore[arg-type]
            resume=True,
        )

    resumed = _Backend()
    controller, journal = _controller(tmp_path, resumed)
    with journal:
        result = run_full_training(
            controller=controller,
            data_source=_Source(),  # type: ignore[arg-type]
            backend=resumed,  # type: ignore[arg-type]
            resume=True,
        )
        assert result.completed and result.resumed
        assert result.trained_anchors == 97
        assert result.cursor.coverage == 100
        assert result.cursor.next_anchor_index == 1
        calls = {"count": 0}

        def held_out_test(_: Any) -> Mapping[str, Any]:
            calls["count"] += 1
            return {"held_out_loss": 0.25, "anchors": 3}

        first = run_final_test_once(
            controller=controller,
            backend=resumed,  # type: ignore[arg-type]
            callback=held_out_test,
        )
        second = run_final_test_once(
            controller=controller,
            backend=resumed,  # type: ignore[arg-type]
            callback=held_out_test,
        )
        assert first == second
        assert calls["count"] == 1

    checkpoint_files = list((tmp_path / "checkpoints").glob("formal-training-c*-s*.pkl"))
    milestone_coverages = {
        int(path.name.split("-c", 1)[1].split("-", 1)[0])
        for path in checkpoint_files
        if int(path.name.split("-c", 1)[1].split("-", 1)[0]) % 5 == 0
    }
    assert milestone_coverages == set(range(5, 101, 5))
    assert len(checkpoint_files) <= 23
    assert unrelated.read_text(encoding="utf-8") == "owned elsewhere"
    completion = json.loads(
        (tmp_path / "checkpoints" / "training_complete.json").read_text(encoding="utf-8")
    )
    assert completion["payload"]["held_out_test_complete"] is True

    final_backend = _Backend()
    controller, journal = _controller(tmp_path, final_backend)
    with journal:
        no_op = run_full_training(
            controller=controller,
            data_source=_Source(),  # type: ignore[arg-type]
            backend=final_backend,  # type: ignore[arg-type]
            resume=True,
        )
    assert no_op.already_complete
    assert no_op.trained_optimizer_steps == 0
    assert final_backend.model.value == 100
