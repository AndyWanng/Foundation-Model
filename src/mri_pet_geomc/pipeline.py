from __future__ import annotations

import json
import os
import socket
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import yaml
from filelock import FileLock, Timeout as FileLockTimeout

from .config import METHOD_NAME
from .resources import snapshot
from .utils import (
    EventLogger,
    atomic_write_json,
    atomic_write_text,
    digest_object,
    read_json,
    sha256_file,
    source_tree_digest,
    utc_now,
)


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    handler: str
    depends_on: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    outputs: list[Path]
    inputs: list[Path] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageContext:
    project_root: Path
    launch_dir: Path
    stage_dir: Path
    config: dict[str, Any]
    spec: StageSpec
    event_logger: EventLogger


Handler = Callable[[StageContext], StageResult]


def _build_geomc_plan(config: Mapping[str, Any]) -> list[StageSpec]:
    stages = [
        StageSpec("00.preflight", "geomc_preflight"),
        StageSpec("01.observations", "geomc_observations", ("00.preflight",)),
        StageSpec("02.asset_snapshot", "geomc_asset_snapshot", ("01.observations",)),
        StageSpec(
            "03.registration_qc",
            "geomc_registration_qc",
            ("02.asset_snapshot",),
        ),
        StageSpec("04.geometry", "geomc_geometry", ("01.observations",)),
        StageSpec(
            "05.model_check",
            "geomc_model_check",
            ("04.geometry",),
        ),
        StageSpec(
            "06.end_to_end_smoke_test",
            "geomc_smoke_test",
            ("03.registration_qc", "04.geometry", "05.model_check"),
        ),
        StageSpec(
            "07.small_sample_overfit_check",
            "geomc_small_sample_overfit_check",
            ("06.end_to_end_smoke_test",),
        ),
    ]
    experiments = [dict(value) for value in config.get("experiments", [])]
    all_train_ids: list[str] = []
    for experiment in experiments:
        identifier = str(experiment["id"])
        for seed_value in experiment["seeds"]:
            seed = int(seed_value)
            stage_id = f"08.experiment.{identifier}.seed{seed}"
            stages.append(
                StageSpec(
                    stage_id,
                    "geomc_train",
                    ("03.registration_qc", "04.geometry", "05.model_check"),
                    {"experiment": experiment, "seed": seed},
                )
            )
            all_train_ids.append(stage_id)
    stages.extend(
        [
            StageSpec(
                "09.evaluate",
                "geomc_evaluate",
                tuple(["07.small_sample_overfit_check", *all_train_ids]),
            ),
            StageSpec("10.report", "geomc_report", ("09.evaluate",)),
        ]
    )
    return stages


def build_plan(config: Mapping[str, Any]) -> list[StageSpec]:
    method = str(config.get("project", {}).get("method", ""))
    if method != METHOD_NAME:
        raise ValueError(
            f"This package implements only {METHOD_NAME!r}; "
            f"refusing legacy or unknown method {method!r}"
        )
    return _build_geomc_plan(config)


def plan_payload(plan: list[StageSpec]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stages": [
            {
                "stage_id": spec.stage_id,
                "handler": spec.handler,
                "depends_on": list(spec.depends_on),
                "params": spec.params,
            }
            for spec in plan
        ],
    }


def _path_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "bytes": resolved.stat().st_size if resolved.is_file() else None,
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def _stage_record_reusable(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    try:
        stage_record = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if stage_record.get("status") != "completed" or stage_record.get("signature") != signature:
        return False
    for record in [*(stage_record.get("inputs") or []), *(stage_record.get("outputs") or [])]:
        source = Path(str(record["path"]))
        if not source.is_file() or sha256_file(source) != record.get("sha256"):
            return False
    return True


@contextmanager
def launch_lock(launch_dir: Path) -> Iterator[None]:
    """Hold a process-scoped launch lock that survives stale lock files safely.

    ``FileLock`` uses an operating-system advisory lock.  The lock is therefore
    released by the kernel if a job is killed or its node disappears; the
    persistent lock file itself is harmless and must not require manual cleanup.
    Owner metadata lives beside it so a concurrent refusal remains diagnosable.
    """

    lock = launch_dir / ".pipeline.lock"
    owner_path = launch_dir / ".pipeline.lock.owner.json"
    launch_dir.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": utc_now()}
    advisory_lock = FileLock(str(lock))
    try:
        advisory_lock.acquire(timeout=0)
    except FileLockTimeout as error:
        try:
            existing = read_json(owner_path)
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {"status": "owner metadata unavailable"}
        raise RuntimeError(
            f"Launch lock is active: {lock}; owner={existing}. "
            "Another process is already using this launch directory."
        ) from error
    try:
        atomic_write_json(owner_path, owner)
        yield
    finally:
        try:
            owner_path.unlink()
        except FileNotFoundError:
            pass
        advisory_lock.release()


class PipelineRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        launch_dir: Path,
        config: dict[str, Any],
        handlers: Mapping[str, Handler],
    ) -> None:
        self.project_root = project_root.resolve()
        self.launch_dir = launch_dir.resolve()
        self.config = config
        self.plan = build_plan(config)
        self.handlers = dict(handlers)
        self.source_digest = source_tree_digest(self.project_root)
        self.config_digest = digest_object(config)
        self.plan_digest = digest_object(plan_payload(self.plan))
        self.state_path = self.launch_dir / "pipeline.state.json"
        self.logger = EventLogger(self.launch_dir / "logs" / "events.jsonl")

    def initialize(self) -> dict[str, Any]:
        self.launch_dir.mkdir(parents=True, exist_ok=True)
        frozen = self.launch_dir / "launch.json"
        identity = {
            "schema_version": 1,
            "source_digest": self.source_digest,
            "config_digest": self.config_digest,
            "plan_digest": self.plan_digest,
        }
        if frozen.is_file():
            previous = read_json(frozen)
            for key, expected in identity.items():
                if previous.get(key) != expected:
                    raise RuntimeError(
                        f"Refusing to resume launch after {key} changed: {self.launch_dir}"
                    )
            state = read_json(self.state_path)
            self.logger.log("launch_resumed", launch_dir=str(self.launch_dir))
            return state

        identity.update({"created_at": utc_now(), "launch_dir": str(self.launch_dir)})
        atomic_write_json(frozen, identity)
        atomic_write_text(
            self.launch_dir / "config.resolved.yaml",
            yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True),
        )
        atomic_write_json(self.launch_dir / "pipeline.plan.json", plan_payload(self.plan))
        state = {
            "schema_version": 1,
            "status": "pending",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "stages": {
                spec.stage_id: {
                    "status": "pending",
                    "attempts": 0,
                    "stage_record": None,
                    "error": None,
                }
                for spec in self.plan
            },
        }
        atomic_write_json(self.state_path, state)
        self.logger.log("launch_created", launch_dir=str(self.launch_dir))
        return state

    def _stage_signature(self, spec: StageSpec, state: dict[str, Any]) -> str:
        dependencies = {
            dependency: state["stages"][dependency].get("result_signature")
            for dependency in spec.depends_on
        }
        return digest_object(
            {
                "stage_id": spec.stage_id,
                "handler": spec.handler,
                "params": spec.params,
                "dependencies": dependencies,
                "source_digest": self.source_digest,
                "config_digest": self.config_digest,
            }
        )

    def run(self) -> dict[str, Any]:
        state = self.initialize()
        with launch_lock(self.launch_dir):
            state["status"] = "running"
            atomic_write_json(self.state_path, state)
            for spec in self.plan:
                stage_state = state["stages"][spec.stage_id]
                signature = self._stage_signature(spec, state)
                stage_dir = self.launch_dir / "stages" / spec.stage_id
                stage_record_path = stage_dir / "stage_record.json"
                if _stage_record_reusable(stage_record_path, signature):
                    stage_record = read_json(stage_record_path)
                    stage_state.update(
                        {
                            "status": "completed",
                            "stage_record": str(stage_record_path),
                            "signature": signature,
                            "result_signature": stage_record["result_signature"],
                            "error": None,
                        }
                    )
                    self.logger.log("stage_reused", stage_id=spec.stage_id)
                    atomic_write_json(self.state_path, state)
                    continue
                for dependency in spec.depends_on:
                    if state["stages"][dependency]["status"] != "completed":
                        raise RuntimeError(f"Dependency {dependency} did not complete")
                if spec.handler not in self.handlers:
                    raise KeyError(f"No handler registered for {spec.handler}")
                stage_dir.mkdir(parents=True, exist_ok=True)
                stage_state["attempts"] = int(stage_state.get("attempts", 0)) + 1
                stage_state.update({"status": "running", "error": None, "signature": signature})
                atomic_write_json(self.state_path, state)
                before = snapshot(self.launch_dir)
                atomic_write_json(stage_dir / "resources_before.json", before)
                started = utc_now()
                self.logger.log("stage_started", stage_id=spec.stage_id, attempt=stage_state["attempts"])
                try:
                    result = self.handlers[spec.handler](
                        StageContext(
                            project_root=self.project_root,
                            launch_dir=self.launch_dir,
                            stage_dir=stage_dir,
                            config=self.config,
                            spec=spec,
                            event_logger=self.logger,
                        )
                    )
                    input_records = [_path_record(path) for path in result.inputs]
                    output_records = [_path_record(path) for path in result.outputs]
                    missing = [record["path"] for record in output_records if not record["exists"]]
                    if missing:
                        raise RuntimeError(f"Stage declared missing outputs: {missing}")
                    after = snapshot(self.launch_dir)
                    atomic_write_json(stage_dir / "resources_after.json", after)
                    result_signature = digest_object(
                        {
                            "signature": signature,
                            "outputs": output_records,
                            "metrics": result.metrics,
                            "details": result.details,
                        }
                    )
                    stage_record = {
                        "schema_version": 1,
                        "stage_id": spec.stage_id,
                        "handler": spec.handler,
                        "status": "completed",
                        "attempt": stage_state["attempts"],
                        "started_at": started,
                        "ended_at": utc_now(),
                        "signature": signature,
                        "result_signature": result_signature,
                        "inputs": input_records,
                        "outputs": output_records,
                        "metrics": result.metrics,
                        "details": result.details,
                    }
                    atomic_write_json(stage_record_path, stage_record)
                    stage_state.update(
                        {
                            "status": "completed",
                            "stage_record": str(stage_record_path),
                            "result_signature": result_signature,
                            "error": None,
                        }
                    )
                    self.logger.log("stage_completed", stage_id=spec.stage_id, **result.metrics)
                except Exception as error:
                    error_payload = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    nonblocking = (
                        str(self.config.get("execution", {}).get("policy", ""))
                        == "exhaustive_nonblocking_v1"
                        and spec.stage_id.startswith(
                            (
                                "06.end_to_end_smoke_test",
                                "07.small_sample_overfit_check",
                                "08.experiment.",
                            )
                        )
                    )
                    if nonblocking:
                        failure_path = atomic_write_json(
                            stage_dir / "stage_exception.nonblocking.json",
                            {
                                "schema_version": 1,
                                "status": "execution_failed_nonblocking",
                                "stage_id": spec.stage_id,
                                "handler": spec.handler,
                                "error": error_payload,
                                "findings_block_other_experiments": False,
                            },
                        )
                        output_records = [_path_record(failure_path)]
                        after = snapshot(self.launch_dir)
                        atomic_write_json(stage_dir / "resources_after.json", after)
                        failure_details = {
                            "status": "execution_failed_nonblocking",
                            "error": error_payload,
                            "findings_block_other_experiments": False,
                        }
                        result_signature = digest_object(
                            {
                                "signature": signature,
                                "outputs": output_records,
                                "metrics": {"execution_succeeded": 0.0},
                                "details": failure_details,
                            }
                        )
                        stage_record = {
                            "schema_version": 1,
                            "stage_id": spec.stage_id,
                            "handler": spec.handler,
                            "status": "completed",
                            "attempt": stage_state["attempts"],
                            "started_at": started,
                            "ended_at": utc_now(),
                            "signature": signature,
                            "result_signature": result_signature,
                            "inputs": [],
                            "outputs": output_records,
                            "metrics": {"execution_succeeded": 0.0},
                            "details": failure_details,
                        }
                        atomic_write_json(stage_record_path, stage_record)
                        stage_state.update(
                            {
                                "status": "completed",
                                "stage_record": str(stage_record_path),
                                "result_signature": result_signature,
                                "error": error_payload,
                                "nonblocking_failure": True,
                            }
                        )
                        state["updated_at"] = utc_now()
                        atomic_write_json(self.state_path, state)
                        self.logger.log(
                            "stage_failed_nonblocking",
                            stage_id=spec.stage_id,
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                        continue
                    stage_state.update(
                        {
                            "status": "failed",
                            "error": error_payload,
                        }
                    )
                    state["status"] = "failed"
                    state["updated_at"] = utc_now()
                    atomic_write_json(self.state_path, state)
                    self.logger.log(
                        "stage_failed",
                        stage_id=spec.stage_id,
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                    raise
                state["updated_at"] = utc_now()
                atomic_write_json(self.state_path, state)
            state["status"] = "completed"
            state["updated_at"] = utc_now()
            atomic_write_json(self.state_path, state)
            self.logger.log("launch_completed", launch_dir=str(self.launch_dir))
        return state


def _failed_verification(
    root: Path,
    failures: list[str],
    *,
    complete: bool = False,
    current_source_matches: bool | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "launch_dir": str(root),
        "integrity": False,
        "artifact_integrity_passed": False,
        "complete": complete,
        "pipeline_exhausted_plan": complete,
        "all_requested_computations_succeeded": False,
        "exhaustive_execution_completed": False,
        "current_source_matches": current_source_matches,
        "passed": False,
        "failures": failures,
        "computation_failures": [],
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _validate_versioned_mapping(value: Any, label: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{label} must be a JSON object"]
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        return [f"{label}.schema_version must be integer 1"]
    return []


def _validate_plan_schema(plan: Any) -> list[str]:
    failures = _validate_versioned_mapping(plan, "frozen plan")
    if failures:
        return failures
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        return ["frozen plan.stages must be a non-empty list"]
    stage_ids: list[str] = []
    dependencies: dict[str, list[str]] = {}
    for index, stage in enumerate(stages):
        label = f"frozen plan.stages[{index}]"
        if not isinstance(stage, Mapping):
            failures.append(f"{label} must be an object")
            continue
        stage_id = stage.get("stage_id")
        if (
            not isinstance(stage_id, str)
            or not stage_id
            or stage_id in {".", ".."}
            or "/" in stage_id
            or "\\" in stage_id
        ):
            failures.append(f"{label}.stage_id must be a safe non-empty string")
        else:
            stage_ids.append(stage_id)
        if not isinstance(stage.get("handler"), str) or not stage.get("handler"):
            failures.append(f"{label}.handler must be a non-empty string")
        depends_on = stage.get("depends_on")
        if not isinstance(depends_on, list) or not all(
            isinstance(value, str) and value for value in depends_on
        ):
            failures.append(f"{label}.depends_on must be a list of non-empty strings")
        elif isinstance(stage_id, str) and stage_id:
            dependencies[stage_id] = depends_on
        if not isinstance(stage.get("params"), Mapping):
            failures.append(f"{label}.params must be an object")
    if len(set(stage_ids)) != len(stage_ids):
        failures.append("frozen plan stage_id values must be unique")
    known_ids = set(stage_ids)
    for stage_id, depends_on in dependencies.items():
        missing = sorted(set(depends_on) - known_ids)
        if missing:
            failures.append(f"frozen plan stage {stage_id} has unknown dependencies: {missing}")
    return failures


def _validate_state_schema(state: Any) -> list[str]:
    failures = _validate_versioned_mapping(state, "pipeline state")
    if failures:
        return failures
    if state.get("status") not in {"pending", "running", "failed", "completed"}:
        failures.append("pipeline state.status is invalid")
    stages = state.get("stages")
    if not isinstance(stages, Mapping):
        failures.append("pipeline state.stages must be an object")
        return failures
    for stage_id, stage_state in stages.items():
        label = f"pipeline state.stages[{stage_id!r}]"
        if not isinstance(stage_id, str) or not stage_id:
            failures.append("pipeline state stage keys must be non-empty strings")
            continue
        if not isinstance(stage_state, Mapping):
            failures.append(f"{label} must be an object")
            continue
        if stage_state.get("status") not in {"pending", "running", "failed", "completed"}:
            failures.append(f"{label}.status is invalid")
        if stage_state.get("status") == "completed":
            if (
                not isinstance(stage_state.get("stage_record"), str)
                or not stage_state.get("stage_record")
            ):
                failures.append(f"{label}.stage_record must be a non-empty string")
            for key in ("signature", "result_signature"):
                if not _valid_sha256(stage_state.get(key)):
                    failures.append(f"{label}.{key} must be a SHA-256 digest")
    return failures


def _validate_identity_schema(identity: Any) -> list[str]:
    failures = _validate_versioned_mapping(identity, "launch identity")
    if failures:
        return failures
    for key in ("source_digest", "config_digest", "plan_digest"):
        if not _valid_sha256(identity.get(key)):
            failures.append(f"launch identity.{key} must be a SHA-256 digest")
    return failures


def _validate_stage_record_schema(stage_record: Any, stage_id: str) -> list[str]:
    label = f"stage_record for {stage_id}"
    failures = _validate_versioned_mapping(stage_record, label)
    if failures:
        return failures
    for key in ("stage_id", "handler", "status"):
        if not isinstance(stage_record.get(key), str) or not stage_record.get(key):
            failures.append(f"{label}.{key} must be a non-empty string")
    for key in ("signature", "result_signature"):
        if not _valid_sha256(stage_record.get(key)):
            failures.append(f"{label}.{key} must be a SHA-256 digest")
    for key in ("metrics", "details"):
        if not isinstance(stage_record.get(key), Mapping):
            failures.append(f"{label}.{key} must be an object")
    for kind in ("inputs", "outputs"):
        records = stage_record.get(kind)
        if not isinstance(records, list):
            failures.append(f"{label}.{kind} must be a list")
            continue
        for index, record in enumerate(records):
            record_label = f"{label}.{kind}[{index}]"
            if not isinstance(record, Mapping):
                failures.append(f"{record_label} must be an object")
                continue
            path = record.get("path")
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                failures.append(f"{record_label}.path must be a non-empty absolute string")
            if not _valid_sha256(record.get("sha256")):
                failures.append(f"{record_label}.sha256 must be a SHA-256 digest")
    return failures


def _verify_launch_impl(root: Path) -> dict[str, Any]:
    state_path = root / "pipeline.state.json"
    plan_path = root / "pipeline.plan.json"
    launch_path = root / "launch.json"
    config_path = root / "config.resolved.yaml"
    failures: list[str] = []
    if not all(path.is_file() for path in (state_path, plan_path, launch_path, config_path)):
        return _failed_verification(
            root,
            ["missing launch/config/plan/state identity artifact"],
        )
    state = read_json(state_path)
    plan = read_json(plan_path)
    identity = read_json(launch_path)
    resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema_failures = [
        *_validate_state_schema(state),
        *_validate_plan_schema(plan),
        *_validate_identity_schema(identity),
    ]
    if not isinstance(resolved_config, Mapping):
        schema_failures.append("resolved config must be a YAML object")
    elif not isinstance(resolved_config.get("runtime"), Mapping):
        schema_failures.append("resolved config.runtime must be an object")
    if schema_failures:
        return _failed_verification(
            root,
            [
                f"malformed or unreadable artifact schema: {failure}"
                for failure in schema_failures
            ],
        )
    if digest_object(plan) != identity.get("plan_digest"):
        failures.append("frozen plan digest mismatch")
    if digest_object(resolved_config) != identity.get("config_digest"):
        failures.append("frozen config digest mismatch")
    project_root_value = (resolved_config.get("runtime") or {}).get("project_root")
    current_source_matches: bool | None = None
    if project_root_value:
        project_root = Path(str(project_root_value)).resolve()
        if project_root.is_dir():
            current_source_matches = (
                source_tree_digest(project_root) == identity.get("source_digest")
            )
            if not current_source_matches:
                failures.append("current source tree differs from frozen launch source")
        else:
            failures.append(f"frozen project_root no longer exists: {project_root}")
    else:
        failures.append("resolved config lacks runtime.project_root")
    complete = state.get("status") == "completed"
    computation_failures: list[str] = []
    planned_ids = [str(stage["stage_id"]) for stage in plan.get("stages", [])]
    if set(state.get("stages", {})) != set(planned_ids):
        failures.append("pipeline state stage set differs from frozen plan")
    for stage in plan["stages"]:
        stage_id = stage["stage_id"]
        stage_state = state.get("stages", {}).get(stage_id, {})
        if stage_state.get("status") != "completed":
            failures.append(f"stage not completed: {stage_id}")
            continue
        stage_record_path = Path(str(stage_state.get("stage_record", "")))
        expected_stage_record_path = root / "stages" / stage_id / "stage_record.json"
        if stage_record_path.resolve() != expected_stage_record_path.resolve():
            failures.append(f"stage_record path escaped frozen stage directory: {stage_id}")
            continue
        if not stage_record_path.is_file():
            failures.append(f"missing stage_record: {stage_id}")
            continue
        stage_record = read_json(stage_record_path)
        stage_record_schema_failures = _validate_stage_record_schema(stage_record, stage_id)
        if stage_record_schema_failures:
            failures.extend(
                f"malformed or unreadable artifact schema: {failure}"
                for failure in stage_record_schema_failures
            )
            continue
        if stage_record.get("status") != "completed":
            failures.append(f"stage_record not completed: {stage_id}")
        execution_metric = (stage_record.get("metrics") or {}).get("execution_succeeded")
        if execution_metric is not None and not bool(execution_metric):
            computation_failures.append(f"requested computation failed: {stage_id}")
        if bool((stage_record.get("details") or {}).get("nonblocking_failure")):
            computation_failures.append(f"nonblocking stage failure: {stage_id}")
        if stage_record.get("signature") != stage_state.get("signature"):
            failures.append(f"stage_record/state signature mismatch: {stage_id}")
        if stage_record.get("result_signature") != stage_state.get("result_signature"):
            failures.append(f"stage_record/state result signature mismatch: {stage_id}")
        expected_result_signature = digest_object(
            {
                "signature": stage_record.get("signature"),
                "outputs": stage_record.get("outputs") or [],
                "metrics": stage_record.get("metrics") or {},
                "details": stage_record.get("details") or {},
            }
        )
        if stage_record.get("result_signature") != expected_result_signature:
            failures.append(f"stage_record result signature content mismatch: {stage_id}")
        if stage_record.get("stage_id") != stage_id or stage_record.get("handler") != stage.get("handler"):
            failures.append(f"stage_record identity mismatch: {stage_id}")
        for kind in ("inputs", "outputs"):
            for record in stage_record.get(kind) or []:
                output = Path(record["path"])
                if not output.is_file():
                    failures.append(f"missing {kind[:-1]}: {output}")
                elif sha256_file(output) != record.get("sha256"):
                    failures.append(f"{kind[:-1]} hash mismatch: {output}")
    integrity = not failures
    computation_failures = sorted(set(computation_failures))
    all_computations = bool(complete and not computation_failures)
    return {
        "schema_version": 1,
        "launch_dir": str(root),
        "integrity": integrity,
        "artifact_integrity_passed": integrity,
        "complete": complete,
        "pipeline_exhausted_plan": complete,
        "all_requested_computations_succeeded": all_computations,
        "exhaustive_execution_completed": bool(integrity and complete),
        "current_source_matches": current_source_matches,
        "passed": bool(integrity and complete and all_computations),
        "failures": failures,
        "computation_failures": computation_failures,
    }


def verify_launch(launch_dir: str | Path) -> dict[str, Any]:
    """Verify a frozen launch, failing closed on unreadable or malformed artifacts."""

    root = Path(launch_dir)
    try:
        root = root.resolve()
        return _verify_launch_impl(root)
    except Exception as error:
        return _failed_verification(
            root,
            [
                "verification failed closed on malformed or unreadable artifact: "
                f"{type(error).__name__}: {error}"
            ],
        )
