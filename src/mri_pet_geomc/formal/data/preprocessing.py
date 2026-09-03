"""Resumable offline preprocessing into deterministic mmap shards."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from ...utils import atomic_write_json, digest_object, read_json, utc_now
from .cache import CacheEntry, ShardStore
from .reference import MaterializedReference, ReferenceSpec, materialize_reference
from .schema import CatalogObservation, FormalDataError


ProgressCallback = Callable[["PreprocessProgress"], None]
_SIMPLEITK_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class PreprocessingConfig:
    output_root: Path
    source_reference: Path
    source_mask: Path
    target_shape: tuple[int, int, int] = (128, 128, 128)
    target_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0)
    shard_size: int = 128
    workers: int = 4
    interpolation_order: int = 1
    registration_mode: str = "simpleitk_rigid_affine"
    mri_final_interpolation: str = "bspline"
    pet_final_interpolation: str = "linear"
    registration_sampling_fraction: float = 0.20
    rigid_iterations: int = 100
    affine_iterations: int = 100
    sitk_threads_per_worker: int = 1
    dcm2niix_command: tuple[str, ...] = (
        "dcm2niix",
        "-z",
        "y",
        "-b",
        "n",
        "-f",
        "converted",
        "-o",
        "{output_dir}",
        "{input_dir}",
    )
    registration_command: tuple[str, ...] = ()
    retry_failures: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "source_reference", Path(self.source_reference))
        object.__setattr__(self, "source_mask", Path(self.source_mask))
        object.__setattr__(self, "target_shape", tuple(int(v) for v in self.target_shape))
        object.__setattr__(self, "target_spacing", tuple(float(v) for v in self.target_spacing))
        object.__setattr__(self, "dcm2niix_command", tuple(self.dcm2niix_command))
        object.__setattr__(self, "registration_command", tuple(self.registration_command))
        if len(self.target_shape) != 3 or any(value <= 0 for value in self.target_shape):
            raise FormalDataError(f"Invalid target_shape: {self.target_shape}")
        if len(self.target_spacing) != 3 or any(value <= 0 for value in self.target_spacing):
            raise FormalDataError(f"Invalid target_spacing: {self.target_spacing}")
        if self.shard_size < 1 or self.workers < 1:
            raise FormalDataError("shard_size and workers must be positive")
        if self.interpolation_order not in {0, 1, 2, 3}:
            raise FormalDataError("interpolation_order must be in [0, 3]")
        if not self.dcm2niix_command:
            raise FormalDataError("dcm2niix_command cannot be empty")
        if self.registration_mode not in {"simpleitk_rigid_affine", "external", "direct"}:
            raise FormalDataError(
                "registration_mode must be simpleitk_rigid_affine, external or direct"
            )
        if self.registration_mode == "external" and not self.registration_command:
            raise FormalDataError("external registration_mode requires registration_command")
        if self.mri_final_interpolation not in {"linear", "bspline"}:
            raise FormalDataError("MRI interpolation must be linear or bspline")
        if self.pet_final_interpolation != "linear":
            raise FormalDataError("PET final interpolation is fixed to linear")
        if not 0.0 < self.registration_sampling_fraction <= 1.0:
            raise FormalDataError("registration_sampling_fraction must be in (0, 1]")
        if self.rigid_iterations < 1 or self.affine_iterations < 1:
            raise FormalDataError("Registration iteration counts must be positive")
        if self.sitk_threads_per_worker < 1:
            raise FormalDataError("sitk_threads_per_worker must be positive")


@dataclass(frozen=True)
class PreprocessProgress:
    event: str
    completed: int
    skipped: int
    failed: int
    total: int
    observation_uid: str = ""
    message: str = ""
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class PreprocessRunSummary:
    total: int
    completed: int
    skipped: int
    failed: int
    pending: int
    elapsed_seconds: float
    cache_entries: tuple[CacheEntry, ...]
    reference: MaterializedReference
    contract_digest: str
    plan_digest: str


@dataclass(frozen=True)
class _CaseResult:
    status: str
    observation_uid: str
    message: str = ""


class _FailureLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, observation: CatalogObservation, error: BaseException) -> None:
        row = {
            "schema_version": 1,
            "timestamp": utc_now(),
            "observation_uid": observation.observation_uid,
            "dataset_id": observation.dataset_id,
            "source": observation.source.to_dict(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())


def _safe_zip_member(archive: Path, requested: str, output_dir: Path) -> Path:
    normalized = requested.replace("\\", "/").lstrip("./")
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        raise FormalDataError(f"Unsafe ZIP member path: {requested!r}")
    with zipfile.ZipFile(archive, "r") as handle:
        names = {name.replace("\\", "/").lstrip("./"): name for name in handle.namelist()}
        requested_parts = normalized.split("/")
        candidates = [normalized]
        if len(requested_parts) >= 3 and requested_parts[0].startswith("sub-"):
            candidates.append("/".join(requested_parts[1:]))
            if requested_parts[1].startswith("ses-"):
                candidates.append("/".join(requested_parts[2:]))
        matches = [names[candidate] for candidate in candidates if candidate in names]
        if not matches:
            raise FormalDataError(f"ZIP member is absent: {archive}!{requested}")
        if len(set(matches)) != 1:
            raise FormalDataError(
                f"ZIP member lookup is ambiguous for {archive}!{requested}: {matches}"
            )
        stored_name = matches[0]
        suffix = ".nii.gz" if normalized.lower().endswith(".nii.gz") else Path(normalized).suffix
        destination = output_dir / f"source{suffix or '.nii'}"
        with handle.open(stored_name, "r") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
    return destination


def _run_command(template: Sequence[str], values: Mapping[str, str]) -> None:
    command = [part.format_map(values) for part in template]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
        raise FormalDataError(
            f"External command failed ({completed.returncode}): {command!r}; {detail}"
        )


def _dicom_to_nifti(source_dir: Path, output_dir: Path, command: Sequence[str]) -> Path:
    if not source_dir.is_dir():
        raise FormalDataError(f"DICOM series directory does not exist: {source_dir}")
    _run_command(command, {"input_dir": str(source_dir), "output_dir": str(output_dir)})
    candidates = sorted(
        [*output_dir.glob("*.nii"), *output_dir.glob("*.nii.gz")],
        key=lambda path: path.name,
    )
    if len(candidates) != 1:
        raise FormalDataError(
            f"dcm2niix must produce exactly one NIfTI for {source_dir}; got {candidates}"
        )
    return candidates[0]


def _three_dimensional(image: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    values = np.asanyarray(image.dataobj)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3:
        raise FormalDataError(f"Input image must be 3D or singleton-4D, got {values.shape}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise FormalDataError("Input image affine must be a finite 4x4 matrix")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-8:
        raise FormalDataError("Input image affine is singular")
    return nib.Nifti1Image(values, affine)


def _configure_registration(
    registration: Any,
    *,
    iterations: int,
    sampling_fraction: float,
    seed: int,
    threads: int,
) -> None:
    import SimpleITK as sitk

    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    if sampling_fraction < 1.0:
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sampling_fraction, seed)
    else:
        registration.SetMetricSamplingStrategy(registration.NONE)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=int(iterations),
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
        estimateLearningRate=registration.EachIteration,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    if hasattr(registration, "SetNumberOfThreads"):
        registration.SetNumberOfThreads(int(threads))


def _rigid_to_affine(sitk: Any, transform: Any) -> Any:
    """Copy an optimized Euler rigid transform into an affine initializer."""

    value = transform
    if value.GetName() == "CompositeTransform":
        composite = sitk.CompositeTransform(value)
        if composite.GetNumberOfTransforms() < 1:
            raise FormalDataError("Rigid registration returned an empty composite transform")
        value = composite.GetBackTransform()
    rigid = sitk.Euler3DTransform(value)
    affine = sitk.AffineTransform(3)
    affine.SetCenter(rigid.GetCenter())
    affine.SetMatrix(rigid.GetMatrix())
    affine.SetTranslation(rigid.GetTranslation())
    return affine


def _sitk_resample(
    sitk: Any,
    image: Any,
    fixed: Any,
    transform: Any,
    *,
    interpolator: int,
    default_value: float,
    output_pixel_type: int,
    threads: int,
) -> Any:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetOutputPixelType(output_pixel_type)
    if hasattr(resampler, "SetNumberOfThreads"):
        resampler.SetNumberOfThreads(int(threads))
    return resampler.Execute(image)


def _simpleitk_rigid_affine(
    moving_path: Path,
    observation: CatalogObservation,
    reference: MaterializedReference,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Optimize rigid then affine transforms, with one final image resample."""

    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise FormalDataError(
            "SimpleITK is required for registration_mode=simpleitk_rigid_affine"
        ) from error
    fixed_key = (str(reference.reference_path), str(reference.mask_path), reference.contract_digest)
    cached = getattr(_SIMPLEITK_THREAD_LOCAL, "fixed", None)
    if cached is None or cached[0] != fixed_key:
        fixed = sitk.ReadImage(str(reference.reference_path), sitk.sitkFloat32)
        fixed_mask = sitk.Cast(sitk.ReadImage(str(reference.mask_path)), sitk.sitkUInt8)
        cached = (fixed_key, fixed, fixed_mask)
        _SIMPLEITK_THREAD_LOCAL.fixed = cached
    else:
        _, fixed, fixed_mask = cached
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    rigid_initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    seed = int.from_bytes(observation.observation_uid.encode("utf-8")[:4], "little") or 1
    rigid_registration = sitk.ImageRegistrationMethod()
    _configure_registration(
        rigid_registration,
        iterations=config.rigid_iterations,
        sampling_fraction=config.registration_sampling_fraction,
        seed=seed,
        threads=config.sitk_threads_per_worker,
    )
    rigid_registration.SetMetricFixedMask(fixed_mask)
    rigid_registration.SetInitialTransform(rigid_initial, inPlace=False)
    rigid_result = rigid_registration.Execute(fixed, moving)

    # Stage 2 starts from an explicit affine copy of the optimized rigid
    # transform.  We do not rely on SetMovingInitialTransform being retained in
    # Execute's returned transform.
    affine_initial = _rigid_to_affine(sitk, rigid_result)
    affine_registration = sitk.ImageRegistrationMethod()
    _configure_registration(
        affine_registration,
        iterations=config.affine_iterations,
        sampling_fraction=config.registration_sampling_fraction,
        seed=seed + 1,
        threads=config.sitk_threads_per_worker,
    )
    affine_registration.SetMetricFixedMask(fixed_mask)
    affine_registration.SetInitialTransform(affine_initial, inPlace=False)
    final_transform = affine_registration.Execute(fixed, moving)

    interpolation_name = (
        config.mri_final_interpolation if observation.modality == "mri" else config.pet_final_interpolation
    )
    interpolator = sitk.sitkBSpline if interpolation_name == "bspline" else sitk.sitkLinear
    final_image = _sitk_resample(
        sitk,
        moving,
        fixed,
        final_transform,
        interpolator=interpolator,
        default_value=0.0,
        output_pixel_type=sitk.sitkFloat32,
        threads=config.sitk_threads_per_worker,
    )
    moving_support = sitk.Image(moving.GetSize(), sitk.sitkUInt8)
    moving_support.CopyInformation(moving)
    moving_support = moving_support + 1
    final_support = _sitk_resample(
        sitk,
        moving_support,
        fixed,
        final_transform,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
        output_pixel_type=sitk.sitkUInt8,
        threads=config.sitk_threads_per_worker,
    )
    # SimpleITK arrays are [z,y,x]; transpose to nibabel [x,y,z].
    values = np.transpose(sitk.GetArrayFromImage(final_image), (2, 1, 0)).astype(
        np.float32, copy=False
    )
    support = np.transpose(sitk.GetArrayFromImage(final_support), (2, 1, 0)) > 0
    target_mask = np.transpose(sitk.GetArrayFromImage(fixed_mask), (2, 1, 0)) > 0
    support &= target_mask & np.isfinite(values)
    if not support.any():
        raise FormalDataError("Registered case has empty structural support")
    values[~support] = 0.0
    transform_payload = {
        "type": final_transform.GetName(),
        "parameters": [float(value) for value in final_transform.GetParameters()],
        "fixed_parameters": [
            float(value) for value in final_transform.GetFixedParameters()
        ],
        "rigid_parameters": [float(value) for value in rigid_result.GetParameters()],
        "rigid_metric": float(rigid_registration.GetMetricValue()),
        "affine_metric": float(affine_registration.GetMetricValue()),
        "rigid_stop": rigid_registration.GetOptimizerStopConditionDescription(),
        "affine_stop": affine_registration.GetOptimizerStopConditionDescription(),
    }
    transform_payload["sha256"] = digest_object(transform_payload)
    return values, support, transform_payload


class OfflinePreprocessor:
    """Case-tolerant executor with exact per-case resume receipts."""

    def __init__(
        self,
        observations: Sequence[CatalogObservation],
        config: PreprocessingConfig,
        *,
        progress: ProgressCallback | None = None,
    ) -> None:
        if not observations:
            raise FormalDataError("Offline preprocessing requires observations")
        self.observations = tuple(sorted(observations, key=lambda row: row.observation_uid))
        self.config = config
        self.progress = progress
        self.receipt_dir = config.output_root / "receipts"
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.failure_ledger = _FailureLedger(config.output_root / "failures.jsonl")
        self._store_lock = threading.Lock()
        self._target_mask: np.ndarray | None = None

    def _emit(
        self,
        event: str,
        completed: int,
        skipped: int,
        failed: int,
        total: int,
        start: float,
        *,
        observation_uid: str = "",
        message: str = "",
    ) -> None:
        if self.progress is not None:
            self.progress(
                PreprocessProgress(
                    event=event,
                    completed=completed,
                    skipped=skipped,
                    failed=failed,
                    total=total,
                    observation_uid=observation_uid,
                    message=message,
                    elapsed_seconds=time.monotonic() - start,
                )
            )

    def _receipt_valid(
        self,
        observation: CatalogObservation,
        *,
        contract_digest: str,
        store: ShardStore,
    ) -> bool:
        path = self.receipt_dir / f"{observation.observation_uid}.json"
        if not path.is_file():
            return False
        value = read_json(path)
        if not isinstance(value, Mapping):
            return False
        entry = store.entry(observation.observation_uid)
        return (
            value.get("status") == "complete"
            and value.get("observation_uid") == observation.observation_uid
            and value.get("source_signature") == observation.source.signature
            and value.get("contract_digest") == contract_digest
            and value.get("plan_digest") == store.plan_digest
            and int(value.get("shard_id", -1)) == entry.shard_id
            and int(value.get("slot", -1)) == entry.slot
        )

    def _source_nifti(self, observation: CatalogObservation, temp: Path) -> Path:
        source = Path(observation.source.path)
        if observation.source.kind == "zip_member":
            if not source.is_file():
                raise FormalDataError(f"FOMO archive does not exist: {source}")
            return _safe_zip_member(source, observation.source.archive_member, temp)
        if observation.source.kind == "dicom_series":
            return _dicom_to_nifti(source, temp, self.config.dcm2niix_command)
        if observation.source.kind == "nifti":
            if not source.is_file():
                raise FormalDataError(f"NIfTI input does not exist: {source}")
            return source
        raise FormalDataError(f"Unsupported source kind: {observation.source.kind}")

    def _prepare_values(
        self,
        observation: CatalogObservation,
        reference: MaterializedReference,
    ) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="geomc-preprocess-") as temporary_text:
            temporary = Path(temporary_text)
            input_path = self._source_nifti(observation, temporary)
            native_image = _three_dimensional(nib.load(input_path))
            native_geometry = {
                "native_shape": [int(value) for value in native_image.shape],
                "native_spacing": [
                    float(value) for value in nib.affines.voxel_sizes(native_image.affine)
                ],
                "native_orientation": "".join(nib.aff2axcodes(native_image.affine)),
                "effective_shape": [int(value) for value in reference.shape],
                "effective_spacing": [float(value) for value in reference.spacing],
            }
            input_image = nib.as_closest_canonical(native_image)
            canonical_path = temporary / "canonical_input.nii.gz"
            nib.save(input_image, canonical_path)
            if self.config.registration_mode == "simpleitk_rigid_affine":
                values, support, transform_details = _simpleitk_rigid_affine(
                    canonical_path, observation, reference, self.config
                )
                if values.shape != reference.shape or support.shape != reference.shape:
                    raise FormalDataError(
                        f"Registration returned wrong shape: {values.shape}/{support.shape}"
                    )
                native_geometry["registration_transform"] = dict(transform_details)
                return values, support, native_geometry
            if self.config.registration_mode == "external":
                registered_path = temporary / "registered.nii.gz"
                _run_command(
                    self.config.registration_command,
                    {
                        "input": str(canonical_path),
                        "reference": str(reference.reference_path),
                        "output": str(registered_path),
                        "output_dir": str(temporary),
                    },
                )
                if not registered_path.is_file():
                    raise FormalDataError(
                        "registration_command succeeded but did not create {output}"
                    )
                input_image = nib.as_closest_canonical(
                    _three_dimensional(nib.load(registered_path))
                )
                if input_image.shape != reference.shape or not np.allclose(
                    input_image.affine, reference.affine, atol=1e-4, rtol=0.0
                ):
                    raise FormalDataError(
                        "external registration output must already match the frozen target lattice"
                    )
                if self._target_mask is None:
                    raise FormalDataError("Target mask was not initialized")
                target_mask = self._target_mask
                values = np.asarray(input_image.dataobj, dtype=np.float32)
                support = target_mask & np.isfinite(values)
                if not support.any():
                    raise FormalDataError("Externally registered case has empty support")
                values[~support] = 0.0
                return values, support, native_geometry
            if self.config.registration_mode != "direct":
                raise FormalDataError(
                    f"Unsupported registration mode: {self.config.registration_mode}"
                )
            finite_source = np.isfinite(np.asanyarray(input_image.dataobj))
            finite_image = nib.Nifti1Image(finite_source.astype(np.uint8), input_image.affine)
            target = (reference.shape, reference.affine)
            resampled = resample_from_to(
                input_image,
                target,
                order=self.config.interpolation_order,
                cval=0.0,
            )
            finite_resampled = resample_from_to(finite_image, target, order=0, cval=0.0)
            if self._target_mask is None:
                raise FormalDataError("Target mask was not initialized")
            target_mask = self._target_mask
            values = np.asarray(resampled.dataobj, dtype=np.float32)
            support = (np.asarray(finite_resampled.dataobj) > 0) & target_mask
            support &= np.isfinite(values)
            if values.shape != reference.shape or support.shape != reference.shape:
                raise FormalDataError(
                    f"Resampling returned wrong shape: image={values.shape}, support={support.shape}"
                )
            if not support.any():
                raise FormalDataError("Resampled case has empty structural support")
            values[~support] = 0.0
            return values, support, native_geometry

    def _process_one(
        self,
        observation: CatalogObservation,
        *,
        reference: MaterializedReference,
        contract_digest: str,
        store: ShardStore,
    ) -> _CaseResult:
        if self._receipt_valid(observation, contract_digest=contract_digest, store=store):
            return _CaseResult("skipped", observation.observation_uid, "receipt-resume")
        try:
            values, support, geometry = self._prepare_values(observation, reference)
            maximum = float(np.max(np.abs(values[support])))
            storage_scale = (
                1.0
                if maximum <= 60000.0
                else float(2.0 ** int(np.ceil(np.log2(maximum / 60000.0))))
            )
            stored_values = values / storage_scale
            if not np.isfinite(stored_values.astype(np.float16)).all():
                raise FormalDataError("Image is non-finite after float16 storage scaling")
            with self._store_lock:
                entry = store.write(observation.observation_uid, stored_values, support)
            receipt = {
                "schema_version": 1,
                "timestamp": utc_now(),
                "status": "complete",
                "observation_uid": observation.observation_uid,
                "source_signature": observation.source.signature,
                "contract_digest": contract_digest,
                "plan_digest": store.plan_digest,
                "shard_id": entry.shard_id,
                "slot": entry.slot,
                "image_path": entry.image_path,
                "support_path": entry.support_path,
                "shape": list(reference.shape),
                "spacing": list(reference.spacing),
                "support_voxels": int(support.sum()),
                "registration_mode": self.config.registration_mode,
                "final_interpolation": (
                    self.config.mri_final_interpolation
                    if observation.modality == "mri"
                    else self.config.pet_final_interpolation
                ),
                "geometry": dict(geometry),
                "intensity_storage_scale": storage_scale,
                "stored_float16_finite": True,
            }
            atomic_write_json(self.receipt_dir / f"{observation.observation_uid}.json", receipt)
            return _CaseResult("completed", observation.observation_uid)
        except Exception as error:
            self.failure_ledger.append(observation, error)
            return _CaseResult("failed", observation.observation_uid, str(error))

    def run(self, *, max_cases: int | None = None) -> PreprocessRunSummary:
        start = time.monotonic()
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        reference = materialize_reference(
            ReferenceSpec(
                source_reference=self.config.source_reference,
                source_mask=self.config.source_mask,
                target_shape=self.config.target_shape,
                target_spacing=self.config.target_spacing,
            ),
            self.config.output_root / "reference",
        )
        self._target_mask = np.asarray(nib.load(reference.mask_path).dataobj) > 0
        contract_digest = preprocessing_contract_digest(
            self.config, reference.contract_digest
        )
        store = ShardStore(
            self.config.output_root,
            self.observations,
            shape=reference.shape,
            shard_size=self.config.shard_size,
            contract_digest=contract_digest,
        )
        selected = self.observations if max_cases is None else self.observations[: max(0, max_cases)]
        total = len(selected)
        completed = skipped = failed = 0
        self._emit("start", completed, skipped, failed, total, start)
        if self.config.workers == 1:
            results = (
                self._process_one(
                    row,
                    reference=reference,
                    contract_digest=contract_digest,
                    store=store,
                )
                for row in selected
            )
            for result in results:
                completed, skipped, failed = _update_counts(
                    result, completed, skipped, failed
                )
                self._emit(
                    result.status,
                    completed,
                    skipped,
                    failed,
                    total,
                    start,
                    observation_uid=result.observation_uid,
                    message=result.message,
                )
        else:
            with ThreadPoolExecutor(
                max_workers=self.config.workers, thread_name_prefix="geomc-preprocess"
            ) as executor:
                pending: dict[Future[_CaseResult], None] = {}
                iterator = iter(selected)
                window = max(self.config.workers * 2, 1)
                while len(pending) < window:
                    try:
                        row = next(iterator)
                    except StopIteration:
                        break
                    pending[
                        executor.submit(
                            self._process_one,
                            row,
                            reference=reference,
                            contract_digest=contract_digest,
                            store=store,
                        )
                    ] = None
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        result = future.result()
                        completed, skipped, failed = _update_counts(
                            result, completed, skipped, failed
                        )
                        self._emit(
                            result.status,
                            completed,
                            skipped,
                            failed,
                            total,
                            start,
                            observation_uid=result.observation_uid,
                            message=result.message,
                        )
                        try:
                            row = next(iterator)
                        except StopIteration:
                            continue
                        pending[
                            executor.submit(
                                self._process_one,
                                row,
                                reference=reference,
                                contract_digest=contract_digest,
                                store=store,
                            )
                        ] = None
        entries = store.freeze_manifest(self.receipt_dir)
        elapsed = time.monotonic() - start
        pending_count = total - completed - skipped - failed
        summary = PreprocessRunSummary(
            total=total,
            completed=completed,
            skipped=skipped,
            failed=failed,
            pending=pending_count,
            elapsed_seconds=elapsed,
            cache_entries=entries,
            reference=reference,
            contract_digest=contract_digest,
            plan_digest=store.plan_digest,
        )
        atomic_write_json(
            self.config.output_root / "preprocessing_summary.json",
            {
                "total": total,
                "completed": completed,
                "skipped": skipped,
                "failed": failed,
                "pending": pending_count,
                "elapsed_seconds": elapsed,
                "cache_complete_count": len(entries),
                "contract_digest": contract_digest,
                "plan_digest": store.plan_digest,
                "reference_receipt": str(reference.receipt_path),
                "reference_contract_digest": reference.contract_digest,
            },
        )
        self._emit("complete", completed, skipped, failed, total, start)
        return summary


def preprocessing_contract_digest(
    config: PreprocessingConfig,
    reference_contract: str,
) -> str:
    """Bind every cached case to one reference and preprocessing protocol."""

    if not str(reference_contract):
        raise FormalDataError("reference_contract is required")
    return digest_object(
        {
            "schema": 1,
            "reference_contract": str(reference_contract),
            "interpolation_order": config.interpolation_order,
            "registration_mode": config.registration_mode,
            "registration_command": config.registration_command,
            "mri_final_interpolation": config.mri_final_interpolation,
            "pet_final_interpolation": config.pet_final_interpolation,
            "registration_sampling_fraction": config.registration_sampling_fraction,
            "rigid_iterations": config.rigid_iterations,
            "affine_iterations": config.affine_iterations,
            "sitk_threads_per_worker": config.sitk_threads_per_worker,
            "cache_dtype": "float16",
            "cache_scaling": "positive_power_of_two_to_abs_le_60000",
            "support_policy": "finite_source_intersection_reference_mask",
        }
    )


def _update_counts(
    result: _CaseResult, completed: int, skipped: int, failed: int
) -> tuple[int, int, int]:
    if result.status == "completed":
        completed += 1
    elif result.status == "skipped":
        skipped += 1
    elif result.status == "failed":
        failed += 1
    else:
        raise FormalDataError(f"Unknown preprocessing result status: {result.status}")
    return completed, skipped, failed
