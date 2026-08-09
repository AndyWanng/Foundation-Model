"""Trainable, provenance-bound access to the SAT3D ``swin2`` image encoder.

The external SAT3D ``Sam3D.forward`` method is decorated with ``no_grad`` and
the builder returns the full model in evaluation mode.  This module therefore
loads the *whole* SAM checkpoint strictly, extracts ``image_encoder``, and only
ever calls that encoder directly.  The forward method in this wrapper is not
decorated with ``no_grad``; autograd behaviour is controlled solely by the
declared trainability policy and by the caller's context.

``TinySAT3DImageEncoder`` is deliberately an integration-test backend.  It lets CPU
tests exercise the same shape, policy, and gradient contracts, but it is not a
scientific substitute for the external SAT3D checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn


TrainabilityPolicy = Literal["frozen", "last_stage", "all"]

SAT3D_INPUT_SHAPE = (1, 128, 128, 128)
SAT3D_OUTPUT_SHAPE = (384, 8, 8, 8)
_SAT3D_PACKAGE = "segment_anything_with_swin_conf"
_SAT3D_BUILDER_MODULE = f"{_SAT3D_PACKAGE}.build_samswin3D"


def _sha256(path: str | Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    sources = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not sources:
        raise RuntimeError(f"SAT3D Python source tree is empty: {root}")
    for source in sources:
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(source)))
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_sat3d_python_root(code_root: str | Path) -> Path:
    root = Path(code_root).resolve()
    candidates = (root / "SAT3D-slicer" / "sat3D", root / "sat3D", root)
    for candidate in candidates:
        builder = candidate / _SAT3D_PACKAGE / "build_samswin3D.py"
        if builder.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find segment_anything_with_swin_conf/build_samswin3D.py "
        f"under {root}."
    )


def sat3d_source_tree_sha256(code_root: str | Path) -> tuple[Path, str]:
    """Resolve the audited Python root and hash every Python source file."""

    python_root = _resolve_sat3d_python_root(code_root)
    return python_root, _source_tree_sha256(python_root)


def _loaded_module_path(module: Any) -> Path | None:
    value = getattr(module, "__file__", None)
    if not value:
        return None
    path = Path(str(value))
    return path.resolve() if path.is_file() else None


def _import_swin2_builder(python_root: Path):
    """Import the requested SAT3D tree without accepting a stale same-name package."""

    for name, loaded in tuple(sys.modules.items()):
        if name != _SAT3D_PACKAGE and not name.startswith(f"{_SAT3D_PACKAGE}."):
            continue
        loaded_path = _loaded_module_path(loaded)
        if loaded_path is None or not _is_within(loaded_path, python_root):
            raise RuntimeError(
                "A different or untraceable segment_anything_with_swin_conf package "
                f"is already loaded ({name}: {loaded_path}); requested root={python_root}. "
                "Start a clean Python process."
            )

    root_text = str(python_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module(_SAT3D_BUILDER_MODULE)
    module_path = _loaded_module_path(module)
    if module_path is None or not _is_within(module_path, python_root):
        raise RuntimeError(
            f"Imported SAT3D builder {module_path} outside requested root {python_root}."
        )

    registry = getattr(module, "sam_model_registry3D", {})
    builder = registry.get("swin2") if isinstance(registry, Mapping) else None
    if builder is None:
        builder = getattr(module, "build_sam3D_swin2", None)
    if not callable(builder):
        found = sorted(str(key) for key in registry) if isinstance(registry, Mapping) else []
        raise KeyError(f"SAT3D builder 'swin2' is unavailable; registry keys={found}.")
    return module, builder


def _safe_torch_load(path: Path) -> Any:
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    manager = safe_globals([argparse.Namespace]) if safe_globals else contextlib.nullcontext()
    with manager:
        return torch.load(path, map_location="cpu", weights_only=True)


def _strip_ddp_prefix(state: Mapping[Any, Any]) -> OrderedDict[str, Any]:
    stripped: OrderedDict[str, Any] = OrderedDict()
    for raw_name, value in state.items():
        name = str(raw_name)
        normalized = name[len("module.") :] if name.startswith("module.") else name
        if normalized in stripped:
            raise RuntimeError(
                f"SAT3D checkpoint contains duplicate key after DDP prefix stripping: {normalized}"
            )
        stripped[normalized] = value
    return stripped


def _parameter_counts(module: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
    }


def _normalise_shape(shape: Sequence[int], *, name: str) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in shape)
    if len(result) != 4 or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain four positive integers [C,D,H,W].")
    return result  # type: ignore[return-value]


def _group_count(channels: int) -> int:
    for groups in range(min(channels, 4), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TinySAT3DImageEncoder(nn.Module):
    """Small 3D encoder with SAT3D-like ``layers``/``norm`` policy boundaries."""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        hidden_channels: int = 8,
        output_channels: int = 16,
        output_grid: int = 2,
    ) -> None:
        super().__init__()
        if min(input_channels, hidden_channels, output_channels, output_grid) <= 0:
            raise ValueError("Tiny backend dimensions must be positive.")
        self.output_channels = int(output_channels)
        self.output_grid = int(output_grid)
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv3d(
                        hidden_channels,
                        output_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(_group_count(output_channels), output_channels),
                    nn.GELU(),
                ),
            ]
        )
        self.pool = nn.AdaptiveAvgPool3d((output_grid, output_grid, output_grid))
        self.norm = nn.GroupNorm(_group_count(output_channels), output_channels)

    def forward(self, image: Tensor) -> Tensor:
        hidden = self.stem(image)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(self.pool(hidden))


class TrainableSAT3DEncoder(nn.Module):
    """Direct image-encoder wrapper with explicit trainability and provenance.

    ``last_stage`` means the final member of ``image_encoder.layers`` plus an
    optional top-level ``image_encoder.norm``.  Earlier frozen stages are kept
    in evaluation mode, so stochastic depth cannot perturb a supposedly fixed
    representation.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        *,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        trainability: TrainabilityPolicy,
        provenance: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if not isinstance(image_encoder, nn.Module):
            raise TypeError("image_encoder must be a torch.nn.Module.")
        if trainability not in {"frozen", "last_stage", "all"}:
            raise ValueError("trainability must be one of: frozen, last_stage, all.")

        self.image_encoder = image_encoder.float()
        self.input_shape = _normalise_shape(input_shape, name="input_shape")
        self.output_shape = _normalise_shape(output_shape, name="output_shape")
        self.trainability: TrainabilityPolicy = trainability
        self._last_stage_module_names = self._discover_last_stage_module_names()
        self._provenance = copy.deepcopy(dict(provenance))
        self._configure_trainability()
        self._provenance.update(
            {
                "trainability": self.trainability,
                "input_shape": list(self.input_shape),
                "output_shape": list(self.output_shape),
                "last_stage_module_names": list(self._last_stage_module_names),
                "parameter_counts": _parameter_counts(self.image_encoder),
                "forward_entrypoint": "image_encoder_direct",
                "sam_forward_used": False,
                "wrapper_no_grad": False,
            }
        )
        self.train(True)

    @classmethod
    def from_sat3d(
        cls,
        code_root: str | Path,
        checkpoint: str | Path,
        *,
        trainability: TrainabilityPolicy = "last_stage",
        input_shape: Sequence[int] = SAT3D_INPUT_SHAPE,
        output_shape: Sequence[int] = SAT3D_OUTPUT_SHAPE,
        expected_checkpoint_sha256: str | None = None,
        expected_source_tree_sha256: str | None = None,
    ) -> "TrainableSAT3DEncoder":
        """Strictly load the whole ``swin2`` SAM, then retain its image encoder."""

        python_root = _resolve_sat3d_python_root(code_root)
        builder_module, builder = _import_swin2_builder(python_root)
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint_sha256 = _sha256(checkpoint_path)
        if (
            expected_checkpoint_sha256 is not None
            and checkpoint_sha256.lower() != expected_checkpoint_sha256.lower()
        ):
            raise RuntimeError(
                "SAT3D checkpoint SHA-256 mismatch: "
                f"expected={expected_checkpoint_sha256}, actual={checkpoint_sha256}"
            )
        source_tree_sha256 = _source_tree_sha256(python_root)
        if (
            expected_source_tree_sha256 is not None
            and source_tree_sha256.lower() != expected_source_tree_sha256.lower()
        ):
            raise RuntimeError(
                "SAT3D source-tree SHA-256 mismatch: "
                f"expected={expected_source_tree_sha256}, actual={source_tree_sha256}"
            )

        # The external builder accepts a checkpoint, but using that route makes
        # strictness and DDP-prefix handling opaque.  Always construct first.
        whole_sam = builder(checkpoint=None)
        if not isinstance(whole_sam, nn.Module):
            raise TypeError("SAT3D swin2 builder did not return torch.nn.Module.")
        image_encoder = getattr(whole_sam, "image_encoder", None)
        if not isinstance(image_encoder, nn.Module):
            raise RuntimeError("SAT3D swin2 model does not expose image_encoder.")

        payload = _safe_torch_load(checkpoint_path)
        if not isinstance(payload, Mapping):
            raise RuntimeError("SAT3D checkpoint payload must be a mapping.")
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise RuntimeError("SAT3D checkpoint must contain model_state_dict.")
        strict_state = _strip_ddp_prefix(state)
        try:
            incompatible = whole_sam.load_state_dict(strict_state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(f"Strict whole-SAM checkpoint load failed: {error}") from error
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Strict whole-SAM checkpoint load returned incompatible keys: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

        builder_source = _loaded_module_path(builder_module)
        if builder_source is None:
            raise RuntimeError("SAT3D builder source cannot be resolved after import.")
        provenance = {
            "backend": "external_sat3d_swin2",
            "builder_key": "swin2",
            "builder_source": str(builder_source),
            "builder_sha256": _sha256(builder_source),
            "python_root": str(python_root),
            "source_tree_sha256": source_tree_sha256,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_load_scope": "whole_sam_strict",
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_best_loss": payload.get("best_loss"),
            "checkpoint_best_dice": payload.get("best_dice"),
            "image_encoder_class": type(image_encoder).__qualname__,
        }
        # No reference to whole_sam is registered on the wrapper, so its
        # no-grad forward and segmentation heads cannot be called accidentally.
        return cls(
            image_encoder,
            input_shape=input_shape,
            output_shape=output_shape,
            trainability=trainability,
            provenance=provenance,
        )

    @classmethod
    def from_sat3d_scratch(
        cls,
        code_root: str | Path,
        *,
        trainability: TrainabilityPolicy = "all",
        seed: int = 0,
        input_shape: Sequence[int] = SAT3D_INPUT_SHAPE,
        output_shape: Sequence[int] = SAT3D_OUTPUT_SHAPE,
        expected_source_tree_sha256: str | None = None,
    ) -> "TrainableSAT3DEncoder":
        """Instantiate the identical audited ``swin2`` image encoder randomly.

        This is a bounded initialization control.  It deliberately does not
        read a checkpoint, but it still binds the exact external source tree
        and builder identity.
        """

        python_root = _resolve_sat3d_python_root(code_root)
        builder_module, builder = _import_swin2_builder(python_root)
        source_tree_sha256 = _source_tree_sha256(python_root)
        if (
            expected_source_tree_sha256 is not None
            and source_tree_sha256.lower() != expected_source_tree_sha256.lower()
        ):
            raise RuntimeError(
                "SAT3D source-tree SHA-256 mismatch: "
                f"expected={expected_source_tree_sha256}, actual={source_tree_sha256}"
            )
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            whole_sam = builder(checkpoint=None)
        if not isinstance(whole_sam, nn.Module):
            raise TypeError("SAT3D swin2 builder did not return torch.nn.Module.")
        image_encoder = getattr(whole_sam, "image_encoder", None)
        if not isinstance(image_encoder, nn.Module):
            raise RuntimeError("SAT3D swin2 model does not expose image_encoder.")
        builder_source = _loaded_module_path(builder_module)
        if builder_source is None:
            raise RuntimeError("SAT3D builder source cannot be resolved after import.")
        return cls(
            image_encoder,
            input_shape=input_shape,
            output_shape=output_shape,
            trainability=trainability,
            provenance={
                "backend": "external_sat3d_swin2",
                "builder_key": "swin2",
                "builder_source": str(builder_source),
                "builder_sha256": _sha256(builder_source),
                "python_root": str(python_root),
                "source_tree_sha256": source_tree_sha256,
                "initialization": "deterministic_random_scratch_control",
                "seed": int(seed),
                "checkpoint_load_scope": "none_scratch_control",
                "image_encoder_class": type(image_encoder).__qualname__,
            },
        )

    @classmethod
    def tiny(
        cls,
        *,
        trainability: TrainabilityPolicy = "all",
        input_size: int = 16,
        hidden_channels: int = 8,
        output_channels: int = 16,
        output_grid: int = 2,
        seed: int = 0,
    ) -> "TrainableSAT3DEncoder":
        """Create a deterministic CPU-friendly integration-test backend."""

        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            image_encoder = TinySAT3DImageEncoder(
                input_channels=1,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                output_grid=output_grid,
            )
        source = Path(__file__).resolve()
        return cls(
            image_encoder,
            input_shape=(1, input_size, input_size, input_size),
            output_shape=(output_channels, output_grid, output_grid, output_grid),
            trainability=trainability,
            provenance={
                "backend": "tiny_3d_integration_only",
                "initialization": "deterministic_random",
                "seed": int(seed),
                "source": str(source),
                "source_sha256": _sha256(source),
                "valid_for_scientific_results": False,
            },
        )

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a defensive copy of the source/checkpoint contract."""

        return copy.deepcopy(self._provenance)

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, parameter in self.image_encoder.named_parameters()
            if parameter.requires_grad
        )

    def _discover_last_stage_module_names(self) -> tuple[str, ...]:
        layers = getattr(self.image_encoder, "layers", None)
        if not isinstance(layers, (nn.ModuleList, nn.Sequential)) or len(layers) == 0:
            if self.trainability == "last_stage":
                raise RuntimeError(
                    "last_stage policy requires image_encoder.layers to be a non-empty "
                    "ModuleList or Sequential."
                )
            return ()
        names = [f"layers.{len(layers) - 1}"]
        if isinstance(getattr(self.image_encoder, "norm", None), nn.Module):
            names.append("norm")
        return tuple(names)

    def _last_stage_modules(self) -> tuple[nn.Module, ...]:
        return tuple(
            self.image_encoder.get_submodule(name)
            for name in self._last_stage_module_names
        )

    def _configure_trainability(self) -> None:
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(self.trainability == "all")
        if self.trainability == "last_stage":
            for module in self._last_stage_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            if not any(parameter.requires_grad for parameter in self.image_encoder.parameters()):
                raise RuntimeError("last_stage policy selected no trainable parameters.")

    def train(self, mode: bool = True) -> "TrainableSAT3DEncoder":
        super().train(mode)
        if not mode:
            self.image_encoder.eval()
        elif self.trainability == "all":
            self.image_encoder.train(True)
        elif self.trainability == "last_stage":
            self.image_encoder.eval()
            for module in self._last_stage_modules():
                module.train(True)
        else:
            self.image_encoder.eval()
        return self

    def forward(self, image: Tensor) -> Tensor:
        """Call ``image_encoder`` directly while leaving autograd enabled."""

        if image.ndim != 5 or tuple(image.shape[1:]) != self.input_shape:
            raise ValueError(
                f"SAT3D input must have shape [B,{','.join(map(str, self.input_shape))}], "
                f"got {tuple(image.shape)}."
            )
        if not torch.is_floating_point(image):
            raise TypeError("SAT3D input must be a floating-point tensor.")
        parameter = next(self.image_encoder.parameters(), None)
        if parameter is not None and parameter.device != image.device:
            raise ValueError(
                f"Input is on {image.device}, but SAT3D image encoder is on {parameter.device}."
            )

        output = self.image_encoder(image)
        if not isinstance(output, Tensor):
            raise RuntimeError("SAT3D image_encoder must return one Tensor.")
        expected = (image.shape[0], *self.output_shape)
        if tuple(output.shape) != expected:
            raise RuntimeError(
                f"SAT3D image_encoder returned {tuple(output.shape)}, expected {expected}."
            )
        return output


__all__ = [
    "SAT3D_INPUT_SHAPE",
    "SAT3D_OUTPUT_SHAPE",
    "TinySAT3DImageEncoder",
    "TrainabilityPolicy",
    "TrainableSAT3DEncoder",
    "sat3d_source_tree_sha256",
]
