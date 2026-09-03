"""Safe activation checkpointing for the vendored SAT3D encoder.

The vendored Swin block exposes ``use_checkpoint`` but its true branch no
longer matches the modified multi-value forward signature.  Enabling that flag
directly would fail on the first training step.  This module instead wraps each
top-level encoder stage with PyTorch's non-reentrant checkpoint implementation,
without modifying vendor source or changing the stage's mathematical forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ActivationCheckpointingReport:
    enabled: bool
    mode: str
    stage_names: tuple[str, ...]
    safe_wrapper_count: int
    native_flag_module_names: tuple[str, ...]
    native_flags_enabled: tuple[str, ...]
    vendor_native_disabled_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "stage_names": list(self.stage_names),
            "safe_wrapper_count": self.safe_wrapper_count,
            "native_flag_module_names": list(self.native_flag_module_names),
            "native_flags_enabled": list(self.native_flags_enabled),
            "vendor_native_disabled_reason": self.vendor_native_disabled_reason,
        }


class CheckpointedEncoderStage(nn.Module):
    """Transparent, stateful stage wrapper using safe non-reentrant checkpointing."""

    use_checkpoint: bool = True

    def __init__(self, stage: nn.Module, *, stage_name: str) -> None:
        super().__init__()
        self.stage = stage
        self.stage_name = str(stage_name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self.training or not torch.is_grad_enabled():
            return self.stage(*args, **kwargs)

        def run_stage(*inner_args: Any) -> Any:
            return self.stage(*inner_args, **kwargs)

        return checkpoint(
            run_stage,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
        )


def _image_encoder(encoder: nn.Module) -> nn.Module:
    value = getattr(encoder, "image_encoder", None)
    return value if isinstance(value, nn.Module) else encoder


def _native_checkpoint_flags(module: nn.Module) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names: list[str] = []
    enabled: list[str] = []
    for name, child in module.named_modules():
        if isinstance(child, CheckpointedEncoderStage):
            continue
        if hasattr(child, "use_checkpoint"):
            value = getattr(child, "use_checkpoint")
            if not isinstance(value, bool):
                raise TypeError(f"{name}.use_checkpoint must be boolean")
            names.append(name or "<root>")
            if value:
                enabled.append(name or "<root>")
    return tuple(names), tuple(enabled)


def configure_sat3d_activation_checkpointing(
    encoder: nn.Module,
    *,
    enabled: bool = True,
) -> ActivationCheckpointingReport:
    """Enable safe whole-stage activation checkpointing on a loaded SAT3D encoder.

    Call this after strict pretrained-weight loading and before constructing the
    EMA bundle or loading a formal-training checkpoint.  Repeated calls are
    idempotent.  Disabling after wrapping is intentionally rejected because it
    would alter state-dict keys; build a fresh encoder for a disabled run.
    """

    image_encoder = _image_encoder(encoder)
    layers = getattr(image_encoder, "layers", None)
    if not isinstance(layers, (nn.ModuleList, nn.Sequential)) or len(layers) == 0:
        if enabled:
            raise RuntimeError(
                "Safe SAT3D activation checkpointing requires image_encoder.layers"
            )
        return ActivationCheckpointingReport(
            enabled=False,
            mode="disabled",
            stage_names=(),
            safe_wrapper_count=0,
            native_flag_module_names=(),
            native_flags_enabled=(),
            vendor_native_disabled_reason=None,
        )
    existing_wrappers = [
        (index, layer)
        for index, layer in enumerate(layers)
        if isinstance(layer, CheckpointedEncoderStage)
    ]
    if existing_wrappers and len(existing_wrappers) != len(layers):
        raise RuntimeError("SAT3D layers are only partially checkpoint-wrapped")
    if not enabled:
        if existing_wrappers:
            raise RuntimeError(
                "Cannot disable checkpointing in place after stage wrapping; rebuild encoder"
            )
        native_names, native_enabled = _native_checkpoint_flags(image_encoder)
        return ActivationCheckpointingReport(
            enabled=False,
            mode="disabled",
            stage_names=(),
            safe_wrapper_count=0,
            native_flag_module_names=native_names,
            native_flags_enabled=native_enabled,
            vendor_native_disabled_reason=None,
        )

    if not existing_wrappers:
        # The vendored true branch has a stale call signature.  Explicitly keep
        # every native flag false and use the tested outer stage wrapper.
        for child in image_encoder.modules():
            if hasattr(child, "use_checkpoint"):
                value = getattr(child, "use_checkpoint")
                if not isinstance(value, bool):
                    raise TypeError("SAT3D use_checkpoint attributes must be boolean")
                setattr(child, "use_checkpoint", False)
        for index, stage in enumerate(tuple(layers)):
            layers[index] = CheckpointedEncoderStage(
                stage, stage_name=f"image_encoder.layers.{index}"
            )

    native_names, native_enabled = _native_checkpoint_flags(image_encoder)
    wrappers = [layer for layer in layers if isinstance(layer, CheckpointedEncoderStage)]
    if len(wrappers) != len(layers) or native_enabled:
        raise RuntimeError("Safe SAT3D checkpointing postcondition failed")
    report = ActivationCheckpointingReport(
        enabled=True,
        mode="safe_non_reentrant_whole_stage",
        stage_names=tuple(wrapper.stage_name for wrapper in wrappers),
        safe_wrapper_count=len(wrappers),
        native_flag_module_names=native_names,
        native_flags_enabled=native_enabled,
        vendor_native_disabled_reason=(
            "vendored block true branch has a stale multi-value forward signature"
        ),
    )
    setattr(encoder, "_formal_activation_checkpointing_report", report)
    return report


def activation_checkpointing_status(encoder: nn.Module) -> ActivationCheckpointingReport:
    """Audit current wrappers/flags without changing the encoder."""

    stored = getattr(encoder, "_formal_activation_checkpointing_report", None)
    image_encoder = _image_encoder(encoder)
    layers = getattr(image_encoder, "layers", None)
    wrappers = (
        [layer for layer in layers if isinstance(layer, CheckpointedEncoderStage)]
        if isinstance(layers, (nn.ModuleList, nn.Sequential))
        else []
    )
    native_names, native_enabled = _native_checkpoint_flags(image_encoder)
    if isinstance(stored, ActivationCheckpointingReport):
        enabled = bool(layers is not None and len(wrappers) == len(layers) and not native_enabled)
        return ActivationCheckpointingReport(
            enabled=enabled,
            mode=stored.mode if enabled else "invalid_partial_state",
            stage_names=tuple(wrapper.stage_name for wrapper in wrappers),
            safe_wrapper_count=len(wrappers),
            native_flag_module_names=native_names,
            native_flags_enabled=native_enabled,
            vendor_native_disabled_reason=stored.vendor_native_disabled_reason,
        )
    return ActivationCheckpointingReport(
        enabled=False,
        mode="disabled",
        stage_names=(),
        safe_wrapper_count=0,
        native_flag_module_names=native_names,
        native_flags_enabled=native_enabled,
        vendor_native_disabled_reason=None,
    )


__all__ = [
    "ActivationCheckpointingReport",
    "CheckpointedEncoderStage",
    "activation_checkpointing_status",
    "configure_sat3d_activation_checkpointing",
]
