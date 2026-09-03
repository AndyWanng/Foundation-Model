from __future__ import annotations

import torch
from torch import nn

from mri_pet_geomc.formal.model import (
    CheckpointedEncoderStage,
    activation_checkpointing_status,
    configure_sat3d_activation_checkpointing,
)


class FakeSwinStage(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.use_checkpoint = False
        self.projection = nn.Conv3d(channels, channels, kernel_size=1)
        self.forward_calls = 0

    def forward(self, value: torch.Tensor, stage_index: int) -> torch.Tensor:
        self.forward_calls += 1
        return torch.nn.functional.silu(self.projection(value)) + float(stage_index) * 0.0


class FakeSAT3DImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeSwinStage(2), FakeSwinStage(2)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value, index)
        return value


class FakeSAT3DWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = FakeSAT3DImageEncoder()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(value)


def test_safe_checkpoint_helper_wraps_every_stage_and_recomputes_on_backward() -> None:
    encoder = FakeSAT3DWrapper().train()
    report = configure_sat3d_activation_checkpointing(encoder, enabled=True)
    assert report.enabled
    assert report.mode == "safe_non_reentrant_whole_stage"
    assert report.safe_wrapper_count == 2
    assert report.native_flags_enabled == ()
    assert all(
        isinstance(stage, CheckpointedEncoderStage)
        for stage in encoder.image_encoder.layers
    )
    value = torch.randn(1, 2, 2, 2, 2, requires_grad=True)
    encoder(value).square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(stage.stage.forward_calls >= 2 for stage in encoder.image_encoder.layers)
    status = activation_checkpointing_status(encoder)
    assert status.enabled and status.safe_wrapper_count == 2
    second = configure_sat3d_activation_checkpointing(encoder, enabled=True)
    assert second.enabled and second.stage_names == report.stage_names
