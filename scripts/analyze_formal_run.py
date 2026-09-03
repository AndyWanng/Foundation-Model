"""Reproducible audit and static visualization for a formal MRI/PET run.

The training JSONL is append-only across resumptions.  Optimizer steps that
were computed after the latest committed checkpoint can therefore appear more
than once.  This script canonicalizes by optimizer_step, keeping the last
occurrence because that is the trajectory which continued into the final
checkpoint.

Only Python's standard library plus NumPy, pandas and Pillow are required.
No checkpoint is loaded and no training artifact is modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


RELATIONS = (
    "same_observation",
    "same_session_cross_sequence",
    "same_session_repeat",
    "longitudinal_same_sequence",
    "longitudinal_same_tracer",
)

PALETTE = {
    "blue": "#2F6B9A",
    "blue_light": "#AFC9DD",
    "gold": "#D6A23B",
    "gold_light": "#F1DFC0",
    "orange": "#D97732",
    "olive": "#7B8E57",
    "pink": "#B56B82",
    "ink": "#1F2933",
    "muted": "#667481",
    "grid": "#DCE3E8",
    "paper": "#FFFFFF",
    "soft": "#F5F7F9",
    "warning": "#A65335",
}

CHART_WIDTH = 2400
CHART_HEIGHT = 1500


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _iso_duration(seconds: float) -> str:
    total = int(round(seconds))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def _finite_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _source_digest(project_root: Path) -> str:
    source_root = project_root / "src" / "mri_pet_geomc"
    rows = []
    for path in sorted(source_root.rglob("*.py")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": digest,
            }
        )
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ParsedRun:
    steps: pd.DataFrame
    abandoned_steps: pd.DataFrame
    validation: pd.DataFrame
    event_counts: Counter[str]
    level_counts: Counter[str]
    notable_events: pd.DataFrame
    held_out_test: dict[str, Any]
    first_timestamp: datetime
    last_timestamp: datetime
    line_count: int
    sequences: list[int]


def _flatten_step(event: Mapping[str, Any], invocation: int) -> dict[str, Any]:
    structured = event.get("structured") or {}
    metrics = event.get("metrics") or {}
    components = structured.get("loss_components") or {}
    modalities = structured.get("modality_losses") or {}
    modality_counts = structured.get("modality_counts") or {}
    relations = structured.get("relation_losses") or {}
    relation_counts = structured.get("relation_counts") or {}
    learning_rates = structured.get("learning_rates") or {}
    resources = structured.get("resources") or {}
    anchors = int(event["anchors_processed"])
    encoded = int(structured.get("encoded_volumes_processed", anchors))
    fallback = int(round(float(components.get("companion_fallback_count", 0.0))))
    used = int(round(float(components.get("companion_used_count", encoded - anchors))))
    requested = int(
        round(float(components.get("companion_requested_count", used + fallback)))
    )
    row: dict[str, Any] = {
        "sequence": int(event["sequence"]),
        "timestamp": str(event["timestamp"]),
        "invocation": int(invocation),
        "coverage": int(event["coverage"]),
        "global_step": int(event["global_step"]),
        "optimizer_step": int(event["optimizer_step"]),
        "next_anchor_index": int(event["next_anchor_index"]),
        "anchors_processed": anchors,
        "encoded_volumes_processed": encoded,
        "elapsed_seconds": float(event["elapsed_seconds"]),
        "loader_wait_seconds": float(structured.get("loader_wait_seconds", 0.0)),
        "throughput_anchors_s": float(structured["throughput_anchors_s"]),
        "throughput_encoded_volumes_s": float(
            structured["throughput_encoded_volumes_s"]
        ),
        "loss_total": float(structured["loss_total"]),
        "loss_mri": _finite_or_nan(modalities.get("mri")),
        "loss_pet": _finite_or_nan(modalities.get("pet")),
        "count_mri": int(modality_counts.get("mri", 0)),
        "count_pet": int(modality_counts.get("pet", 0)),
        "ema_decay": float(structured["ema_decay"]),
        "grad_norm": float(structured["grad_norm"]),
        "lr_geomc_predictor": _finite_or_nan(
            learning_rates.get("geomc_fusion_predictor")
        ),
        "lr_acquisition": _finite_or_nan(
            learning_rates.get("stems_adapters_condition_projector")
        ),
        "lr_sat3d_last_stage": _finite_or_nan(
            learning_rates.get("sat3d_stage_3")
        ),
        "lr_sat3d_patch_embed": _finite_or_nan(
            learning_rates.get("sat3d_patch_embed")
        ),
        "prediction_mean": _finite_or_nan(components.get("prediction_mean")),
        "target_mean": _finite_or_nan(components.get("target_mean")),
        "prediction_std": _finite_or_nan(components.get("prediction_std")),
        "target_std": _finite_or_nan(components.get("target_std")),
        "prediction_effective_rank": _finite_or_nan(
            components.get("prediction_effective_rank")
        ),
        "target_effective_rank": _finite_or_nan(
            components.get("target_effective_rank")
        ),
        "mean_query_count": _finite_or_nan(components.get("mean_query_count")),
        "node_anchor_coverage_mean": _finite_or_nan(
            components.get("node_anchor_coverage_mean")
        ),
        "node_companion_coverage_mean": _finite_or_nan(
            components.get("node_companion_coverage_mean")
        ),
        "node_companion_gate_mean_supported": _finite_or_nan(
            components.get("node_companion_gate_mean_supported")
        ),
        "node_coverage_mean": _finite_or_nan(components.get("node_coverage_mean")),
        "node_feature_rms": _finite_or_nan(components.get("node_node_feature_rms")),
        "visible_normalization_median": _finite_or_nan(
            components.get("visible_normalization_median")
        ),
        "visible_normalization_scale": _finite_or_nan(
            components.get("visible_normalization_scale")
        ),
        "companion_requested_count": requested,
        "companion_used_count": used,
        "companion_fallback_count": fallback,
        "cpu_rss_gib": _finite_or_nan(resources.get("cpu_rss_gib")),
        "gpu_allocated_gib": _finite_or_nan(resources.get("gpu_allocated_gib")),
        "gpu_reserved_gib": _finite_or_nan(resources.get("gpu_reserved_gib")),
        "gpu_peak_allocated_gib": _finite_or_nan(
            resources.get("gpu_peak_allocated_gib")
        ),
        "gpu_peak_reserved_gib": _finite_or_nan(
            resources.get("gpu_peak_reserved_gib")
        ),
    }
    for relation in RELATIONS:
        row[f"loss_relation_{relation}"] = _finite_or_nan(relations.get(relation))
        row[f"count_relation_{relation}"] = int(relation_counts.get(relation, 0))
    return row


def _flatten_validation(event: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(event.get("metrics") or {})
    row: dict[str, Any] = {
        "timestamp": str(event["timestamp"]),
        "sequence": int(event["sequence"]),
        "coverage": int(event["coverage"]),
        "optimizer_step": int(event["optimizer_step"]),
        "tier": str(event["tier"]),
        "reason": str(event.get("reason", "")),
    }
    for key, value in metrics.items():
        if isinstance(value, (str, bool)):
            row[key] = value
        elif value is not None:
            row[key] = value
    return row


def parse_run(jsonl_path: Path) -> ParsedRun:
    steps: dict[int, dict[str, Any]] = {}
    abandoned: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    notable: list[dict[str, Any]] = []
    held_out_test: dict[str, Any] = {}
    sequences: list[int] = []
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    invocation = 0
    notable_names = {
        "formal_training_workflow_started",
        "formal_training_workflow_fatal",
        "resume_source_compatibility_activated",
        "checkpoint_loaded",
        "gpu_reserved_memory_soft_limit",
        "optional_companion_visible_support_empty",
        "observational_validation_repeat_after_resume",
        "formal_training_completed",
        "held_out_test_completed",
        "formal_training_workflow_completed",
    }
    line_count = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_count, line in enumerate(handle, start=1):
            event = json.loads(line)
            event_name = str(event.get("event", ""))
            level = str(event.get("level", ""))
            event_counts[event_name] += 1
            level_counts[level] += 1
            sequence = int(event["sequence"])
            sequences.append(sequence)
            timestamp = _parse_timestamp(str(event["timestamp"]))
            first_timestamp = timestamp if first_timestamp is None else first_timestamp
            last_timestamp = timestamp
            if event_name == "formal_training_workflow_started":
                invocation += 1
            if event_name == "training_optimizer_step":
                row = _flatten_step(event, invocation)
                step = int(row["optimizer_step"])
                if step in steps:
                    abandoned.append(steps[step])
                steps[step] = row
            elif event_name == "validation_completed":
                validations.append(_flatten_validation(event))
            elif event_name == "held_out_test_completed":
                held_out_test = dict(event.get("metrics") or {})
            if event_name in notable_names:
                details: dict[str, Any] = {
                    "sequence": sequence,
                    "timestamp": str(event["timestamp"]),
                    "invocation": invocation,
                    "level": level,
                    "event": event_name,
                    "message": str(event.get("message", "")),
                    "coverage": event.get("coverage"),
                    "optimizer_step": event.get("optimizer_step"),
                }
                if event_name == "checkpoint_loaded":
                    cursor = event.get("cursor") or {}
                    details["coverage"] = cursor.get("coverage")
                    details["optimizer_step"] = cursor.get("optimizer_step")
                    details["next_anchor_index"] = cursor.get("next_anchor_index")
                if event_name == "gpu_reserved_memory_soft_limit":
                    details["gpu_reserved_gib"] = event.get("gpu_reserved_gib")
                    details["soft_limit_gib"] = event.get("soft_limit_gib")
                if event_name == "optional_companion_visible_support_empty":
                    details["fallback_count"] = event.get("fallback_count")
                notable.append(details)
    if first_timestamp is None or last_timestamp is None:
        raise RuntimeError(f"No events found in {jsonl_path}")
    return ParsedRun(
        steps=pd.DataFrame(sorted(steps.values(), key=lambda row: row["optimizer_step"])),
        abandoned_steps=pd.DataFrame(abandoned),
        validation=pd.DataFrame(validations),
        event_counts=event_counts,
        level_counts=level_counts,
        notable_events=pd.DataFrame(notable),
        held_out_test=held_out_test,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        line_count=line_count,
        sequences=sequences,
    )


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask].to_numpy(float), weights=weights[mask].to_numpy(float)))


def aggregate_coverages(steps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for coverage, group in steps.groupby("coverage", sort=True):
        anchors = int(group["anchors_processed"].sum())
        elapsed = float(group["elapsed_seconds"].sum())
        loader_wait = float(group["loader_wait_seconds"].sum())
        count_mri = int(group["count_mri"].sum())
        count_pet = int(group["count_pet"].sum())
        loss_mri = _weighted_mean(group["loss_mri"], group["count_mri"])
        loss_pet = _weighted_mean(group["loss_pet"], group["count_pet"])
        used = int(group["companion_used_count"].sum())
        requested = int(group["companion_requested_count"].sum())
        fallback = int(group["companion_fallback_count"].sum())
        gate = _weighted_mean(
            group["node_companion_gate_mean_supported"],
            group["companion_used_count"],
        )
        if not math.isfinite(gate):
            gate = 0.0
        if coverage <= 2:
            alpha = 0.0
        elif coverage <= 5:
            alpha = (int(coverage) - 2) / 4.0
        else:
            alpha = 1.0
        row: dict[str, Any] = {
            "coverage": int(coverage),
            "optimizer_step_start": int(group["optimizer_step"].min()),
            "optimizer_step_end": int(group["optimizer_step"].max()),
            "optimizer_steps": int(len(group)),
            "anchors": anchors,
            "encoded_volumes": int(group["encoded_volumes_processed"].sum()),
            "train_loss_anchor_mean": _weighted_mean(
                group["loss_total"], group["anchors_processed"]
            ),
            "train_loss_step_p10": float(group["loss_total"].quantile(0.10)),
            "train_loss_step_median": float(group["loss_total"].median()),
            "train_loss_step_p90": float(group["loss_total"].quantile(0.90)),
            "train_loss_mri": loss_mri,
            "train_loss_pet": loss_pet,
            "train_loss_modality_macro": 0.5 * (loss_mri + loss_pet),
            "count_mri": count_mri,
            "count_pet": count_pet,
            "prediction_std": _weighted_mean(
                group["prediction_std"], group["anchors_processed"]
            ),
            "target_std": _weighted_mean(
                group["target_std"], group["anchors_processed"]
            ),
            "prediction_effective_rank": _weighted_mean(
                group["prediction_effective_rank"], group["anchors_processed"]
            ),
            "target_effective_rank": _weighted_mean(
                group["target_effective_rank"], group["anchors_processed"]
            ),
            "mean_query_count": _weighted_mean(
                group["mean_query_count"], group["anchors_processed"]
            ),
            "grad_norm_median": float(group["grad_norm"].median()),
            "grad_norm_p95": float(group["grad_norm"].quantile(0.95)),
            "grad_norm_max": float(group["grad_norm"].max()),
            "grad_clip_fraction": float((group["grad_norm"] > 1.0).mean()),
            "throughput_anchors_s": anchors / elapsed,
            "throughput_encoded_volumes_s": float(
                group["encoded_volumes_processed"].sum()
            )
            / elapsed,
            "loader_wait_fraction": loader_wait / elapsed,
            "cpu_rss_gib_max": float(group["cpu_rss_gib"].max()),
            "gpu_peak_allocated_gib_max": float(
                group["gpu_peak_allocated_gib"].max()
            ),
            "gpu_peak_reserved_gib_max": float(
                group["gpu_peak_reserved_gib"].max()
            ),
            "companion_requested": requested,
            "companion_used": used,
            "companion_fallback": fallback,
            "companion_requested_rate": requested / anchors,
            "companion_used_rate": used / anchors,
            "companion_fallback_rate_requested": fallback / max(requested, 1),
            "node_companion_coverage_mean": _weighted_mean(
                group["node_companion_coverage_mean"], group["anchors_processed"]
            ),
            "node_companion_gate_mean_supported": gate,
            "node_anchor_coverage_mean": _weighted_mean(
                group["node_anchor_coverage_mean"], group["anchors_processed"]
            ),
            "lr_geomc_predictor_mean": float(group["lr_geomc_predictor"].mean()),
            "ema_decay_mean": float(group["ema_decay"].mean()),
            "target_self_rate": 1.0 - 0.5 * alpha,
            "target_cross_sequence_rate": 0.30 * alpha,
            "target_longitudinal_rate": 0.15 * alpha,
            "target_repeat_rate": 0.05 * alpha,
        }
        for relation in RELATIONS:
            count = int(group[f"count_relation_{relation}"].sum())
            row[f"count_relation_{relation}"] = count
            row[f"rate_relation_{relation}"] = count / anchors
            row[f"train_loss_relation_{relation}"] = _weighted_mean(
                group[f"loss_relation_{relation}"],
                group[f"count_relation_{relation}"],
            )
        row["rate_relation_longitudinal_combined"] = (
            row["rate_relation_longitudinal_same_sequence"]
            + row["rate_relation_longitudinal_same_tracer"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _scheduler_multiplier(step_index: np.ndarray, warmup_updates: int, total_updates: int) -> np.ndarray:
    step_index = np.asarray(step_index, dtype=float)
    warm = int(warmup_updates)
    result = np.empty_like(step_index, dtype=float)
    warm_mask = step_index < warm
    result[warm_mask] = (step_index[warm_mask] + 1.0) / float(warm)
    progress = (step_index[~warm_mask] - warm) / float(max(total_updates - warm, 1))
    progress = np.clip(progress, 0.0, 1.0)
    result[~warm_mask] = 0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * progress))
    return result


def add_scheduler_diagnostics(
    steps: pd.DataFrame,
    *,
    base_lr: float,
    warmup_updates: int,
    total_updates: int,
) -> pd.DataFrame:
    result = steps.copy()
    intended = _scheduler_multiplier(
        result["optimizer_step"].to_numpy(int) - 1,
        warmup_updates,
        total_updates,
    )
    result["lr_multiplier_actual"] = result["lr_geomc_predictor"] / float(base_lr)
    result["lr_multiplier_intended"] = intended
    result["lr_multiplier_delta"] = (
        result["lr_multiplier_actual"] - result["lr_multiplier_intended"]
    )
    return result


def validate_canonical_run(
    parsed: ParsedRun,
    coverage_summary: pd.DataFrame,
    run_contract: Mapping[str, Any],
    text_log_path: Path,
) -> dict[str, Any]:
    expected_steps = int(run_contract["total_optimizer_steps"])
    expected_coverages = int(run_contract["fixed_coverages"])
    expected_anchors = int(run_contract["observations"]["train"])
    actual_steps = parsed.steps["optimizer_step"].to_list()
    checks: dict[str, Any] = {
        "jsonl_lines": parsed.line_count,
        "text_log_lines": sum(1 for _ in text_log_path.open("rb")),
        "jsonl_text_line_parity": False,
        "sequence_is_exact_1_to_n": parsed.sequences == list(range(1, parsed.line_count + 1)),
        "unique_optimizer_steps": int(len(parsed.steps)),
        "optimizer_steps_exact_1_to_horizon": actual_steps == list(range(1, expected_steps + 1)),
        "discarded_replayed_step_events": int(len(parsed.abandoned_steps)),
        "discarded_replay_step_min": (
            int(parsed.abandoned_steps["optimizer_step"].min())
            if not parsed.abandoned_steps.empty
            else None
        ),
        "discarded_replay_step_max": (
            int(parsed.abandoned_steps["optimizer_step"].max())
            if not parsed.abandoned_steps.empty
            else None
        ),
        "coverage_count": int(len(coverage_summary)),
        "coverage_count_matches_contract": len(coverage_summary) == expected_coverages,
        "steps_per_coverage_unique": sorted(
            int(value) for value in coverage_summary["optimizer_steps"].unique()
        ),
        "anchors_per_coverage_unique": sorted(
            int(value) for value in coverage_summary["anchors"].unique()
        ),
        "coverage_steps_match_contract": bool(
            (coverage_summary["optimizer_steps"] == int(run_contract["steps_per_coverage"])).all()
        ),
        "coverage_anchors_match_contract": bool(
            (coverage_summary["anchors"] == expected_anchors).all()
        ),
        "modality_counts_sum_to_anchors": bool(
            (
                parsed.steps["count_mri"]
                + parsed.steps["count_pet"]
                == parsed.steps["anchors_processed"]
            ).all()
        ),
        "relation_counts_sum_to_anchors": bool(
            (
                parsed.steps[[f"count_relation_{name}" for name in RELATIONS]].sum(axis=1)
                == parsed.steps["anchors_processed"]
            ).all()
        ),
        "nonfinite_core_step_values": int(
            (~np.isfinite(
                parsed.steps[
                    [
                        "loss_total",
                        "grad_norm",
                        "prediction_std",
                        "target_std",
                        "prediction_effective_rank",
                        "target_effective_rank",
                        "throughput_anchors_s",
                        "gpu_peak_reserved_gib",
                    ]
                ].to_numpy(float)
            )).sum()
        ),
    }
    checks["jsonl_text_line_parity"] = checks["jsonl_lines"] == checks["text_log_lines"]
    return checks


def _font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        path = Path(r"C:\Windows\Fonts\consola.ttf")
    else:
        path = Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = _font(50, bold=True)
FONT_SUBTITLE = _font(27)
FONT_AXIS = _font(25)
FONT_TICK = _font(22, mono=True)
FONT_LEGEND = _font(23)
FONT_NOTE = _font(20)
FONT_ANNOTATION = _font(20)


def _nice_number(value: float, *, rounding: bool) -> float:
    exponent = math.floor(math.log10(value)) if value > 0 else 0
    fraction = value / (10**exponent)
    if rounding:
        nice = 1 if fraction < 1.5 else 2 if fraction < 3 else 5 if fraction < 7 else 10
    else:
        nice = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return nice * (10**exponent)


def _linear_ticks(low: float, high: float, count: int = 6) -> list[float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return [0.0, 1.0]
    if math.isclose(low, high):
        pad = abs(low) * 0.1 or 1.0
        low -= pad
        high += pad
    spacing = _nice_number((high - low) / max(count - 1, 1), rounding=True)
    start = math.floor(low / spacing) * spacing
    end = math.ceil(high / spacing) * spacing
    ticks = []
    current = start
    while current <= end + spacing * 0.5 and len(ticks) < 20:
        ticks.append(float(current))
        current += spacing
    return ticks


def _log_ticks(low: float, high: float) -> list[float]:
    low = max(low, 1e-12)
    powers = range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
    ticks = []
    for power in powers:
        for multiplier in (1.0, 2.0, 5.0):
            value = multiplier * 10**power
            if low <= value <= high:
                ticks.append(value)
    return ticks


def _format_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}k"
    if absolute != 0 and (absolute < 0.001 or absolute >= 100):
        return f"{value:.1e}"
    if absolute < 0.01:
        return f"{value:.4f}"
    if absolute < 1:
        return f"{value:.3f}"
    return f"{value:.1f}"


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 2600) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) <= max_points:
        return x, y
    bins = np.linspace(0, len(x), max_points // 2 + 1, dtype=int)
    out_x: list[float] = []
    out_y: list[float] = []
    for left, right in zip(bins[:-1], bins[1:]):
        if right <= left:
            continue
        current = y[left:right]
        for offset in sorted({int(np.argmin(current)), int(np.argmax(current))}):
            out_x.append(float(x[left + offset]))
            out_y.append(float(current[offset]))
    order = np.argsort(out_x)
    return np.asarray(out_x)[order], np.asarray(out_y)[order]


def line_chart(
    output: Path,
    *,
    title: str,
    subtitle: str,
    series: Sequence[Mapping[str, Any]],
    x_label: str,
    y_label: str,
    y_log: bool = False,
    y_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] | None = None,
    band: Mapping[str, Any] | None = None,
    verticals: Sequence[Mapping[str, Any]] = (),
    horizontals: Sequence[Mapping[str, Any]] = (),
    note: str = "Source: train.metrics.jsonl (canonical last occurrence per optimizer_step).",
) -> None:
    image = Image.new("RGB", (CHART_WIDTH, CHART_HEIGHT), PALETTE["paper"])
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 210, CHART_WIDTH - 90, 225, CHART_HEIGHT - 180
    draw.text((left, 55), title, font=FONT_TITLE, fill=PALETTE["ink"])
    draw.text((left, 125), subtitle, font=FONT_SUBTITLE, fill=PALETTE["muted"])

    all_x = np.concatenate([np.asarray(item["x"], dtype=float) for item in series])
    all_y = np.concatenate([np.asarray(item["y"], dtype=float) for item in series])
    valid_x = all_x[np.isfinite(all_x)]
    valid_y = all_y[np.isfinite(all_y) & ((all_y > 0) if y_log else True)]
    if x_limits is None:
        x_low, x_high = float(valid_x.min()), float(valid_x.max())
    else:
        x_low, x_high = map(float, x_limits)
    if y_limits is None:
        y_low, y_high = float(valid_y.min()), float(valid_y.max())
        if y_log:
            y_low *= 0.85
            y_high *= 1.15
        else:
            pad = max((y_high - y_low) * 0.08, abs(y_high) * 0.01, 1e-12)
            y_low -= pad
            y_high += pad
    else:
        y_low, y_high = map(float, y_limits)
    if math.isclose(x_low, x_high):
        x_high = x_low + 1.0

    def map_x(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * (right - left)

    if y_log:
        log_low, log_high = math.log10(y_low), math.log10(y_high)

        def map_y(value: float) -> float:
            return bottom - (math.log10(max(value, 1e-30)) - log_low) / (log_high - log_low) * (bottom - top)

        y_ticks = _log_ticks(y_low, y_high)
    else:

        def map_y(value: float) -> float:
            return bottom - (value - y_low) / (y_high - y_low) * (bottom - top)

        y_ticks = [value for value in _linear_ticks(y_low, y_high) if y_low <= value <= y_high]

    x_ticks = [value for value in _linear_ticks(x_low, x_high, 7) if x_low <= value <= x_high]
    for value in y_ticks:
        y = map_y(value)
        draw.line((left, y, right, y), fill=PALETTE["grid"], width=2)
        label = _format_number(value)
        width, height = _text_size(draw, label, FONT_TICK)
        draw.text((left - width - 20, y - height / 2), label, font=FONT_TICK, fill=PALETTE["muted"])
    for value in x_ticks:
        x = map_x(value)
        draw.line((x, bottom, x, bottom + 12), fill=PALETTE["ink"], width=2)
        label = _format_number(value)
        width, _ = _text_size(draw, label, FONT_TICK)
        draw.text((x - width / 2, bottom + 22), label, font=FONT_TICK, fill=PALETTE["muted"])
    draw.line((left, top, left, bottom), fill=PALETTE["ink"], width=3)
    draw.line((left, bottom, right, bottom), fill=PALETTE["ink"], width=3)

    if band is not None:
        bx = np.asarray(band["x"], dtype=float)
        lower = np.asarray(band["lower"], dtype=float)
        upper = np.asarray(band["upper"], dtype=float)
        mask = np.isfinite(bx) & np.isfinite(lower) & np.isfinite(upper)
        polygon = [(map_x(float(x)), map_y(float(y))) for x, y in zip(bx[mask], upper[mask])]
        polygon += [
            (map_x(float(x)), map_y(float(y)))
            for x, y in zip(bx[mask][::-1], lower[mask][::-1])
        ]
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.polygon(polygon, fill=band.get("fill", "#AFC9DD55"))
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)

    for item in verticals:
        value = float(item["x"])
        if not x_low <= value <= x_high:
            continue
        x = map_x(value)
        draw.line((x, top, x, bottom), fill=item.get("color", PALETTE["warning"]), width=3)
        label = str(item.get("label", ""))
        if label:
            draw.text((x + 8, top + 10), label, font=FONT_ANNOTATION, fill=item.get("color", PALETTE["warning"]))
    for item in horizontals:
        value = float(item["y"])
        if not y_low <= value <= y_high:
            continue
        y = map_y(value)
        draw.line((left, y, right, y), fill=item.get("color", PALETTE["warning"]), width=3)
        label = str(item.get("label", ""))
        if label:
            draw.text((right - 330, y - 32), label, font=FONT_ANNOTATION, fill=item.get("color", PALETTE["warning"]))

    for item in series:
        x = np.asarray(item["x"], dtype=float)
        y = np.asarray(item["y"], dtype=float)
        if y_log:
            y = np.where(y > 0, y, np.nan)
        x, y = _downsample_xy(x, y)
        points = [(map_x(float(a)), map_y(float(b))) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
        if len(points) >= 2:
            draw.line(points, fill=item["color"], width=int(item.get("width", 5)), joint="curve")
        marker = int(item.get("marker", 0))
        if marker and len(points) <= 100:
            for px, py in points:
                draw.ellipse((px - marker, py - marker, px + marker, py + marker), fill=item["color"], outline=PALETTE["paper"], width=2)

    legend_x = left
    legend_y = 182
    for item in series:
        draw.line((legend_x, legend_y, legend_x + 52, legend_y), fill=item["color"], width=7)
        label = str(item["name"])
        draw.text((legend_x + 66, legend_y - 16), label, font=FONT_LEGEND, fill=PALETTE["ink"])
        width, _ = _text_size(draw, label, FONT_LEGEND)
        legend_x += width + 130

    x_width, _ = _text_size(draw, x_label, FONT_AXIS)
    draw.text(((left + right - x_width) / 2, CHART_HEIGHT - 100), x_label, font=FONT_AXIS, fill=PALETTE["ink"])
    label_width, label_height = _text_size(draw, y_label, FONT_AXIS)
    label_canvas = Image.new(
        "RGBA", (label_width + 24, label_height + 24), (0, 0, 0, 0)
    )
    rdraw = ImageDraw.Draw(label_canvas)
    rdraw.text((12, 12), y_label, font=FONT_AXIS, fill=PALETTE["ink"])
    rotated = label_canvas.rotate(90, expand=True)
    image.paste(rotated, (35, int((top + bottom - rotated.height) / 2)), rotated)
    draw = ImageDraw.Draw(image)
    draw.text((left, CHART_HEIGHT - 45), note, font=FONT_NOTE, fill=PALETTE["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def stacked_relation_chart(output: Path, coverage: pd.DataFrame) -> None:
    image = Image.new("RGB", (CHART_WIDTH, CHART_HEIGHT), PALETTE["paper"])
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 210, CHART_WIDTH - 90, 270, CHART_HEIGHT - 180
    draw.text((left, 55), "Realized relation composition by coverage", font=FONT_TITLE, fill=PALETTE["ink"])
    draw.text(
        (left, 125),
        "Shares are weighted by training anchors; the dark line is the scheduled total companion share",
        font=FONT_SUBTITLE,
        fill=PALETTE["muted"],
    )
    colors = {
        "same_observation": PALETTE["blue_light"],
        "same_session_cross_sequence": PALETTE["blue"],
        "longitudinal_same_sequence": PALETTE["gold"],
        "longitudinal_same_tracer": PALETTE["orange"],
        "same_session_repeat": PALETTE["pink"],
    }
    for fraction in np.linspace(0, 1, 6):
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill=PALETTE["grid"], width=2)
        label = f"{fraction:.0%}"
        width, height = _text_size(draw, label, FONT_TICK)
        draw.text((left - width - 20, y - height / 2), label, font=FONT_TICK, fill=PALETTE["muted"])
    bar_width = (right - left) / len(coverage)
    for index, row in coverage.reset_index(drop=True).iterrows():
        x0 = left + index * bar_width + 1
        x1 = left + (index + 1) * bar_width - 1
        cumulative = 0.0
        for relation in RELATIONS:
            value = float(row[f"rate_relation_{relation}"])
            y1 = bottom - cumulative * (bottom - top)
            cumulative += value
            y0 = bottom - cumulative * (bottom - top)
            draw.rectangle((x0, y0, x1, y1), fill=colors[relation])
    target_x = []
    target_y = []
    for _, row in coverage.iterrows():
        target = 1.0 - float(row["target_self_rate"])
        target_x.append(left + (float(row["coverage"]) - 0.5) / len(coverage) * (right - left))
        target_y.append(bottom - target * (bottom - top))
    draw.line(list(zip(target_x, target_y)), fill=PALETTE["ink"], width=5)
    for coverage_tick in (1, 5, 10, 20, 30, 40, 50):
        x = left + (coverage_tick - 0.5) / len(coverage) * (right - left)
        draw.line((x, bottom, x, bottom + 12), fill=PALETTE["ink"], width=2)
        label = str(coverage_tick)
        width, _ = _text_size(draw, label, FONT_TICK)
        draw.text((x - width / 2, bottom + 22), label, font=FONT_TICK, fill=PALETTE["muted"])
    draw.line((left, top, left, bottom), fill=PALETTE["ink"], width=3)
    draw.line((left, bottom, right, bottom), fill=PALETTE["ink"], width=3)
    legend_items = [
        ("self", colors["same_observation"]),
        ("cross-sequence", colors["same_session_cross_sequence"]),
        ("longitudinal MRI", colors["longitudinal_same_sequence"]),
        ("longitudinal PET", colors["longitudinal_same_tracer"]),
        ("repeat", colors["same_session_repeat"]),
        ("scheduled companion share", PALETTE["ink"]),
    ]
    x = left
    legend_y = 174
    for index, (label, color) in enumerate(legend_items):
        if index == 3:
            x = left
            legend_y = 214
        draw.rectangle((x, legend_y + 6, x + 34, legend_y + 30), fill=color)
        draw.text((x + 46, legend_y), label, font=FONT_LEGEND, fill=PALETTE["ink"])
        width, _ = _text_size(draw, label, FONT_LEGEND)
        x += width + 105
    x_label = "Coverage"
    width, _ = _text_size(draw, x_label, FONT_AXIS)
    draw.text(((left + right - width) / 2, CHART_HEIGHT - 100), x_label, font=FONT_AXIS, fill=PALETTE["ink"])
    draw.text((left, CHART_HEIGHT - 45), "Source: canonical training_optimizer_step relation_counts.", font=FONT_NOTE, fill=PALETTE["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def validation_test_dot_plot(output: Path, validation: pd.DataFrame, test: Mapping[str, Any]) -> None:
    full = validation[validation["tier"] == "full_validation"].sort_values("coverage").iloc[-1]
    categories = ["Fixed MRI/PET macro", "MRI subject mean", "PET subject mean"]
    val_values = [float(full["loss_total"]), float(full["loss_modality_mri"]), float(full["loss_modality_pet"])]
    test_values = [float(test["loss_total"]), float(test["loss_modality_mri"]), float(test["loss_modality_pet"])]
    low = min(val_values + test_values) * 0.96
    high = max(val_values + test_values) * 1.04
    image = Image.new("RGB", (CHART_WIDTH, CHART_HEIGHT), PALETTE["paper"])
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 520, CHART_WIDTH - 120, 300, CHART_HEIGHT - 220
    draw.text(
        (180, 55),
        "Final full validation versus one-time held-out test",
        font=FONT_TITLE,
        fill=PALETTE["ink"],
    )
    draw.text(
        (180, 125),
        "Connected dots use a focused x-axis with exact loss labels; this is not a zero-based magnitude bar chart",
        font=FONT_SUBTITLE,
        fill=PALETTE["muted"],
    )
    for tick in _linear_ticks(low, high, 7):
        if not low <= tick <= high:
            continue
        x = left + (tick - low) / (high - low) * (right - left)
        draw.line((x, top, x, bottom), fill=PALETTE["grid"], width=2)
        label = f"{tick:.5f}"
        width, _ = _text_size(draw, label, FONT_TICK)
        draw.text((x - width / 2, bottom + 25), label, font=FONT_TICK, fill=PALETTE["muted"])
    row_gap = (bottom - top) / len(categories)
    for index, category in enumerate(categories):
        y = top + row_gap * (index + 0.5)
        label_width, label_height = _text_size(draw, category, FONT_AXIS)
        draw.text((left - label_width - 35, y - label_height / 2), category, font=FONT_AXIS, fill=PALETTE["ink"])
        xv = left + (val_values[index] - low) / (high - low) * (right - left)
        xt = left + (test_values[index] - low) / (high - low) * (right - left)
        draw.line((xv, y, xt, y), fill=PALETTE["muted"], width=4)
        draw.ellipse((xv - 13, y - 13, xv + 13, y + 13), fill=PALETTE["blue"], outline=PALETTE["paper"], width=3)
        draw.ellipse((xt - 13, y - 13, xt + 13, y + 13), fill=PALETTE["gold"], outline=PALETTE["paper"], width=3)
        draw.text((xv - 80, y - 58), f"{val_values[index]:.6f}", font=FONT_TICK, fill=PALETTE["blue"])
        draw.text((xt - 80, y + 26), f"{test_values[index]:.6f}", font=FONT_TICK, fill=PALETTE["gold"])
    draw.line((left, bottom, right, bottom), fill=PALETTE["ink"], width=3)
    draw.ellipse((180, 190, 204, 214), fill=PALETTE["blue"])
    draw.text((220, 184), "Full validation @ coverage 50", font=FONT_LEGEND, fill=PALETTE["ink"])
    draw.ellipse((650, 190, 674, 214), fill=PALETTE["gold"])
    draw.text((690, 184), "Held-out test", font=FONT_LEGEND, fill=PALETTE["ink"])
    draw.text((left, CHART_HEIGHT - 55), "Aggregation: subject first, then fixed MRI/PET 0.5/0.5 macro.", font=FONT_NOTE, fill=PALETTE["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def write_english_figure_guide(output_dir: Path) -> None:
    guide = """
ENGLISH FIGURE GUIDE / 英文图逐图阅读说明
============================================================

适用目录：figures_en
适用 run：formal-24k-c9060f1c6a2fba59

重要总原则
------------------------------------------------------------
1. Loss 越低通常表示当前预训练目标完成得越好，但不等于下游临床任务一定更好。
2. Train、fixed monitor、full validation 和 held-out test 的聚合口径不同，不能只比较线条高低而忽略定义。
3. “通常较好”表示常见诊断意义，不是硬阈值。最终判断需要 matched baseline、多个 seed、per-subject 结果和置信区间。
4. 图中的 canonical trajectory 对重复 optimizer_step 保留最后一次执行；fatal 后被重跑的旧事件没有重复计入。
5. 本地没有 checkpoint，因此图只能解释已保存日志与 aggregate，不能证明最终权重可加载或独立复现。

常用英文总词汇
------------------------------------------------------------
Canonical：规范化后的唯一有效轨迹。本报告对每个 optimizer step 保留最后一次记录。
Coverage：一次完整训练覆盖；本 run 中每个 coverage 让 19,167 个 train observation 各出现一次。
Optimizer step：完成 gradient accumulation 后真正执行一次参数更新的步数。
Anchor：当前被预测自身 EMA latent 的主要 observation。
Companion：如果存在合法关系，可额外提供给模型的上下文 observation；它不是监督 target。
Observation：一条影像观测，例如某 subject 的一次 MRI sequence 或 PET tracer acquisition。
Subject mean：先在每个 subject 内聚合，再跨 subject 求平均。
Macro：对不同组给予固定相同权重，而不是按 observation 数量加权。
EMA：Exponential Moving Average，指数滑动平均 teacher/target 网络。
Latent：模型内部表示向量，而不是原始体素。
Held-out：训练和 validation 都不使用、只在最后评估一次的测试划分。
Log scale：对数轴；相同的垂直距离代表相同倍数变化，不代表相同绝对差值。
P10/P50/P95/P90：第 10/50/95/90 百分位数。
GiB：2^30 bytes 的显存单位，不等同于十进制 GB。


[01] 01_optimizer_loss_curve.png
标题：Canonical optimizer-step training loss
------------------------------------------------------------
图的目的：查看每个 committed optimizer step 的训练目标是否下降、是否发散，以及 resume 前后是否产生明显断层。

英文名词：
- Training loss：训练目标值；本项目是 masked-Huber latent prediction loss。
- Masked-Huber loss：只在被 mask/query 的 latent 位置计算 Huber loss；小误差近似平方损失，大误差近似绝对值损失。
- Individual step：每个 optimizer step 的原始 loss，浅蓝线。
- 250-step rolling median：以约 250 个 step 的滑动窗口取中位数，深蓝线；比单 step 更能显示长期趋势。
- Coverage 7 resume / patch：coverage 7 从已提交 checkpoint 恢复并启用兼容性修复的位置。
- Coverage 43 validation resume：coverage 43 只恢复 validation、没有重放 optimizer step 的位置。
- Superseded replay events excluded：被后一次执行替代的 105 条旧 step 事件已排除。

怎么看：
- 先看深蓝 rolling median 的整体方向，再看浅蓝单 step 的离散程度。
- 对数纵轴适合同时展示早期高 loss 与后期低 loss；例如 0.02 到 0.01 与 0.01 到 0.005 都是减半。
- 查看两条 resume 竖线附近是否出现永久性跳升、持续振荡或 non-finite 中断。

通常较好：rolling median 平稳下降并进入稳定平台；单步噪声有界；resume 后迅速接回原趋势。
警报：持续上升、周期性爆炸、突然抬升后不恢复、出现 NaN/Inf，或者 loss 过早变成近零而 representation 指标同时 collapse。
本 run：早期快速下降，之后进入低 loss 长平台；两个 resume 点没有造成永久发散。该图支持“优化过程稳定”，不支持“表示具备临床价值”。


[02] 02_train_validation_loss_by_coverage.png
标题：Train and observational validation loss by coverage
------------------------------------------------------------
图的目的：比较训练 objective 与两种保存的 observational validation 是否共同改善。

英文名词：
- Train modality macro：先分别计算 MRI 与 PET 的 train mean，再以 0.5/0.5 合并。
- Fixed monitor：每个 coverage 评估的 512-anchor stratified monitor；selector 包含 coverage，因此不是固定纵向样本面板。
- Full validation：每 5 个 coverage 对全部 2,398 个 cached validation observation 做一次评估。
- Subject-first fixed MRI/PET macro：先按 subject 聚合，再分别得到 MRI/PET mean，最后固定 0.5/0.5 合并。
- Observational validation：validation 只用于观察，不做 early stopping、best-checkpoint selection 或训练决策。
- Shaded band / P10-P90：每个 coverage 内单 step train loss 的第 10 到第 90 百分位范围。

怎么看：
- 深蓝 train line 看整体训练趋势；橙色 full-validation 点最适合看保存的泛化趋势；金色 monitor 更频繁但更噪。
- 阴影越窄表示同一 coverage 内 step loss 更集中，但宽窄还受 batch/relation composition 影响。
- 不能把 train 与 validation 的绝对差简单当作传统 supervised generalization gap，因为二者聚合口径不同。

通常较好：full validation 随训练总体下降；后期没有持续反弹；train 继续下降时 validation 不明显恶化。
警报：train 持续下降但 full validation 连续升高；monitor 与 full validation 长期方向相反；validation 出现突然不可恢复跳变。
本 run：full validation 从 coverage 5 的 0.007778 降到 coverage 50 的 0.005799，最终值是保存的最低 full-validation 值。coverage 25-30 有小幅回升但随后恢复。monitor 的最低点不能视为固定样本上的最佳点。


[03] 03_modality_loss_by_coverage.png
标题：MRI and PET loss by coverage
------------------------------------------------------------
图的目的：查看 MRI 与 PET 是否同时改善，以及固定 macro 是否掩盖模态差异。

英文名词：
- MRI：Magnetic Resonance Imaging，磁共振影像。
- PET：Positron Emission Tomography，正电子发射断层影像。
- Observation-weighted train mean：按每个 modality 中的 observation 数量加权的训练均值。
- Full-validation subject mean：先按 subject 聚合得到的完整 validation 模态均值。
- Solid lines：连续的 train modality 曲线。
- Markers：每 5 coverage 保存一次的 full-validation 点。

怎么看：
- 同色粗线是 train，同色带圆点细线是 full validation。
- 看每个 modality 自己是否随 coverage 下降，比直接比较 MRI 与 PET 谁更低更重要。
- MRI/PET loss 的绝对高低可能受影像统计、mask 难度与 target variance 影响，不能直接解释为哪个 modality 的语义更容易。

通常较好：MRI 与 PET 都有持续下降；full-validation 点与各自 train 趋势大体一致；不存在一个 modality 改善而另一个长期恶化。
警报：只有一个 modality 改善；固定 macro 看似稳定但两种 modality 反向移动；某模态 validation 与 train 明显分叉。
本 run：后期 PET loss 低于 MRI，两者总体都改善。但最终 test 中 MRI 相对 final validation 约恶化 3.03%，PET 约改善 2.94%，因此必须结合图 13，不能只报告总体 macro。


[04] 04_relation_loss_by_coverage.png
标题：Training loss by actual relation type
------------------------------------------------------------
图的目的：比较实际使用不同 context relation 时，anchor 自身 latent prediction objective 的难度。

英文名词：
- Actual relation type：fallback 处理后真正进入该 step 的 relation 类型。
- Self：没有使用其他 observation；anchor 自己提供 context。
- Cross-sequence：同一 session 的不同 MRI sequence。
- Longitudinal MRI：同一 subject、同一 MRI acquisition 类型的不同时间点。
- Longitudinal PET：同一 subject、同一 PET tracer 的不同时间点。
- Repeat：同一 session、同一 acquisition 的重复观测。
- Relation-specific anchor count：某 relation 实际包含的 anchor 数量，用作 coverage aggregate 权重。
- Anchor's own EMA latent：无论 relation 是什么，target 都是 anchor 自己的 EMA latent。

怎么看：
- 每条线只表示该 relation 条件下的 anchor objective；它不是 cross-modal equality loss。
- 稀有 relation 的每 coverage 样本少，曲线自然更抖；需要结合 relation count/mix，而不能只按最高线排序。
- 跨 relation 的 absolute loss 可能包含数据难度差异，不等于某种 relation 一定有害。

通常较好：主要 relation 随训练下降且稳定；有足够样本的 companion relation 不持续显著恶化；稀有 relation 的波动与样本量相符。
警报：高频 companion relation 长期显著高于 self 且不收敛；某 relation 引发持续尖峰或不稳定。
本 run：relation loss 均总体下降，但 repeat 与 longitudinal relation 更稀疏、更噪。该图不能证明 companion context 带来收益，需要 no-companion matched ablation。


[05] 05_realized_relation_mix.png
标题：Realized relation composition by coverage
------------------------------------------------------------
图的目的：比较配置希望抽到的 companion 比例与数据中真正存在合法候选后实现的 relation composition。

英文名词：
- Realized composition：经过合法候选检查与 fallback 后真正执行的构成。
- Stacked bars：堆叠柱；每根柱的所有颜色之和为 100%。
- Scheduled companion share：配置中的 nominal companion draw mass；不是保证一定能找到合法 companion 的比例。
- Legal companion：满足 subject/session/acquisition 与 split 约束、允许作为 context 的 observation。
- Fallback to self：没有合法 companion 时保持 anchor target/objective 不变，并退回 self relation。
- Relation share：该 relation anchor 数除以该 coverage 的全部 anchor 数。

怎么看：
- 浅蓝 self 区域越大，说明越多 anchor 实际没有 companion context。
- 其他颜色相加才是实际 companion 使用比例；深色参考线是 scheduled companion share。
- 图 3-5 是 curriculum ramp，coverage 6 以后参考线稳定在 50%。

通常较好：如果方法目标依赖 companion，实际 companion share 应有足够覆盖并接近设计目标；各种关键 relation 不能长期接近零。
警报：scheduled share 很高但 realized share 很低；这意味着训练名义上是 multi-relation，实际大部分是 self。
本 run：steady-state 实际 companion 使用约 17.6%，远低于 scheduled 50%。这是数据关系覆盖不足的明确信号，也意味着本 run 主要在训练 anchor-only objective；它不是 sampler 错配的直接证据，因为 fallback 是合同允许行为。


[06] 06_learning_rate_actual_vs_config.png
标题：Actual versus configured LR multiplier
------------------------------------------------------------
图的目的：核对 as-run learning-rate scheduler 是否与 resolved config 一致。

英文名词：
- LR：Learning Rate，学习率。
- Base LR：某 parameter group 配置的基础峰值学习率；GeoMC/predictor 为 0.0002。
- LR multiplier：实际 LR 除以 base LR。
- Actual：每个 optimizer update 使用前日志记录的真实 LR。
- Configured 2-coverage warmup：resolved config 要求用前两个 coverage 从低 LR 升到峰值。
- Warmup：训练初期逐步提高 LR，减少不稳定风险。
- Equivalent coverage：optimizer_step / 1198，把 step 换算成 coverage 进度。
- Patch resume：coverage 7 checkpoint resume 后启用 scheduler 修正的位置。

怎么看：
- 两条线重合表示实际执行与配置合同一致。
- 重点放大查看 coverage 1-2 的 warmup，以及 patch resume 前后是否对齐。
- 后期两条线几乎重合并不能抹去前期 mismatch，因为最终权重已经经历不同更新幅度。

通常较好：Actual 与 Configured 全程重合，或差异被提前声明并有匹配对照。
警报：warmup 长度、峰值或 decay horizon 不同；resume 后曲线突跳；日志与 config 无法对应。
本 run：coverage 1 已完成旧的 1-coverage warmup，而配置要求 2 coverage；step 7,190 起才永久匹配配置。因此这是 hybrid-scheduler run，不应描述为全程严格执行 resolved scheduler。


[07] 07_gradient_norm_stability.png
标题：Gradient norm stability by coverage
------------------------------------------------------------
图的目的：判断反向传播梯度是否稳定，以及 gradient clipping 是否经常介入。

英文名词：
- Global gradient norm：所有参与优化参数梯度的全局 L2 norm。
- clip_grad_norm_：PyTorch gradient clipping 操作；返回的是裁剪前总 norm。
- Median：每个 coverage 中间位置的 gradient norm。
- P95：95% step 不超过的值，用来看高尾部。
- Max：该 coverage 的最大裁剪前 norm。
- Clip threshold：超过 1.0 时会按比例缩小梯度。

怎么看：
- Median/P95 反映日常水平，Max 反映少数尖峰。
- 低于 threshold 不代表梯度越小越好；长期接近零也可能表示学习停滞。
- 偶发 Max 越线通常可接受，持续 P95/Median 越线才值得警惕。

通常较好：所有值 finite；Median/P95 稳定；只有少量早期或偶发 step 触发 clipping；没有逐 coverage 放大的趋势。
警报：P95 长期高于 threshold、Max 不断增大、出现 NaN/Inf，或所有 norm 迅速趋近机器精度。
本 run：最大裁剪前 norm 为 1.425，共 37 个 step 超过 1.0；后期 Median 与 P95 很低且稳定，没有 non-finite/OOM fatal。属于稳定而非持续依赖 clipping。


[08] 08_latent_standard_deviation.png
标题：Prediction and EMA-target latent standard deviation
------------------------------------------------------------
图的目的：检测 prediction amplitude 是否相对 teacher target 发生明显收缩或爆炸。

英文名词：
- Prediction：student/predictor 输出的 masked query latent。
- EMA target：停止梯度的 EMA teacher latent target。
- Latent standard deviation：选中 query latent 的数值标准差，衡量振幅/离散程度。
- Amplitude collapse：prediction std 趋近零，输出几乎变成常数。

怎么看：
- 比较蓝色 Prediction 与金色 EMA target 的距离和长期趋势。
- 两者不需要完全相等，但 prediction 持续接近零或比 target 小很多是危险信号。
- std 只能检查振幅，不检查语义、方向或下游可分性。

通常较好：Prediction std 非零、稳定，并与 target 处于相同数量级；没有突然爆炸。
警报：Prediction 趋近 0、不断远离 target，或二者同时异常缩小且 effective rank 也下降。
本 run：coverage 50 为 prediction 0.3839、target 0.3994，比值约 0.961；不支持明显 amplitude collapse，但不能据此证明 representation 有效。


[09] 09_latent_effective_rank.png
标题：Prediction and EMA-target entropy effective rank
------------------------------------------------------------
图的目的：检测 latent 是否只剩极少数有效方向，即 dimensional collapse。

英文名词：
- SVD：Singular Value Decomposition，奇异值分解。
- Entropy effective rank：把归一化奇异值视为分布，计算 exp(entropy) 得到的有效维度数。
- Channel dimension 128：latent 通道上限为 128；effective rank 不会简单等于参数维度。
- Flattening selected tokens：把选中的 token 展平后对通道相关结构做 SVD。
- Dimensional collapse：effective rank 接近 1 或持续大幅下降，表示输出集中在极少方向。

怎么看：
- 蓝色 Prediction 与金色 EMA target 都应保持明显大于 1。
- 看长期是否稳定，以及 prediction 是否越来越远离 target。
- effective rank 越高不必然越好；有意义的压缩也可能降低 rank，所以必须结合 downstream 结果。

通常较好：rank 保持多维、没有趋近 1，并与 target 走势相对稳定。
警报：prediction rank 快速跌到 1-2、持续下降，或与 target gap 不断扩大且其他 collapse 指标同步恶化。
本 run：coverage 50 prediction 13.44、target 15.73，比值约 0.854。存在一定压缩但不是 rank-1 collapse；是否有益只能由 downstream probe/ablation 判断。


[10] 10_companion_gate_and_usage.png
标题：Companion context usage and learned fusion gate
------------------------------------------------------------
图的目的：同时查看 companion 是否实际出现、覆盖多少 geometry node，以及模型给它多大有效融合权重。

英文名词：
- Companion used / anchors：实际使用 companion 的 anchor 比例。
- Mean companion node coverage：companion 特征对 FEM node 的平均可见/支持覆盖度。
- Mean supported gate：已经乘入 companion-node coverage 因子的有效 fusion gate mean。
- Fusion gate：控制 companion context 注入 anchor path 的门值。
- Supported gate：不是裸 sigmoid 参数；它包含有效 node support，因此更接近实际贡献。
- Dimensionless：无量纲，不能按 GiB、loss 等单位解读。
- Orders of magnitude：数量级差异；对数轴每跨一大格可能是 10 倍。

怎么看：
- 金色线说明 companion 有多少，橄榄线说明 node 支持范围，蓝线说明模型最终真正放行多少。
- 蓝线远低于另外两条表示 companion 虽然存在，但模型实际几乎忽略。
- 因为使用对数轴，0.1、0.01、0.001 的垂直间隔相似。

通常较好：取决于研究目标。若 companion 是核心机制，使用率应足够且 gate 不应消失；若 companion 很噪，gate 下降可能是保护性适应。
警报：方法声称依赖 companion，但 supported gate 接近零；或 gate 很高而 loss/gradient 明显恶化。
本 run：steady-state companion 使用约 17.6%，coverage 50 supported gate mean 约 7.21e-6，实际贡献几乎可忽略。这对训练稳定可能是保护，但对证明 companion 机制有效是负面证据；必须与显式 no-companion baseline 比较。


[11] 11_training_throughput.png
标题：Training throughput by coverage
------------------------------------------------------------
图的目的：判断工作站训练速度是否稳定，以及 companion 激活后额外编码是否改变吞吐。

英文名词：
- Throughput：单位时间处理量。
- Anchors/s：每秒处理的 anchor 数。
- Encoded volumes/s：每秒通过 encoder 的 volume 数；使用 companion 时可能大于 anchors/s。
- Coverage aggregate：整个 coverage 的总处理数除以所有 step elapsed time 之和。
- Step elapsed：包括 forward/backward/update 与记录的 loader wait。
- Loader wait：等待数据准备/加载的时间。

怎么看：
- 看相同 workload 阶段是否稳定；coverage 1-5 curriculum 改变时不必要求完全平坦。
- Encoded volumes/s 与 anchors/s 的差来自 companion 额外 volume，不表示免费加速。
- throughput 只能评价系统效率，不能评价模型准确性。

通常较好：curriculum 稳定后曲线平坦、没有持续衰减或周期性停顿；loader wait 占比低。
警报：吞吐逐 coverage 下降、resume 后永久降低、频繁大幅跌落，可能表示内存泄漏、I/O 或热/系统问题。
本 run：aggregate 约 5.54 anchors/s、6.44 encoded volumes/s，早期变化后长期稳定；loader wait share 极低，未显示明显 I/O 瓶颈。


[12] 12_gpu_memory.png
标题：GPU peak memory by coverage
------------------------------------------------------------
图的目的：查看真实 tensor allocation 与 CUDA allocator reservation 距离规划线和物理显存有多远。

英文名词：
- Peak allocated：活跃 tensor allocation 的历史峰值。
- Peak reserved：PyTorch CUDA allocator 向驱动保留的内存历史峰值，包括可复用缓存，通常高于 allocated。
- CUDA maximum-memory counters：每个进程内部维护的峰值计数器。
- Soft planning line：项目用于规划的 76 GiB 软阈值，不是硬 OOM 边界。
- Reported physical total：GPU 报告的总显存 94.970 GiB。
- OOM：Out Of Memory，显存不足。
- Resume：新进程恢复；max-memory 历史会在新进程中重新建立。

怎么看：
- 蓝线 allocated 最接近活跃计算负载；金线 reserved 还包含 allocator cache。
- reserved 超过 soft line 不等于活跃 tensor 超限，也不等于 OOM。
- 重点看 allocated 是否逼近 physical total，以及是否出现与 workload 无关的持续爬升。

通常较好：allocated 留有稳定余量；reserved 低于 physical total；没有 OOM；长期没有不可解释的单调增长。
警报：allocated 接近物理上限、每 coverage 持续增长、reserved 几乎无余量并伴随 allocation retry/OOM。
本 run：峰值 allocated 73.674 GiB，低于 76 GiB planning line；reserved 87.215 GiB，高于 planning line但距物理总量仍约 7.755 GiB，且没有 OOM。第三次 resume 后峰值重新建立是计数器语义，不一定是 workload 突变。


[13] 13_final_validation_test_comparison.png
标题：Final full validation versus one-time held-out test
------------------------------------------------------------
图的目的：直接比较训练期结束时的 full validation 与只运行一次的 held-out test，并检查总体 macro 是否掩盖模态反向变化。

英文名词：
- Connected dot plot：连接点图；每一行两个点对应两个 split，同一行连接线表示差值方向与大小。
- Focused x-axis：横轴围绕实际数值缩放，不从 0 开始，用于看细小差异。
- Exact loss labels：点旁直接标出精确 loss。
- Full validation @ coverage 50：最后一个 coverage 的完整 validation aggregate。
- One-time held-out test：固定 horizon 完成后只评估一次的 test。
- Fixed MRI/PET macro：MRI subject mean 与 PET subject mean 各占 0.5。
- MRI/PET subject mean：先在每个 subject 内聚合，再对 subject 求均值。

怎么看：
- loss 越低通常越好；同一行 test 点在 validation 点右侧表示 test loss 更高，左侧表示更低。
- 顶行 macro 很接近不代表下面两种 modality 都接近，因为反向差值可以相互抵消。
- 横轴不是从零开始，所以只能比较差值，不能把连接线长度当作相对总幅度。

通常较好：test 与 validation 在 macro 和每个 modality 上都接近；没有一个 modality 明显恶化；多个 seed/per-subject CI 支持稳定性。
警报：macro 接近但 modality 反向移动；test 明显高于 validation；只用总体点掩盖 subgroup shift。
本 run：macro 从 0.005799 到 0.005808，仅 +0.167%；MRI 从 0.006037 到 0.006220，约恶化 3.03%；PET 从 0.005560 到 0.005397，约改善 2.94%。总体接近是好信号，但存在模态抵消。由于没有 per-subject rows/CI，不能判断 3% 是否超过抽样波动。


最终判断速查
------------------------------------------------------------
- 优化稳定性：图 01、07 较好。
- 保存的 validation 趋势：图 02 较好，coverage 50 是最低 full-validation aggregate。
- 模态一致性：图 03、13 为黄色警报；总体 macro 掩盖 MRI/PET 反向变化。
- Companion 机制证据：图 05、10 偏负面；实际覆盖低且有效 gate 近零。
- Scheduler 合同一致性：图 06 明确不一致；这是 hybrid-scheduler run。
- Collapse 诊断：图 08、09 未显示完全 collapse，但不能替代 downstream evaluation。
- 系统效率与显存：图 11、12 稳定，无 OOM；reserved 高于 soft planning line 但仍低于物理总量。
"""
    guide_path = output_dir / "figures_en" / "list.txt"
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    guide_path.write_text(guide.strip() + "\n", encoding="utf-8")


def generate_charts(
    output_dir: Path,
    steps: pd.DataFrame,
    coverage: pd.DataFrame,
    validation: pd.DataFrame,
    test: Mapping[str, Any],
    *,
    total_gpu_gib: float,
    soft_limit_gib: float,
) -> list[dict[str, Any]]:
    charts_dir = output_dir / "figures_en"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_rows: list[dict[str, Any]] = []

    def record(name: str, question: str, family: str, claim: str) -> Path:
        path = charts_dir / name
        chart_rows.append(
            {
                "file": path.relative_to(output_dir).as_posix(),
                "analytical_question": question,
                "chart_family": family,
                "supported_claim": claim,
                "palette_policy": "single-root or hard two-root cap; explicit colors",
            }
        )
        return path

    rolling = steps["loss_total"].rolling(250, min_periods=25, center=True).median()
    path = record(
        "01_optimizer_loss_curve.png",
        "How did per-step training loss evolve along the committed trajectory?",
        "trend / log line",
        "Rapid early decline followed by a long low-loss regime without numerical divergence.",
    )
    line_chart(
        path,
        title="Canonical optimizer-step training loss",
        subtitle="Light line: individual steps; dark line: 250-step rolling median; 105 superseded replay events excluded",
        series=[
            {"name": "Individual step", "x": steps["optimizer_step"], "y": steps["loss_total"], "color": PALETTE["blue_light"], "width": 2},
            {"name": "250-step median", "x": steps["optimizer_step"], "y": rolling, "color": PALETTE["blue"], "width": 7},
        ],
        x_label="Optimizer step",
        y_label="Masked-Huber loss (log scale)",
        y_log=True,
        verticals=[
            {"x": 7189, "label": "coverage 7 resume / patch", "color": PALETTE["warning"]},
            {"x": 51514, "label": "coverage 43 validation resume", "color": PALETTE["olive"]},
        ],
    )

    monitor = validation[validation["tier"] == "fixed_monitor"].sort_values("coverage")
    full = validation[validation["tier"] == "full_validation"].sort_values("coverage")
    path = record(
        "02_train_validation_loss_by_coverage.png",
        "Did held-out validation improve across the fixed 50-coverage horizon?",
        "trend / line with interval",
        "Full validation reached its lowest saved value at the final coverage; monitor values are noisier.",
    )
    line_chart(
        path,
        title="Train and observational validation loss by coverage",
        subtitle="Train = modality-macro coverage mean; validation = subject-first fixed MRI/PET macro",
        series=[
            {"name": "Train modality macro", "x": coverage["coverage"], "y": coverage["train_loss_modality_macro"], "color": PALETTE["blue"], "width": 7, "marker": 3},
            {"name": "Fixed monitor", "x": monitor["coverage"], "y": monitor["loss_total"], "color": PALETTE["gold"], "width": 5, "marker": 5},
            {"name": "Full validation", "x": full["coverage"], "y": full["loss_total"], "color": PALETTE["orange"], "width": 7, "marker": 8},
        ],
        band={"x": coverage["coverage"], "lower": coverage["train_loss_step_p10"], "upper": coverage["train_loss_step_p90"], "fill": "#AFC9DD55"},
        x_label="Coverage",
        y_label="Masked-Huber loss (log scale)",
        y_log=True,
        y_limits=(0.0045, 0.040),
        note="Train and validation aggregation differ; the shaded band is the within-coverage step P10-P90.",
    )

    path = record(
        "03_modality_loss_by_coverage.png",
        "How did MRI and PET objective values differ over training?",
        "trend / highlighted multi-series line",
        "PET loss stayed lower than MRI over most of the later trajectory; validation markers retain the same direction.",
    )
    line_chart(
        path,
        title="MRI and PET loss by coverage",
        subtitle="Solid lines = observation-weighted train means; markers = full-validation subject means",
        series=[
            {"name": "Train MRI", "x": coverage["coverage"], "y": coverage["train_loss_mri"], "color": PALETTE["blue"], "width": 7},
            {"name": "Train PET", "x": coverage["coverage"], "y": coverage["train_loss_pet"], "color": PALETTE["gold"], "width": 7},
            {"name": "Full-val MRI", "x": full["coverage"], "y": full["loss_modality_mri"], "color": PALETTE["blue"], "width": 3, "marker": 9},
            {"name": "Full-val PET", "x": full["coverage"], "y": full["loss_modality_pet"], "color": PALETTE["gold"], "width": 3, "marker": 9},
        ],
        x_label="Coverage",
        y_label="Masked-Huber loss (log scale)",
        y_log=True,
        y_limits=(0.0045, 0.035),
    )

    relation_series = [
        ("Self", "same_observation", PALETTE["blue"]),
        ("Cross-sequence", "same_session_cross_sequence", PALETTE["orange"]),
        ("Longitudinal MRI", "longitudinal_same_sequence", PALETTE["olive"]),
        ("Longitudinal PET", "longitudinal_same_tracer", PALETTE["gold"]),
        ("Repeat", "same_session_repeat", PALETTE["pink"]),
    ]
    path = record(
        "04_relation_loss_by_coverage.png",
        "Do relation-conditioned anchor objectives show different difficulty?",
        "trend / multi-series line",
        "Relation losses are heterogeneous, but sparse relation types are noisy and are not equality objectives.",
    )
    line_chart(
        path,
        title="Training loss by actual relation type",
        subtitle="Weighted by relation-specific anchor count; every target remains the anchor's own EMA latent",
        series=[
            {"name": label, "x": coverage["coverage"], "y": coverage[f"train_loss_relation_{field}"], "color": color, "width": 5, "marker": 3}
            for label, field, color in relation_series
        ],
        x_label="Coverage",
        y_label="Masked-Huber loss (log scale)",
        y_log=True,
        y_limits=(0.0035, 0.035),
    )

    path = record(
        "05_realized_relation_mix.png",
        "How much legal companion context was actually available versus scheduled?",
        "composition / stacked bars",
        "Legal-candidate fallback makes realized companion use much lower than the 50% scheduled draw mass.",
    )
    stacked_relation_chart(path, coverage)

    path = record(
        "06_learning_rate_actual_vs_config.png",
        "Did the logged scheduler follow the resolved two-coverage warmup contract?",
        "trend / comparison line",
        "The run used a one-coverage pre-fix warmup before the coverage-7 correction switched later updates to the configured horizon.",
    )
    line_chart(
        path,
        title="Actual versus configured LR multiplier",
        subtitle="GeoMC/predictor group; actual LR is logged immediately before each optimizer update",
        series=[
            {"name": "Actual", "x": steps["optimizer_step"] / 1198.0, "y": steps["lr_multiplier_actual"], "color": PALETTE["blue"], "width": 7},
            {"name": "Configured 2-coverage warmup", "x": steps["optimizer_step"] / 1198.0, "y": steps["lr_multiplier_intended"], "color": PALETTE["gold"], "width": 6},
        ],
        x_label="Equivalent coverage (optimizer_step / 1198)",
        y_label="LR / base LR",
        y_limits=(0.0, 1.05),
        verticals=[{"x": 7189 / 1198.0, "label": "patch resume", "color": PALETTE["warning"]}],
    )

    path = record(
        "07_gradient_norm_stability.png",
        "Were gradients numerically stable and how often was clipping active?",
        "uncertainty / quantile line",
        "Gradient norms stayed well below the clip threshold after the earliest steps.",
    )
    line_chart(
        path,
        title="Gradient norm stability by coverage",
        subtitle="clip_grad_norm_ reports the total norm before clipping; the threshold is fixed at 1.0",
        series=[
            {"name": "Median", "x": coverage["coverage"], "y": coverage["grad_norm_median"], "color": PALETTE["blue"], "width": 7},
            {"name": "P95", "x": coverage["coverage"], "y": coverage["grad_norm_p95"], "color": PALETTE["gold"], "width": 6},
            {"name": "Max", "x": coverage["coverage"], "y": coverage["grad_norm_max"], "color": PALETTE["orange"], "width": 4, "marker": 4},
        ],
        x_label="Coverage",
        y_label="Global gradient norm",
        y_log=True,
        y_limits=(0.01, 2.0),
        horizontals=[{"y": 1.0, "label": "clip threshold = 1.0", "color": PALETTE["warning"]}],
    )

    path = record(
        "08_latent_standard_deviation.png",
        "Did prediction amplitude collapse relative to EMA targets?",
        "trend / two-series line",
        "Prediction standard deviation remained close to but below target standard deviation late in training.",
    )
    line_chart(
        path,
        title="Prediction and EMA-target latent standard deviation",
        subtitle="Coverage mean across selected query latents; nonzero amplitude alone is not an ablation",
        series=[
            {"name": "Prediction", "x": coverage["coverage"], "y": coverage["prediction_std"], "color": PALETTE["blue"], "width": 7},
            {"name": "EMA target", "x": coverage["coverage"], "y": coverage["target_std"], "color": PALETTE["gold"], "width": 7},
        ],
        x_label="Coverage",
        y_label="Latent standard deviation",
        y_limits=(0.28, 0.62),
    )

    path = record(
        "09_latent_effective_rank.png",
        "Did representation dimensionality collapse?",
        "trend / two-series line",
        "Prediction rank remained multi-dimensional and tracked below the EMA target rank.",
    )
    line_chart(
        path,
        title="Prediction and EMA-target entropy effective rank",
        subtitle="SVD entropy rank after flattening selected tokens; channel dimension is 128",
        series=[
            {"name": "Prediction", "x": coverage["coverage"], "y": coverage["prediction_effective_rank"], "color": PALETTE["blue"], "width": 7},
            {"name": "EMA target", "x": coverage["coverage"], "y": coverage["target_effective_rank"], "color": PALETTE["gold"], "width": 7},
        ],
        x_label="Coverage",
        y_label="Entropy effective rank",
        y_limits=(6.0, 24.0),
    )

    gate_cov = coverage[coverage["coverage"] >= 3]
    path = record(
        "10_companion_gate_and_usage.png",
        "Did the model retain substantial companion-fusion weight?",
        "trend / log line",
        "The coverage-weighted companion gate fell by orders of magnitude while legal companion usage remained nonzero.",
    )
    line_chart(
        path,
        title="Companion context usage and learned fusion gate",
        subtitle="Shared dimensionless log axis; supported gate already includes the companion-node coverage factor",
        series=[
            {"name": "Companion used / anchors", "x": gate_cov["coverage"], "y": gate_cov["companion_used_rate"], "color": PALETTE["gold"], "width": 7},
            {"name": "Mean companion node coverage", "x": gate_cov["coverage"], "y": gate_cov["node_companion_coverage_mean"], "color": PALETTE["olive"], "width": 6},
            {"name": "Mean supported gate", "x": gate_cov["coverage"], "y": gate_cov["node_companion_gate_mean_supported"], "color": PALETTE["blue"], "width": 7},
        ],
        x_label="Coverage",
        y_label="Rate / gate value (log scale)",
        y_log=True,
        y_limits=(1e-7, 0.4),
    )

    path = record(
        "11_training_throughput.png",
        "Was the workstation throughput stable over the run?",
        "trend / two-series line",
        "Anchor throughput was stable; encoded-volume throughput rose when companion context was active.",
    )
    line_chart(
        path,
        title="Training throughput by coverage",
        subtitle="Coverage aggregate = total processed / sum(step elapsed); elapsed includes loader wait",
        series=[
            {"name": "Anchors/s", "x": coverage["coverage"], "y": coverage["throughput_anchors_s"], "color": PALETTE["blue"], "width": 7},
            {"name": "Encoded volumes/s", "x": coverage["coverage"], "y": coverage["throughput_encoded_volumes_s"], "color": PALETTE["gold"], "width": 7},
        ],
        x_label="Coverage",
        y_label="Items per second",
        y_limits=(4.5, 7.2),
    )

    path = record(
        "12_gpu_memory.png",
        "How close did peak GPU memory come to planning and physical limits?",
        "benchmark / line with references",
        "Peak allocated memory stayed below the 76 GiB planning line, while allocator-reserved memory exceeded it without OOM.",
    )
    line_chart(
        path,
        title="GPU peak memory by coverage",
        subtitle="Per-process CUDA maximum-memory counters; peak history is rebuilt after the third resume",
        series=[
            {"name": "Peak allocated", "x": coverage["coverage"], "y": coverage["gpu_peak_allocated_gib_max"], "color": PALETTE["blue"], "width": 7},
            {"name": "Peak reserved", "x": coverage["coverage"], "y": coverage["gpu_peak_reserved_gib_max"], "color": PALETTE["gold"], "width": 7},
        ],
        x_label="Coverage",
        y_label="GPU memory (GiB)",
        y_limits=(68.0, total_gpu_gib + 1.0),
        horizontals=[
            {"y": soft_limit_gib, "label": "soft planning line", "color": PALETTE["warning"]},
            {"y": total_gpu_gib, "label": "reported physical total", "color": PALETTE["ink"]},
        ],
    )

    path = record(
        "13_final_validation_test_comparison.png",
        "Did final validation transfer to the held-out test split?",
        "comparison / connected dot plot",
        "Overall macro loss matched closely, but MRI and PET moved in opposite directions and canceled in the headline macro.",
    )
    validation_test_dot_plot(path, validation, test)

    return chart_rows


def build_summary(
    parsed: ParsedRun,
    steps: pd.DataFrame,
    coverage: pd.DataFrame,
    validation: pd.DataFrame,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
    qa: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    full = validation[validation["tier"] == "full_validation"].sort_values("coverage")
    monitor = validation[validation["tier"] == "fixed_monitor"].sort_values("coverage")
    final_full = full.iloc[-1]
    first_full = full.iloc[0]
    best_full = full.loc[full["loss_total"].idxmin()]
    test = parsed.held_out_test
    steady = coverage[coverage["coverage"] >= 6]
    first_started = parsed.notable_events[
        parsed.notable_events["event"] == "formal_training_workflow_started"
    ].sort_values("sequence")
    fatal = parsed.notable_events[
        parsed.notable_events["event"] == "formal_training_workflow_fatal"
    ]
    max_gpu_reserved = float(steps["gpu_peak_reserved_gib"].max())
    max_gpu_allocated = float(steps["gpu_peak_allocated_gib"].max())
    gpu_total = float(result["gpu"]["total_memory_gib"])
    scheduler_abs = np.abs(steps["lr_multiplier_delta"].to_numpy(float))
    scheduler_match = scheduler_abs < 1e-10
    first_permanent_match = None
    for index in range(len(scheduler_match)):
        if bool(scheduler_match[index:].all()):
            first_permanent_match = int(steps.iloc[index]["optimizer_step"])
            break
    full_test_delta = float(test["loss_total"]) - float(final_full["loss_total"])
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(contract["run_id"]),
        "run_directory": str((project_root / "runs" / "workstation_formal").resolve()),
        "latest_run_selection": {
            "formal_run_directories_found": ["runs/workstation_formal"],
            "requested_alias_missing": "runs/workstation/_formal",
            "selected_by": "only formal run directory plus internal completion timestamp",
        },
        "completion": {
            "completed": bool(result["completed"]),
            "fixed_coverages": int(result["fixed_horizon_coverages"]),
            "final_optimizer_step": int(result["cursor"]["optimizer_step"]),
            "final_global_step": int(result["cursor"]["global_step"]),
            "held_out_test_completed": bool(test),
            "wall_clock_first_start_to_result_seconds": (
                parsed.last_timestamp - parsed.first_timestamp
            ).total_seconds(),
            "wall_clock_display": _iso_duration(
                (parsed.last_timestamp - parsed.first_timestamp).total_seconds()
            ),
            "canonical_optimizer_elapsed_seconds": float(steps["elapsed_seconds"].sum()),
            "canonical_optimizer_elapsed_display": _iso_duration(
                float(steps["elapsed_seconds"].sum())
            ),
            "workflow_invocations": int(len(first_started)),
            "fatal_events": int(len(fatal)),
            "warnings": int(parsed.level_counts.get("WARNING", 0)),
        },
        "data_contract": {
            "catalog_observations": int(contract["observations"]["catalog"]),
            "cached_observations": int(contract["observations"]["cached"]),
            "cache_success_rate": float(contract["observations"]["cached"])
            / float(contract["observations"]["catalog"]),
            "uncached_observations": int(contract["observations"]["catalog"])
            - int(contract["observations"]["cached"]),
            "train_observations": int(contract["observations"]["train"]),
            "validation_observations": int(contract["observations"]["validation"]),
            "test_observations": int(contract["observations"]["test"]),
            "train_subjects": int(contract["subjects"]["train"]),
            "validation_subjects": int(contract["subjects"]["validation"]),
            "test_subjects": int(contract["subjects"]["test"]),
            "catalog_relations": int(contract["relations"]),
            "train_mri_observations_per_coverage": int(coverage.iloc[0]["count_mri"]),
            "train_pet_observations_per_coverage": int(coverage.iloc[0]["count_pet"]),
        },
        "trajectory": {
            "unique_optimizer_steps": int(len(steps)),
            "canonical_train_anchors": int(steps["anchors_processed"].sum()),
            "expected_train_anchors": int(contract["observations"]["train"])
            * int(contract["fixed_coverages"]),
            "canonical_encoded_volumes": int(steps["encoded_volumes_processed"].sum()),
            "raw_step_events": int(parsed.event_counts["training_optimizer_step"]),
            "discarded_replayed_events": int(len(parsed.abandoned_steps)),
            "discarded_replayed_anchor_executions": int(
                parsed.abandoned_steps["anchors_processed"].sum()
                if not parsed.abandoned_steps.empty
                else 0
            ),
            "discarded_step_range": [
                int(parsed.abandoned_steps["optimizer_step"].min()),
                int(parsed.abandoned_steps["optimizer_step"].max()),
            ]
            if not parsed.abandoned_steps.empty
            else [],
            "resume_checkpoint_after_fatal_step": 7188,
            "coverage43_validation_only_resume_step": 51514,
        },
        "loss": {
            "train_coverage1_modality_macro": float(
                coverage.iloc[0]["train_loss_modality_macro"]
            ),
            "train_coverage50_modality_macro": float(
                coverage.iloc[-1]["train_loss_modality_macro"]
            ),
            "first_full_validation_coverage": int(first_full["coverage"]),
            "first_full_validation_loss": float(first_full["loss_total"]),
            "final_full_validation_loss": float(final_full["loss_total"]),
            "best_full_validation_coverage": int(best_full["coverage"]),
            "best_full_validation_loss": float(best_full["loss_total"]),
            "monitor_min_coverage": int(monitor.loc[monitor["loss_total"].idxmin()]["coverage"]),
            "monitor_min_loss": float(monitor["loss_total"].min()),
            "held_out_test_loss": float(test["loss_total"]),
            "test_minus_final_validation": full_test_delta,
            "test_vs_final_validation_percent": 100.0
            * full_test_delta
            / float(final_full["loss_total"]),
            "final_validation_mri": float(final_full["loss_modality_mri"]),
            "test_mri": float(test["loss_modality_mri"]),
            "final_validation_pet": float(final_full["loss_modality_pet"]),
            "test_pet": float(test["loss_modality_pet"]),
        },
        "representation": {
            "coverage50_prediction_std": float(coverage.iloc[-1]["prediction_std"]),
            "coverage50_target_std": float(coverage.iloc[-1]["target_std"]),
            "coverage50_std_ratio": float(coverage.iloc[-1]["prediction_std"])
            / float(coverage.iloc[-1]["target_std"]),
            "coverage50_prediction_effective_rank": float(
                coverage.iloc[-1]["prediction_effective_rank"]
            ),
            "coverage50_target_effective_rank": float(
                coverage.iloc[-1]["target_effective_rank"]
            ),
            "coverage50_rank_ratio": float(
                coverage.iloc[-1]["prediction_effective_rank"]
            )
            / float(coverage.iloc[-1]["target_effective_rank"]),
            "test_prediction_std": float(test["prediction_std"]),
            "test_target_std": float(test["target_std"]),
        },
        "companion_context": {
            "steady_coverage_mean_requested_rate": float(
                steady["companion_requested_rate"].mean()
            ),
            "steady_coverage_mean_used_rate": float(steady["companion_used_rate"].mean()),
            "scheduled_steady_companion_draw_rate": 0.50,
            "total_fallback_count": int(steps["companion_fallback_count"].sum()),
            "coverage3_gate_first_step": float(
                steps[steps["coverage"] == 3].iloc[0]["node_companion_gate_mean_supported"]
            ),
            "coverage3_gate_last_step": float(
                steps[steps["coverage"] == 3].iloc[-1]["node_companion_gate_mean_supported"]
            ),
            "coverage50_gate_weighted_mean": float(
                coverage.iloc[-1]["node_companion_gate_mean_supported"]
            ),
        },
        "scheduler": {
            "configured_warmup_coverages": int(
                config["training"]["scheduler"]["warmup_coverages"]
            ),
            "configured_warmup_updates": int(contract["steps_per_coverage"])
            * int(config["training"]["scheduler"]["warmup_coverages"]),
            "observed_first_peak_step": int(
                steps.loc[steps["lr_multiplier_actual"].idxmax()]["optimizer_step"]
            ),
            "first_step_from_which_actual_permanently_matches_configured": first_permanent_match,
            "max_absolute_multiplier_deviation": float(scheduler_abs.max()),
            "inference": (
                "Steps 1-7189 follow the pre-fix one-coverage-warmup scheduler; "
                "the new LambdaLR takes effect after the coverage-7 checkpoint resume."
            ),
        },
        "stability_and_resources": {
            "gradient_norm_max": float(steps["grad_norm"].max()),
            "gradient_steps_above_clip_threshold": int((steps["grad_norm"] > 1.0).sum()),
            "gradient_clip_threshold": float(config["training"]["gradient_clip_norm"]),
            "nonfinite_or_oom_fatals": 0,
            "throughput_anchors_s_aggregate": float(steps["anchors_processed"].sum())
            / float(steps["elapsed_seconds"].sum()),
            "throughput_encoded_volumes_s_aggregate": float(
                steps["encoded_volumes_processed"].sum()
            )
            / float(steps["elapsed_seconds"].sum()),
            "loader_wait_share": float(steps["loader_wait_seconds"].sum())
            / float(steps["elapsed_seconds"].sum()),
            "gpu_peak_allocated_gib": max_gpu_allocated,
            "gpu_peak_reserved_gib": max_gpu_reserved,
            "gpu_total_gib": gpu_total,
            "gpu_reserved_headroom_gib": gpu_total - max_gpu_reserved,
            "gpu_soft_limit_gib": float(config["resources"]["vram_soft_limit_gib"]),
        },
        "provenance": {
            "config_digest": str(contract["hashes"]["config"]),
            "frozen_python_source_digest": str(contract["hashes"]["python_source"]),
            "current_python_source_digest": _source_digest(project_root),
            "resume_migration": "companion-fallback-and-configured-horizon-scheduler-v1",
            "checkpoint_files_present_locally": len(list((project_root / "runs" / "workstation_formal" / "checkpoints").glob("*"))),
        },
        "data_quality_qa": dict(qa),
        "limitations": [
            "Checkpoints and atomic completion markers were not synchronized locally, so checkpoint bytes, parameter tensors, and independent re-evaluation cannot be verified in this analysis.",
            "The 512-anchor monitor subset changes with coverage because its deterministic selector includes coverage in the hash; it is stratified but not a fixed longitudinal panel.",
            "Full validation uses all cached validation observations but changes deterministic mask/relation views with coverage; comparisons are descriptive, not paired-view estimates.",
            "No uncertainty intervals or per-subject loss rows were saved; significance and paired subject-level uncertainty cannot be reconstructed from aggregate logs.",
            "A low fusion gate is evidence of strong attenuation, not a causal ablation proving GeoMC or companion context is unnecessary.",
            "Pretraining JEPA loss does not establish downstream clinical utility; downstream task evaluation remains required.",
        ],
    }
    return summary


def write_bundle_index(output_dir: Path, summary: Mapping[str, Any], chart_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Formal training analysis bundle",
        "",
        f"Run: `{summary['run_id']}`",
        "",
        "This folder is derived from the synchronized contracts and append-only logs. The original run files were not modified.",
        "",
        "## Reproducible tables",
        "",
        "- `canonical_steps.csv.gz`: last occurrence for each optimizer step; abandoned replay events are excluded.",
        "- `coverage_summary.csv`: weighted coverage-level summaries used by most figures.",
        "- `validation_summary.csv`: fixed monitor, full validation, and held-out test aggregates.",
        "- `notable_events.csv`: starts, resumes, fatal/warning events, checkpoint loads, and completion.",
        "- `event_counts.csv`: complete event-type and severity counts.",
        "- `summary.json`: computed headline values and explicit limitations.",
        "- `qa_checks.json`: structural checks applied before chart generation.",
        "- `report_data.sqlite`: bounded derived tables used by the validated interactive report.",
        "- `report_sql_snapshot.json`: actual SQL text and the exact bounded rows returned for the report.",
        "- `TRAINING_ANALYSIS.md`: Chinese technical analysis with every standalone figure embedded separately.",
        "- `figures_en/list.txt`: per-figure Chinese glossary, reading method, favorable signs, warnings, run-specific interpretation, and evidence limits.",
        "",
        "## Standalone figures",
        "",
    ]
    for row in chart_rows:
        lines.append(f"- `{row['file']}` — {row['analytical_question']}")
    lines.extend(
        [
            "",
            "## Canonicalization rule",
            "",
            "The JSONL contains 60,005 optimizer-step events but only 59,900 unique optimizer-step IDs. Steps 7,189–7,293 were computed before a fatal optional-companion error, then replayed from the last committed checkpoint. The final trajectory keeps the later occurrence for those 105 IDs.",
            "",
            "## Evidence boundary",
            "",
            "No local checkpoint was available. The bundle analyzes contracts, logs, and saved aggregate evaluation only; it cannot independently load the final model or rerun validation/test.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_sql_report_snapshot(
    output_dir: Path,
    steps: pd.DataFrame,
    coverage: pd.DataFrame,
    validation: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    """Materialize the bounded report datasets behind real, rerunnable SQL.

    The Data Analytics report renderer requires provenance SQL for native
    charts.  This database contains derived analysis tables only; it does not
    copy checkpoint tensors or modify any synchronized run artifact.
    """

    coverage_report = coverage.copy()
    fixed_monitor = validation[validation["tier"] == "fixed_monitor"].set_index(
        "coverage"
    )
    full_validation = validation[
        validation["tier"] == "full_validation"
    ].set_index("coverage")
    coverage_report["fixed_monitor_loss"] = coverage_report["coverage"].map(
        fixed_monitor["loss_total"]
    )
    coverage_report["full_validation_loss"] = coverage_report["coverage"].map(
        full_validation["loss_total"]
    )
    coverage_report["full_validation_mri"] = coverage_report["coverage"].map(
        full_validation["loss_modality_mri"]
    )
    coverage_report["full_validation_pet"] = coverage_report["coverage"].map(
        full_validation["loss_modality_pet"]
    )
    scheduler_by_coverage = steps.groupby("coverage", sort=True)[
        ["lr_multiplier_actual", "lr_multiplier_intended"]
    ].mean()
    coverage_report["lr_actual_multiplier"] = coverage_report["coverage"].map(
        scheduler_by_coverage["lr_multiplier_actual"]
    )
    coverage_report["lr_configured_multiplier"] = coverage_report["coverage"].map(
        scheduler_by_coverage["lr_multiplier_intended"]
    )

    relation_fields = (
        ("self", "rate_relation_same_observation"),
        ("cross-sequence", "rate_relation_same_session_cross_sequence"),
        ("longitudinal MRI", "rate_relation_longitudinal_same_sequence"),
        ("longitudinal PET", "rate_relation_longitudinal_same_tracer"),
        ("repeat", "rate_relation_same_session_repeat"),
    )
    relation_rows: list[dict[str, Any]] = []
    for _, row in coverage.iterrows():
        coverage_index = int(row["coverage"])
        if coverage_index <= 2:
            target_companion_rate = 0.0
        elif coverage_index == 3:
            target_companion_rate = 0.25
        elif coverage_index == 4:
            target_companion_rate = 0.375
        elif coverage_index == 5:
            target_companion_rate = 0.4375
        else:
            target_companion_rate = 0.5
        for relation, field in relation_fields:
            relation_rows.append(
                {
                    "coverage": coverage_index,
                    "relation": relation,
                    "rate": float(row[field]),
                    "target_companion_rate": target_companion_rate,
                    "anchors": int(row["anchors"]),
                }
            )
    relation_mix = pd.DataFrame(relation_rows)

    full_final = validation[
        (validation["tier"] == "full_validation")
        & (validation["coverage"].astype(int) == 50)
    ].iloc[-1]
    held_out = validation[validation["tier"] == "held_out_test"].iloc[-1]
    generalization_rows: list[dict[str, Any]] = []
    for scope, field in (
        ("Fixed MRI/PET macro", "loss_total"),
        ("MRI subject mean", "loss_modality_mri"),
        ("PET subject mean", "loss_modality_pet"),
    ):
        for split, row in (
            ("Full validation @ c50", full_final),
            ("Held-out test", held_out),
        ):
            generalization_rows.append(
                {
                    "scope": scope,
                    "split": split,
                    "loss": float(row[field]),
                    "subject_count": int(row["subject_count"]),
                    "anchor_views": int(row["anchor_views"]),
                }
            )
    generalization = pd.DataFrame(generalization_rows)

    headline = pd.DataFrame(
        [
            {
                "final_validation_loss": summary["loss"]["final_full_validation_loss"],
                "heldout_test_loss": summary["loss"]["held_out_test_loss"],
                "test_gap_percent": summary["loss"]["test_vs_final_validation_percent"]
                / 100.0,
                "optimizer_steps": summary["trajectory"]["unique_optimizer_steps"],
                "canonical_anchors": summary["trajectory"]["canonical_train_anchors"],
                "replayed_events_discarded": summary["trajectory"][
                    "discarded_replayed_events"
                ],
                "throughput_anchors_s": summary["stability_and_resources"][
                    "throughput_anchors_s_aggregate"
                ],
                "gpu_peak_allocated_gib": summary["stability_and_resources"][
                    "gpu_peak_allocated_gib"
                ],
                "gpu_peak_reserved_gib": summary["stability_and_resources"][
                    "gpu_peak_reserved_gib"
                ],
                "checkpoint_files_local": summary["provenance"][
                    "checkpoint_files_present_locally"
                ],
            }
        ]
    )

    database_path = output_dir / "report_data.sqlite"
    queries = {
        "headline": "SELECT * FROM headline_metrics",
        "coverage": "SELECT * FROM coverage_summary ORDER BY coverage",
        "relation_mix": (
            "SELECT coverage, relation, rate, target_companion_rate, anchors "
            "FROM relation_mix ORDER BY coverage, relation"
        ),
        "generalization": (
            "SELECT scope, split, loss, subject_count, anchor_views "
            "FROM generalization ORDER BY scope, split"
        ),
        "evaluation": (
            "SELECT timestamp, coverage, optimizer_step, tier, reason, aggregation, "
            "anchor_views, subject_count, loss_total, loss_modality_mri, "
            "loss_modality_pet, loss_observation_mean, prediction_std, target_std, "
            "companion_requested_count, companion_used_count, companion_fallback_count "
            "FROM validation_summary "
            "WHERE tier IN ('full_validation', 'held_out_test') "
            "ORDER BY optimizer_step, CASE tier WHEN 'full_validation' THEN 0 ELSE 1 END"
        ),
    }
    datasets: dict[str, list[dict[str, Any]]] = {}
    with sqlite3.connect(database_path) as connection:
        coverage_report.to_sql(
            "coverage_summary", connection, if_exists="replace", index=False
        )
        validation.to_sql("validation_summary", connection, if_exists="replace", index=False)
        relation_mix.to_sql("relation_mix", connection, if_exists="replace", index=False)
        generalization.to_sql("generalization", connection, if_exists="replace", index=False)
        headline.to_sql("headline_metrics", connection, if_exists="replace", index=False)
        connection.row_factory = sqlite3.Row
        for dataset, query in queries.items():
            datasets[dataset] = [dict(row) for row in connection.execute(query).fetchall()]

    payload = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "database": database_path.name,
        "queries": queries,
        "datasets": datasets,
    }
    (output_dir / "report_sql_snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/workstation_formal"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/workstation_formal/analysis_20260825"),
    )
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((run_dir / "contracts" / "config.resolved.json").read_text(encoding="utf-8"))
    contract = json.loads((run_dir / "contracts" / "run_contract.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "formal_training_result.json").read_text(encoding="utf-8"))
    parsed = parse_run(run_dir / "logs" / "train.metrics.jsonl")
    coverage = aggregate_coverages(parsed.steps)
    steps = add_scheduler_diagnostics(
        parsed.steps,
        base_lr=float(config["training"]["learning_rates"]["geomc_predictor"]),
        warmup_updates=int(contract["steps_per_coverage"])
        * int(config["training"]["scheduler"]["warmup_coverages"]),
        total_updates=int(contract["total_optimizer_steps"]),
    )
    parsed.steps = steps
    qa = validate_canonical_run(
        parsed,
        coverage,
        contract,
        run_dir / "logs" / "train.log",
    )
    blocking_checks = [
        "jsonl_text_line_parity",
        "sequence_is_exact_1_to_n",
        "optimizer_steps_exact_1_to_horizon",
        "coverage_count_matches_contract",
        "coverage_steps_match_contract",
        "coverage_anchors_match_contract",
        "modality_counts_sum_to_anchors",
        "relation_counts_sum_to_anchors",
    ]
    failed = [name for name in blocking_checks if not bool(qa[name])]
    if qa["nonfinite_core_step_values"]:
        failed.append("nonfinite_core_step_values")
    if failed:
        raise RuntimeError("Analysis QA failed: " + ", ".join(failed))

    validation = parsed.validation.copy()
    test_row = {
        "timestamp": parsed.last_timestamp.isoformat(),
        "sequence": parsed.line_count,
        "coverage": int(result["cursor"]["coverage"]),
        "optimizer_step": int(result["cursor"]["optimizer_step"]),
        "tier": "held_out_test",
        "reason": "one-time held-out test after fixed horizon",
        **parsed.held_out_test,
    }
    validation_with_test = pd.concat(
        [validation, pd.DataFrame([test_row])], ignore_index=True, sort=False
    )

    summary = build_summary(
        parsed,
        steps,
        coverage,
        validation,
        config,
        contract,
        result,
        qa,
        project_root,
    )
    chart_rows = generate_charts(
        output_dir,
        steps,
        coverage,
        validation,
        parsed.held_out_test,
        total_gpu_gib=float(result["gpu"]["total_memory_gib"]),
        soft_limit_gib=float(config["resources"]["vram_soft_limit_gib"]),
    )
    write_english_figure_guide(output_dir)

    with gzip.open(output_dir / "canonical_steps.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        steps.to_csv(handle, index=False)
    coverage.to_csv(output_dir / "coverage_summary.csv", index=False)
    validation_with_test.to_csv(output_dir / "validation_summary.csv", index=False)
    parsed.notable_events.to_csv(output_dir / "notable_events.csv", index=False)
    pd.DataFrame(
        [
            {"kind": "event", "name": key, "count": value}
            for key, value in sorted(parsed.event_counts.items())
        ]
        + [
            {"kind": "level", "name": key, "count": value}
            for key, value in sorted(parsed.level_counts.items())
        ]
    ).to_csv(output_dir / "event_counts.csv", index=False)
    pd.DataFrame(chart_rows).to_csv(output_dir / "chart_map.csv", index=False)
    (output_dir / "qa_checks.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_sql_report_snapshot(
        output_dir, steps, coverage, validation_with_test, summary
    )
    write_bundle_index(output_dir, summary, chart_rows)

    receipt = {
        "output_dir": str(output_dir),
        "figures": len(chart_rows),
        "unique_optimizer_steps": len(steps),
        "discarded_replayed_events": len(parsed.abandoned_steps),
        "qa": "passed",
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
