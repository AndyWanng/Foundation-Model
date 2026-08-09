"""Pure, subject-first metrics for the bounded GeoMC feasibility study.

The functions in this module never treat tokens, masked views, or repeated
relations as independent subjects.  Latent metrics are reduced within each
batch item first; tabular metrics are reduced within subject (and condition)
before cohort summaries or paired bootstrap comparisons are computed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _latent_inputs(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    token_weight: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target, device=prediction.device)
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("prediction and target must have identical [B,T,D] shapes")
    if prediction.shape[-1] < 1:
        raise ValueError("The latent feature dimension must be non-empty")

    selected = torch.as_tensor(mask, device=prediction.device, dtype=torch.bool)
    if selected.shape != prediction.shape[:2]:
        raise ValueError("mask must have shape [B,T]")
    if torch.any(selected.sum(dim=1) < 1):
        raise ValueError("Every batch item must select at least one token")

    if token_weight is None:
        weight = selected.to(dtype=torch.float32)
    else:
        weight = torch.as_tensor(
            token_weight, device=prediction.device, dtype=torch.float32
        )
        if weight.shape != selected.shape:
            raise ValueError("token_weight must have shape [B,T]")
        if not torch.isfinite(weight).all() or torch.any(weight < 0):
            raise ValueError("token_weight must be finite and non-negative")
        weight = weight * selected
        if torch.any(weight.sum(dim=1) <= 0):
            raise ValueError("Every batch item needs positive selected token weight")

    return prediction.detach(), target.detach(), selected, weight


@torch.no_grad()
def masked_latent_metrics(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    token_weight: Tensor | None = None,
    huber_beta: float = 1.0,
    cosine_epsilon: float = 1.0e-8,
) -> dict[str, float]:
    """Compute masked Huber, MSE, and cosine with equal batch-item weighting.

    The final feature axis is the latent channel axis.  Each token metric is
    averaged inside its batch item using ``token_weight`` (or a uniform mask),
    then the item means are averaged.  Values outside ``mask`` are never read,
    so non-finite padding outside the selected region is harmless.
    """

    if not np.isfinite(huber_beta) or float(huber_beta) <= 0:
        raise ValueError("huber_beta must be finite and positive")
    if not np.isfinite(cosine_epsilon) or float(cosine_epsilon) <= 0:
        raise ValueError("cosine_epsilon must be finite and positive")
    prediction, target, selected, weight = _latent_inputs(
        prediction, target, mask, token_weight
    )

    example_huber: list[Tensor] = []
    example_mse: list[Tensor] = []
    example_cosine: list[Tensor] = []
    selected_token_counts: list[int] = []
    for item in range(prediction.shape[0]):
        left = prediction[item, selected[item]].float()
        right = target[item, selected[item]].float()
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError("Selected prediction and target values must be finite")
        item_weight = weight[item, selected[item]].to(device=left.device)
        item_weight = item_weight / item_weight.sum()

        huber_per_token = F.smooth_l1_loss(
            left, right, beta=float(huber_beta), reduction="none"
        ).mean(dim=-1)
        mse_per_token = (left - right).square().mean(dim=-1)
        cosine_per_token = F.cosine_similarity(
            left, right, dim=-1, eps=float(cosine_epsilon)
        )
        example_huber.append((huber_per_token * item_weight).sum())
        example_mse.append((mse_per_token * item_weight).sum())
        example_cosine.append((cosine_per_token * item_weight).sum())
        selected_token_counts.append(int(selected[item].sum()))

    return {
        "huber": float(torch.stack(example_huber).mean().cpu()),
        "mse": float(torch.stack(example_mse).mean().cpu()),
        "cosine": float(torch.stack(example_cosine).mean().cpu()),
        "mean_selected_token_count": float(np.mean(selected_token_counts)),
    }


def _effective_rank(value: Tensor, *, epsilon: float) -> Tensor:
    centered = value.float() - value.float().mean(dim=0, keepdim=True)
    singular_energy = torch.linalg.svdvals(centered).square()
    total = singular_energy.sum()
    if float(total) <= epsilon:
        return torch.zeros((), device=value.device)
    probability = singular_energy / total
    entropy = -(probability * probability.clamp_min(epsilon).log()).sum()
    return entropy.exp()


@torch.no_grad()
def representation_statistics(
    latent: Tensor,
    mask: Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, float]:
    """Return subject-first variance, effective rank, and RMS amplitude.

    Variance is measured after centering selected tokens within each subject.
    Effective rank is the entropy rank of the centered token-by-channel matrix.
    A constant or single-token representation has effective rank zero, which
    makes complete collapse explicit instead of assigning it rank one.
    """

    if not np.isfinite(epsilon) or float(epsilon) <= 0:
        raise ValueError("epsilon must be finite and positive")
    latent = torch.as_tensor(latent).detach()
    if latent.ndim != 3 or latent.shape[-1] < 1:
        raise ValueError("latent must have shape [B,T,D] with D > 0")
    selected = torch.as_tensor(mask, device=latent.device, dtype=torch.bool)
    if selected.shape != latent.shape[:2]:
        raise ValueError("mask must have shape [B,T]")
    if torch.any(selected.sum(dim=1) < 1):
        raise ValueError("Every batch item must select at least one token")

    rows: list[dict[str, Tensor]] = []
    for item in range(latent.shape[0]):
        value = latent[item, selected[item]].float()
        if not torch.isfinite(value).all():
            raise ValueError("Selected latent values must be finite")
        centered = value - value.mean(dim=0, keepdim=True)
        rows.append(
            {
                "variance": centered.square().mean(),
                "effective_rank": _effective_rank(value, epsilon=float(epsilon)),
                "rms_amplitude": value.square().mean().sqrt(),
            }
        )
    return {
        name: float(torch.stack([row[name] for row in rows]).mean().cpu())
        for name in ("variance", "effective_rank", "rms_amplitude")
    }


def _metric_keys(metric_keys: Sequence[str]) -> tuple[str, ...]:
    keys = tuple(str(key) for key in metric_keys)
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("metric_keys must be a non-empty sequence of unique names")
    return keys


def _finite_metric(row: Mapping[str, Any], key: str, row_index: int) -> float:
    if key not in row:
        raise KeyError(f"Row {row_index} is missing metric {key!r}")
    value = row[key]
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Row {row_index} metric {key!r} is boolean, not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Row {row_index} metric {key!r} is not numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"Row {row_index} metric {key!r} must be finite")
    return result


def aggregate_subject_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str],
    subject_key: str = "subject_id",
    group_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Average repeated views within subject before producing group summaries.

    ``group_keys`` can preserve conditions such as ``pairing_mode``.  The result
    is JSON-friendly and deterministically ordered.  ``subject_rows`` contains
    one row per subject/group; ``summary_rows`` gives an equal-subject mean for
    every group.
    """

    if not rows:
        raise ValueError("rows must be non-empty")
    metrics = _metric_keys(metric_keys)
    groups = tuple(str(key) for key in group_keys)
    if any(not key for key in groups) or len(set(groups)) != len(groups):
        raise ValueError("group_keys must contain unique, non-empty names")
    if not subject_key or subject_key in groups:
        raise ValueError("subject_key must be non-empty and distinct from group_keys")
    reserved = {subject_key, *groups, "view_count"}
    overlap = reserved.intersection(metrics)
    if overlap:
        raise ValueError(f"Metric names conflict with grouping fields: {sorted(overlap)}")

    grouped: dict[tuple[tuple[Any, ...], str], list[dict[str, float]]] = defaultdict(list)
    view_counts_by_group: dict[tuple[Any, ...], int] = defaultdict(int)
    for index, row in enumerate(rows):
        if subject_key not in row or not str(row[subject_key]).strip():
            raise KeyError(f"Row {index} has no non-empty {subject_key!r}")
        missing_groups = [key for key in groups if key not in row]
        if missing_groups:
            raise KeyError(f"Row {index} is missing group keys {missing_groups}")
        group = tuple(row[key] for key in groups)
        try:
            hash(group)
        except TypeError as error:
            raise TypeError(f"Row {index} group values must be hashable") from error
        subject = str(row[subject_key])
        values = {key: _finite_metric(row, key, index) for key in metrics}
        grouped[(group, subject)].append(values)
        view_counts_by_group[group] += 1

    def sort_group(group: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(repr(value) for value in group)

    subject_rows: list[dict[str, Any]] = []
    for (group, subject), values in sorted(
        grouped.items(), key=lambda item: (sort_group(item[0][0]), item[0][1])
    ):
        subject_rows.append(
            {
                **dict(zip(groups, group, strict=True)),
                subject_key: subject,
                **{
                    key: float(np.mean([value[key] for value in values]))
                    for key in metrics
                },
                "view_count": len(values),
            }
        )

    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        by_group[tuple(row[key] for key in groups)].append(row)
    summary_rows: list[dict[str, Any]] = []
    for group, selected_rows in sorted(by_group.items(), key=lambda item: sort_group(item[0])):
        summary_rows.append(
            {
                **dict(zip(groups, group, strict=True)),
                **{
                    key: float(np.mean([float(row[key]) for row in selected_rows]))
                    for key in metrics
                },
                "subject_count": len(selected_rows),
                "view_count": view_counts_by_group[group],
            }
        )

    return {
        "subject_key": subject_key,
        "group_keys": list(groups),
        "metric_keys": list(metrics),
        "unique_subject_count": len({row[subject_key] for row in subject_rows}),
        "view_count": len(rows),
        "subject_rows": subject_rows,
        "summary_rows": summary_rows,
    }


def _subject_mapping(values: Mapping[object, float], name: str) -> dict[str, float]:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    normalized: dict[str, float] = {}
    for raw_subject, raw_value in values.items():
        subject = str(raw_subject)
        if not subject.strip():
            raise ValueError(f"{name} contains an empty subject identifier")
        if subject in normalized:
            raise ValueError(f"{name} has colliding subject identifier {subject!r}")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"{name}[{subject!r}] must be finite")
        normalized[subject] = value
    return normalized


def paired_subject_bootstrap_difference(
    left: Mapping[object, float],
    right: Mapping[object, float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the mean paired subject difference ``left - right``.

    Subject sets must match exactly.  Resampling the already paired differences
    preserves dependence between conditions and prevents view-level
    pseudoreplication.  The interval is a deterministic percentile interval.
    """

    if int(n_resamples) != n_resamples or int(n_resamples) < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    left_values = _subject_mapping(left, "left")
    right_values = _subject_mapping(right, "right")
    left_subjects = set(left_values)
    right_subjects = set(right_values)
    if left_subjects != right_subjects:
        missing_left = sorted(right_subjects - left_subjects)
        missing_right = sorted(left_subjects - right_subjects)
        raise ValueError(
            "Paired bootstrap requires identical subject sets; "
            f"missing_from_left={missing_left}, missing_from_right={missing_right}"
        )

    subjects = sorted(left_subjects)
    differences = np.asarray(
        [left_values[subject] - right_values[subject] for subject in subjects],
        dtype=np.float64,
    )
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, len(subjects), size=(int(n_resamples), len(subjects)), endpoint=False
    )
    bootstrap_means = differences[indices].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail], method="linear")
    return {
        "difference_definition": "left_minus_right",
        "estimate": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
        "subject_count": len(subjects),
        "subject_difference_sd": float(differences.std(ddof=1)) if len(subjects) > 1 else 0.0,
        "bootstrap_fraction_above_zero": float(np.mean(bootstrap_means > 0.0)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "subject_ids": subjects,
    }


def compare_pairing_conditions(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    subject_key: str = "subject_id",
    pairing_key: str = "pairing_mode",
    verified_pairing: object = "verified",
    mismatched_pairing: object = "wrong_subject",
    unpaired_pairing: object = "unpaired",
    higher_is_better: bool = False,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare verified pairing with mismatched-subject and unpaired controls.

    Repeated rows are first averaged within subject and pairing mode.  Returned
    advantages are oriented so that positive always means the verified-pair
    pairing is better.  For a loss this is ``control - verified``; for a score
    where larger is better it is ``verified - control``.
    """

    aggregation = aggregate_subject_metrics(
        rows,
        metric_keys=[metric_key],
        subject_key=subject_key,
        group_keys=[pairing_key],
    )
    pairing_values_by_subject: dict[object, dict[str, float]] = defaultdict(dict)
    for row in aggregation["subject_rows"]:
        pairing_values_by_subject[row[pairing_key]][str(row[subject_key])] = float(row[metric_key])

    required = (verified_pairing, mismatched_pairing, unpaired_pairing)
    missing = [pairing for pairing in required if pairing not in pairing_values_by_subject]
    if missing:
        raise ValueError(f"Missing required pairing conditions: {missing}")
    verified_values = pairing_values_by_subject[verified_pairing]

    comparisons: dict[str, dict[str, Any]] = {}
    for offset, (label, control_condition) in enumerate(
        (("verified_vs_mismatched", mismatched_pairing), ("verified_vs_unpaired", unpaired_pairing))
    ):
        control = pairing_values_by_subject[control_condition]
        if higher_is_better:
            result = paired_subject_bootstrap_difference(
                verified_values,
                control,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=int(seed) + offset,
            )
            definition = f"{verified_pairing} - {control_condition}"
        else:
            result = paired_subject_bootstrap_difference(
                control,
                verified_values,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=int(seed) + offset,
            )
            definition = f"{control_condition} - {verified_pairing}"
        comparisons[label] = {
            **result,
            "advantage_definition": definition,
            "positive_means": "verified_pairing_is_better",
            "control_pairing": control_condition,
        }

    pairing_means = {
        row[pairing_key]: float(row[metric_key])
        for row in aggregation["summary_rows"]
    }
    return {
        "metric": metric_key,
        "higher_is_better": bool(higher_is_better),
        "pairing_means": pairing_means,
        "comparisons": comparisons,
        "subject_rows": aggregation["subject_rows"],
    }


__all__ = [
    "aggregate_subject_metrics",
    "compare_pairing_conditions",
    "masked_latent_metrics",
    "paired_subject_bootstrap_difference",
    "representation_statistics",
]
