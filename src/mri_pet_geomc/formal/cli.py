"""Separate command-line entry points for formal preprocessing and training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from .config import load_formal_config


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="configs/workstation_formal.yaml",
        help="Formal workstation YAML configuration.",
    )
    parser.add_argument(
        "--plain-progress",
        action="store_true",
        help="Use the clean text fallback instead of the Rich live display.",
    )
    return parser


def _resume_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", dest="resume", action="store_true", default=True)
    group.add_argument("--no-resume", dest="resume", action="store_false")


def preprocess_main(argv: Sequence[str] | None = None) -> int:
    parser = _common_parser("Build the catalog and resumable model-ready cache.")
    _resume_flags(parser)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Development-only upper bound on cases; never use for the formal run.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry cases recorded as structurally failed on an earlier attempt.",
    )
    args = parser.parse_args(argv)
    config = load_formal_config(args.config)
    from .workflows import run_preprocessing_workflow

    result = run_preprocessing_workflow(
        config,
        resume=bool(args.resume),
        force_plain_progress=bool(args.plain_progress),
        limit=args.limit,
        retry_failed=bool(args.retry_failed),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = _common_parser("Train the single formal MRI/PET GeoMC-JEPA model.")
    _resume_flags(parser)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the bounded synthetic integration path, never real training.",
    )
    args = parser.parse_args(argv)
    config = load_formal_config(args.config)
    from .workflows import run_training_workflow

    result = run_training_workflow(
        config,
        resume=bool(args.resume),
        force_plain_progress=bool(args.plain_progress),
        smoke=bool(args.smoke),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def inspect_main(argv: Sequence[str] | None = None) -> int:
    parser = _common_parser("Inspect paths, assets, inventory and cache contracts.")
    args = parser.parse_args(argv)
    config = load_formal_config(args.config)
    from .workflows import inspect_workstation

    result = inspect_workstation(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Formal workstation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("preprocess", "catalog plus resumable offline preprocessing"),
        ("train", "single formal training run"),
        ("inspect", "read-only contract inspection"),
    ):
        subparsers.add_parser(name, help=help_text, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    dispatch: dict[str, Any] = {
        "preprocess": preprocess_main,
        "train": train_main,
        "inspect": inspect_main,
    }
    return int(dispatch[args.command](remaining))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["inspect_main", "main", "preprocess_main", "train_main"]
