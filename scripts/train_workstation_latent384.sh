#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/workstation_formal_latent384.yaml}"

# Reuse the standard environment, live terminal output and strict --resume path.
exec bash "$PROJECT_ROOT/scripts/train_workstation.sh" "$CONFIG"
