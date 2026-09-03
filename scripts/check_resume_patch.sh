#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/workstation_formal.yaml}"
ENV_NAME="${FOUNDATION_ENV_NAME:-foundation-model-test}"

exec conda run --no-capture-output --name "$ENV_NAME" \
  python scripts/check_resume_patch.py --config "$CONFIG"
