#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MRI_PET_GEOMC_PYTHON:-python}"
CONFIG_PATH="${MRI_PET_GEOMC_CONFIG:-${PROJECT_ROOT}/configs/server.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Server config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m pip check

# Extra arguments are forwarded. Reusing the same --launch-dir (or the fixed
# run.launch_dir in server.yaml) reuses stages only when their content-hash-checked records match.
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/run_all.py" full \
  --config "${CONFIG_PATH}" "$@"
