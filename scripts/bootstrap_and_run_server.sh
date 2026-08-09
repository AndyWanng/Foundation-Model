#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SYSTEM_PYTHON="${MRI_PET_GEOMC_BOOTSTRAP_PYTHON:-python3}"
VENV_ROOT="${MRI_PET_GEOMC_VENV:-${PROJECT_ROOT}/.venv_geomc}"
REQUIREMENTS="${PROJECT_ROOT}/requirements.txt"
STAMP="${VENV_ROOT}/.requirements.sha256"

if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "requirements.txt is missing: ${REQUIREMENTS}" >&2
  exit 2
fi

"${SYSTEM_PYTHON}" -c 'import sys; version=sys.version_info[:2]; assert (3, 10) <= version < (3, 13), f"GeoMC requires Python 3.10-3.12, got {sys.version.split()[0]}. Set MRI_PET_GEOMC_BOOTSTRAP_PYTHON to a compatible interpreter."'

REQUIREMENTS_SHA="$("${SYSTEM_PYTHON}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${REQUIREMENTS}")"
INSTALLED_SHA=""
if [[ -f "${STAMP}" ]]; then
  INSTALLED_SHA="$(tr -d '\r\n' < "${STAMP}")"
fi

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  "${SYSTEM_PYTHON}" -m venv "${VENV_ROOT}"
fi

if [[ "${INSTALLED_SHA}" != "${REQUIREMENTS_SHA}" ]]; then
  "${VENV_ROOT}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV_ROOT}/bin/python" -m pip install --requirement "${REQUIREMENTS}"
  "${VENV_ROOT}/bin/python" -m pip check
  printf '%s\n' "${REQUIREMENTS_SHA}" > "${STAMP}"
fi

# Source can change without requirements changing. Refresh the editable package
# on every invocation, but do not resolve or mutate the dependency environment.
"${VENV_ROOT}/bin/python" -m pip install --force-reinstall --no-deps --editable "${PROJECT_ROOT}"
"${VENV_ROOT}/bin/python" -m pip check

SOURCE_VERSION="$(awk -F '"' '/^version = "/ {print $2; exit}' "${PROJECT_ROOT}/pyproject.toml")"
INSTALLED_VERSION="$("${VENV_ROOT}/bin/python" -c 'from importlib.metadata import version; print(version("mri-pet-geomc"))')"
IMPORT_VERSION="$("${VENV_ROOT}/bin/python" -c 'import mri_pet_geomc; print(mri_pet_geomc.__version__)')"
printf 'source_version=%s installed_version=%s import_version=%s project_root=%s\n' \
  "${SOURCE_VERSION}" "${INSTALLED_VERSION}" "${IMPORT_VERSION}" "${PROJECT_ROOT}"
if [[ "${SOURCE_VERSION}" != "${INSTALLED_VERSION}" || "${SOURCE_VERSION}" != "${IMPORT_VERSION}" ]]; then
  echo "Package version metadata does not match the synchronized source." >&2
  exit 2
fi

export MRI_PET_GEOMC_PYTHON="${VENV_ROOT}/bin/python"
exec bash "${SCRIPT_DIR}/run_server_full.sh" "$@"
