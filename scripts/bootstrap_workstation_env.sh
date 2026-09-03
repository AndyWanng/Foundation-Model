#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="${FOUNDATION_ENV_NAME:-foundation-model-test}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda/mamba is required for the bundled dcm2niix environment." >&2
  exit 2
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env create --name "$ENV_NAME" --file environment.workstation.yml
else
  conda env update --name "$ENV_NAME" --file environment.workstation.yml --prune
fi

conda run --no-capture-output --name "$ENV_NAME" \
  python -m pip install --requirement requirements.txt
conda run --no-capture-output --name "$ENV_NAME" \
  python -m pip install --no-deps --editable .
conda run --no-capture-output --name "$ENV_NAME" python -m pip check
conda run --no-capture-output --name "$ENV_NAME" dcm2niix --version
conda run --no-capture-output --name "$ENV_NAME" python -c \
  'import torch; assert torch.__version__.split("+", 1)[0] == "2.11.0", torch.__version__; assert torch.version.cuda == "12.8", torch.version.cuda; assert torch.cuda.is_available(), "CUDA is unavailable"; assert torch.cuda.device_count() >= 1, "no visible GPU"; assert torch.cuda.is_bf16_supported(), "cuda:0 lacks BF16"; print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu_count": torch.cuda.device_count(), "cuda0": torch.cuda.get_device_name(0), "bf16": True})'

echo "Environment ready: $ENV_NAME"
echo "Provision the local-only assets listed in BUNDLED_ASSETS.json."
echo "Edit configs/workstation_formal.yaml raw-data paths before preprocessing."
