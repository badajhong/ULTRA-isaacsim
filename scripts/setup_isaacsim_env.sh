#!/usr/bin/env bash
# Create the ULTRA conda environment for Isaac Sim 5.1.0 + Isaac Lab 2.3.2 (Python 3.11, CUDA 12.8 torch).
#
# Usage (from anywhere):
#   scripts/setup_isaacsim_env.sh
# Optional environment variables:
#   ENV_NAME       conda env name                         (default: ultra-isaacsim)
#   ISAACLAB_PATH  existing or target Isaac Lab checkout  (default: <repo>/../IsaacLab)
#   TORCH_INDEX    PyTorch wheel index                    (default: https://download.pytorch.org/whl/cu128)
#   ACCEPT_EULA=1  accept the NVIDIA Omniverse EULA for this Isaac Sim install non-interactively
#
# The cu128 torch build is required for RTX 50xx / RTX PRO Blackwell GPUs and also works on older GPUs.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-ultra-isaacsim}"
ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO/../IsaacLab}"
ISAACLAB_VERSION="v2.3.2"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
CONSTRAINTS="$REPO/scripts/isaacsim_constraints.txt"

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] conda env '$ENV_NAME' already exists, reusing it"
else
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
# 1. torch first, so that Isaac Sim / Isaac Lab do not pull a CUDA build without Blackwell kernels.
pip install torch==2.7.0 torchvision==0.22.0 --index-url "$TORCH_INDEX"
# 2. Isaac Sim 5.1.0 (pip distribution).
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
# 3. Isaac Lab 2.3.2 core package (editable install from a git checkout).
if [ ! -d "$ISAACLAB_PATH/source/isaaclab" ]; then
  git clone --branch "$ISAACLAB_VERSION" --depth 1 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
fi
pip install -e "$ISAACLAB_PATH/source/isaaclab" -c "$CONSTRAINTS"
# 4. ULTRA dependencies (the constraints keep the versions Isaac Sim pins).
pip install -r "$REPO/requirements.txt" -c "$CONSTRAINTS"

if [ "${ACCEPT_EULA:-0}" = "1" ]; then
  # locate the package without importing it (importing isaacsim would ask for the EULA interactively)
  KIT_DIR="$(python -c 'import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec("isaacsim").origin), "kit"))')"
  echo "yes" > "$KIT_DIR/EULA_ACCEPTED"
  echo "[setup] NVIDIA Omniverse EULA accepted for $KIT_DIR"
fi

python - <<'EOF'
import torch, numpy
print("[setup] torch", torch.__version__, "cuda", torch.version.cuda, "numpy", numpy.__version__)
print("[setup] GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
EOF
echo "[setup] done. Activate with: conda activate $ENV_NAME"
