#!/usr/bin/env bash
# Clean GPU environment setup for the Caduceus-Ph stack on an HPC (e.g. TSCC).
#
# Builds mamba-ssm + causal-conv1d FROM SOURCE against the installed torch. This
# makes the kernels' C++ ABI match torch automatically (avoiding the prebuilt
# wheel mismatch) and pip resolves a mutually-compatible mamba/causal pair, so
# Mamba's fused fast path works.
#
# PREREQUISITES (do these first, on a GPU node):
#   1. conda available.
#   2. A CUDA 12.x toolkit on PATH so `nvcc` is present and matches torch's cu121
#      build (CUDA major version must match to compile the extensions):
#          module avail cuda
#          module load cuda/12.4      # any 12.x; NOT 11.x or 13.x
#
# Usage:
#   bash setup_gpu_env.sh [env_name]        # default env name: caduceus
#
# Optional env vars:
#   MAX_JOBS   parallel compile jobs (default 4; lower if the build OOMs)

set -eo pipefail

ENV_NAME="${1:-caduceus}"
PYTHON_VERSION="3.11"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Checking prerequisites"
command -v conda >/dev/null || { echo "ERROR: conda not found on PATH."; exit 1; }
if ! command -v nvcc >/dev/null; then
	echo "ERROR: nvcc not found. Load a CUDA 12.x toolkit first, e.g.:"
	echo "         module avail cuda"
	echo "         module load cuda/12.4"
	exit 1
fi
NVCC_MAJOR="$(nvcc --version | grep -oP 'release \K[0-9]+' | head -1)"
if [ "${NVCC_MAJOR}" != "12" ]; then
	echo "ERROR: nvcc is CUDA ${NVCC_MAJOR}.x, but torch is built for cu121."
	echo "       Load a 12.x toolkit (module load cuda/12.x) and re-run, so the"
	echo "       compiled extensions match torch's CUDA major version."
	exit 1
fi
echo "    nvcc $(nvcc --version | grep -oP 'release \K[0-9.]+' | head -1) OK"

echo "==> Creating conda env '${ENV_NAME}' (python ${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

echo "==> Installing torch + transformers + project deps"
# Installs torch FIRST (required before building the kernels below).
python -m pip install -r "${REPO_ROOT}/requirements_gpu.txt"

echo "==> Building causal-conv1d and mamba-ssm from source (ABI matches torch)"
export MAX_JOBS="${MAX_JOBS:-4}"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
python -m pip install packaging ninja setuptools wheel
python -m pip install --no-build-isolation causal_conv1d
python -m pip install --no-build-isolation mamba_ssm

echo "==> Installing runtime library-path hook (torch + nvidia libs)"
# Compiled extensions link libc10.so / libcudart.so.12; make torch's bundled libs
# discoverable on every `conda activate` (incl. inside SLURM job scripts).
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/torch_ld_library_path.sh" << 'EOF'
_torch_libdirs="$(python -c '
import os, glob, torch
sp = os.path.dirname(os.path.dirname(torch.__file__))
d = [os.path.join(os.path.dirname(torch.__file__), "lib")]
d += sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(":".join(d))
')"
export LD_LIBRARY_PATH="${_torch_libdirs}:${LD_LIBRARY_PATH:-}"
unset _torch_libdirs
EOF
conda deactivate
conda activate "${ENV_NAME}"

echo "==> Verifying"
python -c "import torch, mamba_ssm, causal_conv1d; from mamba_ssm import Mamba; \
print('OK | torch', torch.__version__, '| transformers', __import__('transformers').__version__, \
'| mamba', mamba_ssm.__version__, '| causal', causal_conv1d.__version__, \
'| gpu', torch.cuda.is_available())"

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Smoke test:           pytest seq_models/caduceus/ -v"
echo "Dev fine-tune:        python -m seq_models.caduceus.fine_tune \\"
echo "                        --config scripts/training/training_configs/caduceus/dev.yaml \\"
echo "                        --output_dir scripts/training/output/caduceus"
