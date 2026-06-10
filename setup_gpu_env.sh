#!/usr/bin/env bash
# GPU setup for the Caduceus-Ph stack, run INSIDE an already-active conda env.
#
# Builds mamba-ssm + causal-conv1d FROM SOURCE against the env's torch. Compiling
# locally makes the kernels' C++ ABI match torch automatically (the prebuilt
# wheels have no matching abiFALSE + torch-2.5 + cpp_functions combo) and pip
# resolves a mutually-compatible mamba/causal pair, so Mamba's fused fast path
# works.
#
# PREREQUISITES (on a GPU node, from the repo root):
#   1. Your target conda env (python 3.11) is ACTIVE:  conda activate <env>
#   2. A CUDA 12.x toolkit is loaded so `nvcc` is on PATH. On TSCC:
#          module load cuda12.0/toolkit/12.0.1
#      (You don't need this to RUN torch — only to COMPILE these kernels.)
#
# Usage:
#   bash setup_gpu_env.sh
#
# Optional env vars:
#   MAX_JOBS   parallel compile jobs (default 4; lower if the build OOMs)

set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Checking prerequisites"
if [ -z "${CONDA_PREFIX:-}" ]; then
	echo "ERROR: no active conda env (CONDA_PREFIX unset). Run 'conda activate <env>' first."
	exit 1
fi
echo "    active env: ${CONDA_PREFIX}"
if ! command -v nvcc >/dev/null; then
	echo "ERROR: nvcc not found. Load a CUDA 12.x toolkit, e.g. on TSCC:"
	echo "         module load cuda12.0/toolkit/12.0.1"
	exit 1
fi
NVCC_MAJOR="$(nvcc --version | grep -oP 'release \K[0-9]+' | head -1)"
if [ "${NVCC_MAJOR}" != "12" ]; then
	echo "ERROR: nvcc is CUDA ${NVCC_MAJOR}.x, but torch is built for cu121."
	echo "       Load a 12.x toolkit so the compiled extensions match torch's CUDA major."
	exit 1
fi
echo "    nvcc $(nvcc --version | grep -oP 'release \K[0-9.]+' | head -1) OK"

echo "==> Installing torch + transformers + project deps"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements_gpu.txt"

echo "==> Building causal-conv1d and mamba-ssm from source (ABI matches torch)"
# Remove any prebuilt installs so pip actually rebuilds from the PyPI sdist.
python -m pip uninstall -y causal_conv1d mamba_ssm || true
export MAX_JOBS="${MAX_JOBS:-4}"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
python -m pip install packaging ninja setuptools wheel
python -m pip install --no-build-isolation --no-binary=causal_conv1d causal_conv1d
python -m pip install --no-build-isolation --no-binary=mamba_ssm mamba_ssm

echo "==> Installing runtime library-path hook (torch + nvidia libs)"
# Compiled extensions link libc10.so / libcudart.so.12; make torch's bundled libs
# discoverable on every 'conda activate' (incl. inside SLURM job scripts).
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
# Apply to the current shell too, so the verify step below can load the kernels.
export LD_LIBRARY_PATH="$(python -c '
import os, glob, torch
sp = os.path.dirname(os.path.dirname(torch.__file__))
d = [os.path.join(os.path.dirname(torch.__file__), "lib")]
d += sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(":".join(d))
'):${LD_LIBRARY_PATH:-}"

echo "==> Verifying"
python -c "import torch, mamba_ssm, causal_conv1d; from mamba_ssm import Mamba; \
print('OK | torch', torch.__version__, '| transformers', __import__('transformers').__version__, \
'| mamba', mamba_ssm.__version__, '| causal', causal_conv1d.__version__, \
'| gpu', torch.cuda.is_available())"

echo
echo "Done. Open a fresh shell (or 'conda deactivate && conda activate <env>') so the"
echo "LD_LIBRARY_PATH hook is loaded automatically, then:"
echo "  pytest seq_models/caduceus/ -v"
echo "  python -m seq_models.caduceus.fine_tune \\"
echo "    --config scripts/training/training_configs/caduceus/dev.yaml \\"
echo "    --output_dir scripts/training/output/caduceus"
