#!/usr/bin/env bash
# GPU setup for the Caduceus-Ph stack, run INSIDE an already-active conda env.
#
# Installs the Caduceus authors' tested kernel triple (from their caduceus_env.yml):
#   torch 2.2.0+cu118 + mamba-ssm 1.2.0.post1 + causal-conv1d 1.2.0.post2
# These mamba/causal wheels were built against torch 2.2.0 exactly, so there's no
# ABI/symbol drift. PREBUILT: no CUDA module, no nvcc, no compiling.
#
# PREREQUISITES (on a GPU node, from the repo root):
#   - A FRESH python 3.11 conda env is ACTIVE (this changes torch 2.x / CUDA 12->11,
#     so a clean env avoids leftover cu12 packages):
#         conda create -y -n caduceus python=3.11 && conda activate caduceus
#   - No 'module load cuda' needed (cu118 runs on the modern driver).
#
# Usage:  bash setup_gpu_env.sh

set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prebuilt kernel wheels matched to torch 2.2.0 (cp311 / cu118 / abiFALSE).
CAUSAL_WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu118torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
MAMBA_WHL="https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu118torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# torch's bundled lib dirs (no torch import needed).
_torch_libdirs() {
	local sp ld d
	sp="$(ls -d "${CONDA_PREFIX}"/lib/python*/site-packages 2>/dev/null | head -1 || true)"
	ld="${sp}/torch/lib"
	for d in "${sp}"/nvidia/*/lib; do
		[ -d "${d}" ] && ld="${ld}:${d}"
	done
	printf '%s' "${ld}"
}

echo "==> Checking prerequisites"
if [ -z "${CONDA_PREFIX:-}" ]; then
	echo "ERROR: no active conda env (CONDA_PREFIX unset). Run 'conda activate <env>' first."
	exit 1
fi
echo "    active env: ${CONDA_PREFIX}"

echo "==> Installing torch 2.2.0 + transformers + project deps (numpy<2 / pandas<3)"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements_gpu.txt"

echo "==> Installing prebuilt mamba/causal kernels (--no-deps)"
python -m pip uninstall -y causal_conv1d mamba_ssm || true
python -m pip install --no-deps "${CAUSAL_WHL}"
python -m pip install --no-deps "${MAMBA_WHL}"
python -m pip install einops   # mamba runtime dep (skipped by --no-deps)

echo "==> Installing runtime library-path hook (torch + nvidia libs)"
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/torch_ld_library_path.sh" << 'EOF'
# Put torch's bundled libs first so compiled extensions find libc10.so /
# libcudart. Pure shell: does not import torch.
_sp="$(ls -d "${CONDA_PREFIX}"/lib/python*/site-packages 2>/dev/null | head -1 || true)"
_ld="${_sp}/torch/lib"
for _d in "${_sp}"/nvidia/*/lib; do
	[ -d "${_d}" ] && _ld="${_ld}:${_d}"
done
export LD_LIBRARY_PATH="${_ld}:${LD_LIBRARY_PATH:-}"
unset _sp _ld _d
EOF
# Apply to the current shell for the verify step.
export LD_LIBRARY_PATH="$(_torch_libdirs):${LD_LIBRARY_PATH:-}"

echo "==> Verifying"
python -c "import torch, numpy, mamba_ssm, causal_conv1d; from mamba_ssm import Mamba; \
print('OK | torch', torch.__version__, '| numpy', numpy.__version__, \
'| mamba', mamba_ssm.__version__, '| causal', causal_conv1d.__version__, \
'| gpu', torch.cuda.is_available())"

echo
echo "Done. Open a fresh shell (or reactivate) so the LD hook loads, then:"
echo "  pytest seq_models/caduceus/ -v"
echo "  python -m seq_models.caduceus.fine_tune \\"
echo "    --config scripts/training/training_configs/caduceus/dev.yaml \\"
echo "    --output_dir scripts/training/output/caduceus"
