#!/usr/bin/env bash
# GPU setup for the Caduceus-Ph stack, run INSIDE an already-active conda env.
#
# Uses PREBUILT mamba-ssm + causal-conv1d wheels matched to torch's C++ ABI.
# NO CUDA module, NO nvcc, NO compiling. torch 2.6 is used because both projects
# publish matched fused-capable cu12torch2.6 wheels in both ABIs (torch 2.5 did
# not, which is what forced the source build before).
#
# PREREQUISITES (on a GPU node, from the repo root):
#   - Your target conda env (python 3.11) is ACTIVE:  conda activate <env>
#   - Driver supports CUDA 12.4 (TSCC A100 / driver 580 does).
#   - You do NOT need 'module load cuda'. (If one is loaded, this script still
#     works — it forces torch's own libs ahead of any system CUDA libs.)
#
# Usage:  bash setup_gpu_env.sh

set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Matched, fused-capable pair (both expose causal-conv1d's cpp_functions API).
# TORCH_TAG must match the torch version pinned in requirements_gpu.txt.
MAMBA_VER="2.3.2.post1"
CAUSAL_VER="1.6.2.post1"
TORCH_TAG="cu12torch2.6"

# Locate torch's bundled lib dirs WITHOUT importing torch (torch may fail to
# import while a system CUDA module shadows its libs).
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

echo "==> Installing torch + transformers + project deps"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements_gpu.txt"

echo "==> Putting torch's bundled CUDA libs first on LD_LIBRARY_PATH"
# Ensures 'import torch' below works even if a system CUDA module is loaded.
export LD_LIBRARY_PATH="$(_torch_libdirs):${LD_LIBRARY_PATH:-}"

echo "==> Selecting prebuilt mamba/causal wheels matching torch's ABI"
ABI="$(python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"
PYTAG="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
echo "    torch ABI: cxx11abi${ABI} | python: ${PYTAG}"
CAUSAL_WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_VER}/causal_conv1d-${CAUSAL_VER}+${TORCH_TAG}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
MAMBA_WHL="https://github.com/state-spaces/mamba/releases/download/v${MAMBA_VER}/mamba_ssm-${MAMBA_VER}+${TORCH_TAG}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"

echo "==> Installing prebuilt kernels"
python -m pip uninstall -y causal_conv1d mamba_ssm || true
python -m pip install "${CAUSAL_WHL}"
python -m pip install "${MAMBA_WHL}"

echo "==> Installing runtime library-path hook (torch + nvidia libs)"
# Compiled extensions link libc10.so / libcudart.so.12; put torch's bundled libs
# first so they're found on every 'conda activate' (incl. inside SLURM jobs).
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/torch_ld_library_path.sh" << 'EOF'
# Pure shell (does not import torch).
_sp="$(ls -d "${CONDA_PREFIX}"/lib/python*/site-packages 2>/dev/null | head -1 || true)"
_ld="${_sp}/torch/lib"
for _d in "${_sp}"/nvidia/*/lib; do
	[ -d "${_d}" ] && _ld="${_ld}:${_d}"
done
export LD_LIBRARY_PATH="${_ld}:${LD_LIBRARY_PATH:-}"
unset _sp _ld _d
EOF

echo "==> Verifying"
python -c "import torch, mamba_ssm, causal_conv1d; from mamba_ssm import Mamba; \
print('OK | torch', torch.__version__, '| transformers', __import__('transformers').__version__, \
'| mamba', mamba_ssm.__version__, '| causal', causal_conv1d.__version__, \
'| gpu', torch.cuda.is_available())"

echo
echo "Done. Open a fresh shell (or 'conda deactivate && conda activate <env>') so the"
echo "LD_LIBRARY_PATH hook loads automatically, then:"
echo "  pytest seq_models/caduceus/ -v"
echo "  python -m seq_models.caduceus.fine_tune \\"
echo "    --config scripts/training/training_configs/caduceus/dev.yaml \\"
echo "    --output_dir scripts/training/output/caduceus"
