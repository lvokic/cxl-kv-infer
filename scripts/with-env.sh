#!/usr/bin/env bash
# Run a command with the repository's Python and CUDA toolchain.
# Usage: bash scripts/with-env.sh python ...
set -euo pipefail

replay_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$replay_root/.venv/bin/python" ]]; then
    echo "Missing .venv; follow docs/DEVELOPMENT.md first." >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/with-env.sh COMMAND [ARG ...]" >&2
    exit 2
fi

replay_mode="cuda"
if [[ "$1" == "--cpu" || "$1" == "--cuda" ]]; then
    replay_mode="${1#--}"
    shift
    if [[ $# -eq 0 ]]; then
        echo "Usage: bash scripts/with-env.sh [--cpu|--cuda] COMMAND [ARG ...]" >&2
        exit 2
    fi
fi

source "$replay_root/.venv/bin/activate"
if [[ "$replay_mode" == "cpu" ]]; then
    exec "$@"
fi
replay_site="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
if [[ ! -x "$replay_site/nvidia/cu13/bin/nvcc" ]]; then
    echo "Missing local CUDA 13.0 toolchain; install requirements/gpu-cu130.lock." >&2
    exit 1
fi
export CUDA_HOME="$replay_site/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
# NVIDIA pip wheels provide lib/libcudart.so.13, while FlashInfer links -lcudart
# and assumes a toolkit lib64 directory. Keep the linker alias outside the wheel.
replay_link_dir="$replay_root/artifacts/cuda-link"
mkdir -p "$replay_link_dir"
if [[ -e "$replay_link_dir/libcudart.so" || -L "$replay_link_dir/libcudart.so" ]]; then
    if [[ ! "$replay_link_dir/libcudart.so" -ef "$CUDA_HOME/lib/libcudart.so.13" ]]; then
        echo "Unexpected libcudart linker alias in $replay_link_dir" >&2
        exit 1
    fi
else
    ln -s "$CUDA_HOME/lib/libcudart.so.13" "$replay_link_dir/libcudart.so"
fi
export LIBRARY_PATH="$replay_link_dir${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Compile for the A100; do not probe other jobs or change NUMA placement.
export TORCH_CUDA_ARCH_LIST="8.0"
export FLASHINFER_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASHINFER_WORKSPACE_BASE="$replay_root/artifacts/flashinfer"
export TORCH_EXTENSIONS_DIR="$replay_root/artifacts/torch_extensions"

exec "$@"
