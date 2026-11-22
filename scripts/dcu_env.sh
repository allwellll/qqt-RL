#!/usr/bin/env bash
# Source from any directory. Only this shell and its children are affected.
_qqt_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$_qqt_root/.venv/bin/python" ]]; then
  echo 'Missing project .venv; see docs/dcu_migration.md' >&2
  return 1
fi
if [[ ! -r /opt/mpi/lib/libmpi.so ]]; then
  echo 'Missing DTK OpenMPI library /opt/mpi/lib/libmpi.so' >&2
  return 1
fi
source "$_qqt_root/.venv/bin/activate"
# DTK hipfftMp references MPI symbols without a DT_NEEDED entry.
export LD_PRELOAD="/opt/mpi/lib/libmpi.so${LD_PRELOAD:+:$LD_PRELOAD}"
export JAX_PLATFORMS=rocm
export JAX_PLATFORM_NAME=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$_qqt_root/.jax_cache/dcu"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$_qqt_root/.cache/tmp"
export PIP_CACHE_DIR="$_qqt_root/.cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
unset _qqt_root
