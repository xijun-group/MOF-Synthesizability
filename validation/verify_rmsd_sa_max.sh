#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_prefix="$(realpath "${1:?Usage: verify_rmsd_sa_max.sh RUNTIME_PREFIX}")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
# Same private compiled MOFid paths as launch.py; never change shared /tmp links.
exec bwrap --die-with-parent --unshare-pid --bind / / --dev-bind /dev /dev \
  --proc /proc --tmpfs /tmp --ro-bind "$task_prefix/mofid-build" /tmp/mofid-build -- \
  "$task_prefix/mofsynth-env/bin/python" "$task_root/validation/verify_rmsd_sa_max.py" "$task_prefix"
