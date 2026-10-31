#!/usr/bin/env bash
# usage: run_sbu.sh <mofid-build dir> <cif> <outdir> [wrapper...]  -- runs sbu with the build bound at /tmp/mofid-build
set -euo pipefail
B=$(readlink -f "$1"); CIF=$(readlink -f "$2"); OUT=$(readlink -m "$3"); shift 3
mkdir -p "$OUT"
exec bwrap --die-with-parent --ro-bind / / --dev-bind /dev /dev --proc /proc --tmpfs /tmp \
  --bind "$OUT" "$OUT" --ro-bind "$B" /tmp/mofid-build --bind /home/sun07ao/syn_perf_20261023/v3 /home/sun07ao/syn_perf_20261023/v3 -- \
  env OMP_NUM_THREADS=1 LC_ALL=C "$@" /tmp/mofid-build/bin/sbu "$CIF" "$OUT"
