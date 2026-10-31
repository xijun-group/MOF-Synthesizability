#!/usr/bin/env bash
# Noise-floor / candidate matrix for the Ehull relaxation on one GPU.
# usage: run_matrix.sh <python> <gpu-id> <lanes> <outroot> [jobs file: "<mof> <solver> <infer> <rep>" lines]
# Jobs whose bench.json exists are skipped (resumable). Writes <outroot>/matrix.done at the end.
set -uo pipefail
PY=$1; GPU=$2; LANES=$3; OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd); IN=$HERE/../inputs
mkdir -p "$OUT"
JOBS=$OUT/jobs.txt; : > "$JOBS"
if [ -n "${5:-}" ]; then cat "$5" >> "$JOBS"; else
for m in reference58 gen_49320289d14b3171 gen_eed3ec90c711abe1 gen_2aed42338cd9ddfb gen_5b466e79428854e0; do
  for v in "orig default 1" "chol default 1" "orig nockpt_merge 1" "chol nockpt_merge 1" "orig default 2"; do
    echo "$m $v" >> "$JOBS"
  done
done
fi
run_job() {
  set -- $1
  d=$OUT/$1/$2_$3_r$4
  [ -f "$d/bench.json" ] && return 0
  rm -rf "$d"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    timeout 7200 "$PY" "$HERE/relax_bench.py" "$IN/$1.cif" "$1" "$d" --solver "$2" --infer "$3" > "$d/stdout.log" 2> "$d/stderr.log"
  rc=$?
  echo "$(date +%FT%T) rc=$rc $1 $2 $3 r$4" >> "$OUT/progress.log"
}
export -f run_job; export OUT PY GPU HERE IN
xargs -a "$JOBS" -d '\n' -P "$LANES" -I{} bash -c 'run_job "$@"' _ {}
echo "done $(date +%FT%T)" > "$OUT/matrix.done"
