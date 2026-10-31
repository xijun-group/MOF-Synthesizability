#!/usr/bin/env bash
# Full five-feature pipeline: original (--baseline) vs current default, five validation MOFs,
# alternating order, fresh and equal-length work paths (MOFid output depends on path length).
# usage: bench_cur.sh <package dir> <gpu-id> <outroot> [extra predict args...]
set -uo pipefail
PKG=$1; GPU=$2; OUT=$3; shift 3
IN=$HOME/syn_perf_20261023/v3/inputs
mkdir -p "$OUT"
declare -A FLAGS=([orig]="--baseline" [curr]="")
i=0
for m in reference58 gen_49320289d14b3171 gen_eed3ec90c711abe1 gen_2aed42338cd9ddfb gen_5b466e79428854e0; do
  if (( i % 2 == 0 )); then order="orig curr"; else order="curr orig"; fi
  i=$((i + 1))
  for mode in $order; do
    d=$OUT/$m/$mode
    [ -f "$d/trial.json" ] && continue
    mkdir -p "$d"
    { uptime; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader; } > "$d/load_before.txt" 2>&1
    s=$(date +%s.%N)
    # shellcheck disable=SC2086
    bash "$PKG/predict.sh" ${FLAGS[$mode]} --cif "$IN/$m.cif" --label "$m" --work "$d/work" --gpu-id "$GPU" "$@" \
      > "$d/stdout-stderr.log" 2>&1
    rc=$?
    w=$(echo "$(date +%s.%N) - $s" | bc)
    echo "{\"label\": \"$m\", \"mode\": \"$mode\", \"rc\": $rc, \"wall_s\": $w, \"order\": \"$order\"}" > "$d/trial.json"
    echo "$(date +%FT%T) $m $mode rc=$rc wall=$w" >> "$OUT/progress.log"
  done
done
echo "done $(date +%FT%T)" > "$OUT/bench.done"
