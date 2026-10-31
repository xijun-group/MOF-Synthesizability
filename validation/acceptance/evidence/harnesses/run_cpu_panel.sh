#!/usr/bin/env bash
# v3 CPU four-feature panel on the five validation MOFs (no Ehull), fresh work dirs.
# usage: run_cpu_panel.sh <package dir> <outroot> [extra predict args...]
set -uo pipefail
PKG=$1; OUT=$2; shift 2
mkdir -p "$OUT"
for m in reference58 gen_49320289d14b3171 gen_eed3ec90c711abe1 gen_2aed42338cd9ddfb gen_5b466e79428854e0; do
  w=$OUT/$m; rm -rf "$w"; mkdir -p "$w"
  s=$(date +%s.%N)
  bash "$PKG/predict.sh" --cif "$HOME/syn_perf_20261023/v3/inputs/$m.cif" --label "$m" --work "$w/work" --skip ehull "$@" > "$w/stdout-stderr.log" 2>&1
  rc=$?
  echo "{\"label\": \"$m\", \"rc\": $rc, \"wall_s\": $(echo "$(date +%s.%N) - $s" | bc)}" > "$w/trial.json"
done
echo done > "$OUT/panel.done"
