#!/usr/bin/env python3
"""Run predict.sh over many CIFs with several concurrent slots.

Each CIF is an ordinary independent predict.sh run (same code path and options, its own work
directory); only the scheduling differs. Slots get disjoint CPU ranges, so concurrent runs do not
share the 8 CPUs that a lone predict.sh pins by default, and GPUs are assigned round-robin.
A single relaxation leaves most of a GPU idle, so several slots per GPU raise throughput.

usage: python3 predict_batch.py --cifs a.cif b.cif ... --out DIR [--slots N] [--gpus 0,1]
                                [--cpus-per-slot 6] [-- extra predict.sh options]
       python3 predict_batch.py --cif-list list.txt --out DIR ...
Writes DIR/<label>/prediction.json, DIR/<label>/run.log and DIR/summary.csv. Existing
DIR/<label>/prediction.json files are kept and not recomputed.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'acceleration'))
from ehull_protocol import from_environment
FEATURES = ("solvent_removal_stability", "RMSD_[A]", "Linker_HOMO_[eV]", "sascore", "Ehull")


def reusable_prediction(doc, cif, extra):
    if '--baseline' in extra:
        return True  # Explicit historical route, outside the 3.1.1 protocol.
    def option(name, default):
        value = default
        for i, token in enumerate(extra):
            if token == name and i + 1 < len(extra):
                value = extra[i + 1]
            elif token.startswith(name + '='):
                value = token.split('=', 1)[1]
        return value
    detail = doc.get('details', {}).get('ehull', {})
    expected = dict(requested_model=option('--ehull-model', os.environ.get('EHULL_MODEL') or 'uma-s-1p1'),
                    solver=option('--ehull-solver', 'chol'),
                    inference=option('--ehull-inference', 'default'), energy_type='ODAC_MLIP')
    return (doc.get('package_version') == '3.1.1'
            and detail.get('convergence_protocol') == from_environment()
            and detail.get('relaxation_converged') is True
            and detail.get('run_configuration') == expected
            and detail.get('input_sha256') == hashlib.sha256(cif.read_bytes()).hexdigest())


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cifs", nargs="*", default=[])
    ap.add_argument("--cif-list", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--gpus", default="0", help="comma-separated CUDA device indices")
    ap.add_argument("--cpus-per-slot", type=int, default=6)
    a = ap.parse_args(argv)
    cifs = [Path(c).resolve() for c in a.cifs]
    if a.cif_list:
        cifs += [Path(line.strip()).resolve() for line in a.cif_list.read_text().splitlines() if line.strip()]
    labels = [c.stem for c in cifs]
    if len(set(labels)) != len(labels):
        raise SystemExit("CIF file names must be unique (they become the run labels)")
    allowed = sorted(os.sched_getaffinity(0))
    if a.slots < 1 or a.cpus_per_slot < 1 or a.slots * a.cpus_per_slot > len(allowed):
        raise SystemExit(f"--slots x --cpus-per-slot must fit the {len(allowed)} allowed CPUs")
    gpus = [g for g in a.gpus.split(",") if g]
    a.out.mkdir(parents=True, exist_ok=True)

    free = list(range(a.slots))
    lock = threading.Lock()

    def one(cif):
        label = cif.stem
        d = a.out / label
        if (d / "prediction.json").exists():
            doc = json.loads((d / "prediction.json").read_text())
            if reusable_prediction(doc, cif, extra):
                return label, 0, 0.0
            print(f'{label}: existing prediction does not certify the requested 3.1.1 Ehull protocol/input; use a fresh --out directory.', flush=True)
            return label, 3, 0.0
        d.mkdir(exist_ok=True)
        with lock:
            slot = free.pop(0)
        try:
            cpus = allowed[-(slot + 1) * a.cpus_per_slot:][:a.cpus_per_slot]
            cmd = ["bash", str(ROOT / "predict.sh"), "--cif", str(cif), "--label", label,
                   "--work", str(d / "work"), "--out", str(d / "prediction.json"),
                   "--cpu-list", ",".join(map(str, cpus)), "--gpu-id", gpus[slot % len(gpus)], *extra]
            start = time.monotonic()
            with (d / "run.log").open("w") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            return label, rc, time.monotonic() - start
        finally:
            with lock:
                free.append(slot)

    start = time.monotonic()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.slots) as pool:
        for label, rc, wall in pool.map(one, cifs):
            p = a.out / label / "prediction.json"
            doc = json.loads(p.read_text()) if p.exists() and rc == 0 else {}
            feats = doc.get("features", {})
            rows.append({"label": label, "returncode": rc, "wall_s": round(wall, 2),
                         **{k: feats.get(k) for k in FEATURES},
                         "native_score": doc.get("native_score"), "logit": doc.get("logit")})
            print(f"{label}: rc={rc} {wall:.1f}s score={doc.get('native_score')}", flush=True)
    with (a.out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} CIFs, {sum(r['returncode'] == 0 for r in rows)} succeeded, "
          f"makespan {time.monotonic() - start:.1f}s -> {a.out / 'summary.csv'}")
    return 0 if all(r["returncode"] == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
