#!/usr/bin/env python3
"""Canonical per-identity linker QM campaign (Linker_HOMO identity repair).

For every unique standardised linker identity (stereo-free neutral parent
SMILES from the sascore-repair standardiser) run ONE deterministic xTB GFN2
calculation:

    std SMILES --(RDKit ETKDGv3, fixed seed, 16 confs, MMFF/UFF prescreen)-->
    canonical embedded conformer --(xtb --sp)--> SP-preopt HOMO/LUMO
                               --(xtb --opt)--> optimised geometry
                               --(xtb --sp)--> HOMO/LUMO at opt (PRIMARY)

All MOF instances of the same identity share this single calculation, so the
repaired Linker_HOMO is identical for identical linkers by construction, and
the radical/charge-state identity artefact is removed at the source.

Protocol constants (binary, flags, timeout, thread env) mirror the 2026-09-14
in-house MOFSynth-QM rerun (neutral, default spin, no hydrogen modification).

Usage:
  canonical_linker_qm.py run --identities ids.txt --run-dir DIR [--workers N]
                             [--limit K] [--shuffle-seed S]
  canonical_linker_qm.py collect --run-dir DIR --out CSV

Idempotent: identities with a converged result.json are skipped on rerun.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
ROOT = CHAIN.parent
sys.path.insert(0, str(CHAIN))
sys.path.insert(0, str(ROOT / "_deps"))

XTB = os.environ.get("XTB_BIN") or "/home/cheyue/data/SYN-final/tools/xtb_env_v1/bin/xtb"
TIMEOUT_S = 1800
EMBED_SEED = 0xC0FFEE
N_CONFS = 16

HOMO_RE = re.compile(r"^\s*\d+\s+(?:-?[\d.]+\s+){1,2}(-?[\d.]+)\s+\((HOMO|LUMO)\)\s*$")

_ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1", OMP_STACKSIZE="256M")


def identity_key(std_linker: str) -> str:
    return hashlib.sha256(std_linker.encode()).hexdigest()[:16]


def embed_canonical(std_linker: str):
    """Return (xyz_block, info) or raise. Deterministic given RDKit version."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(std_linker)
    if mol is None:
        raise ValueError("std SMILES not parsable by RDKit")
    mol = Chem.AddHs(mol)

    attempts = []
    p = AllChem.ETKDGv3()
    p.randomSeed = EMBED_SEED
    p.numThreads = 1
    attempts.append(("ETKDGv3", p, N_CONFS))
    p = AllChem.ETKDGv3()
    p.randomSeed = EMBED_SEED
    p.numThreads = 1
    p.useRandomCoords = True
    attempts.append(("ETKDGv3_useRandomCoords", p, N_CONFS))
    p = AllChem.ETKDGv3()
    p.randomSeed = EMBED_SEED
    p.numThreads = 1
    p.useRandomCoords = True
    p.ignoreSmoothingFailures = True
    # numConfs kept at 2: on metal macrocycles each distance-geometry attempt
    # can take minutes and RDKit's C++ loop cannot be interrupted mid-call.
    attempts.append(("ETKDGv3_randomCoords_noSmooth", p, 2))

    ids, method = [], None
    for name, params, n_confs in attempts:
        try:
            ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs,
                                                  params=params))
        except Exception:
            ids = []
        if ids:
            method = name
            break
    if not ids:
        raise ValueError("embedding failed")

    def mmff_energy(cid):
        if AllChem.MMFFHasAllMoleculeParams(mol):
            ff = AllChem.MMFFGetMoleculeForceField(
                mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
            return ("MMFF", ff.CalcEnergy()) if ff else None
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        return ("UFF", ff.CalcEnergy()) if ff else None

    scored = [(cid, mmff_energy(cid)) for cid in ids]
    scored = [(cid, e) for cid, e in scored if e is not None]
    if not scored:
        # no force-field support for some atom (e.g. exotic metal): take conf 0
        scored = [(ids[0], ("none", 0.0))]
    force_field = scored[0][1][0]
    cid, (_, energy) = min(scored, key=lambda t: t[1][1])
    xyz = Chem.MolToXYZBlock(mol, confId=cid)
    info = {"embed_method": method, "n_confs": len(ids), "chosen_conf": int(cid),
            "force_field": force_field, "ff_energy": float(energy),
            "embed_seed": EMBED_SEED, "natoms": int(mol.GetNumAtoms())}
    return xyz, info


def run_xtb_redirected(workdir: Path, args: list[str]) -> dict:
    t0 = time.monotonic()
    with open(workdir / "check.out", "wb") as out, open(workdir / "check.err", "wb") as err:
        try:
            proc = subprocess.run(["timeout", "--kill-after=30s", f"{TIMEOUT_S}s",
                                   XTB, *args], cwd=workdir, env=_ENV,
                                  stdout=out, stderr=err)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return {"returncode": rc, "elapsed_s": time.monotonic() - t0}


def parse_homo_lumo(check_out: Path):
    homo = lumo = None
    with check_out.open(errors="replace") as f:
        for line in f:
            mobj = HOMO_RE.match(line)
            if mobj:
                if mobj.group(2) == "HOMO" and homo is None:
                    homo = float(mobj.group(1))
                elif mobj.group(2) == "LUMO" and lumo is None:
                    lumo = float(mobj.group(1))
    if homo is None or lumo is None:
        raise ValueError("HOMO/LUMO not found in check.out")
    return homo, lumo


def parse_total_energy(check_out: Path):
    pat = re.compile(r"::\s+total energy\s+(-?[\d.]+)\s+Eh")
    energy = None
    with check_out.open(errors="replace") as f:
        for line in f:
            mobj = pat.search(line)
            if mobj:
                # --opt prints an initial and a final summary. The first
                # summary is the embedded geometry, not the optimised energy.
                energy = float(mobj.group(1))
    return energy


def run_one(std_linker: str, run_dir: Path) -> dict:
    key = identity_key(std_linker)
    work = run_dir / "linkers" / key
    result_path = work / "result.json"
    if result_path.exists():
        try:
            old = json.loads(result_path.read_text())
            if old.get("status") == "converged":
                old["skipped"] = True
                return old
        except Exception:
            pass
    t_start = time.monotonic()
    work.mkdir(parents=True, exist_ok=True)
    (work / "input.smi").write_text(std_linker + "\n")
    (work / "stage").write_text("embedding")
    res = {"std_linker": std_linker, "identity_key": key, "status": "failed"}
    try:
        xyz, info = embed_canonical(std_linker)
        res.update(info)
        (work / "embed.json").write_text(json.dumps(info, indent=2))
        (work / "linker.xyz").write_text(xyz)

        # SP on the embedded canonical conformer (MOFSynth-style SP record)
        (work / "stage").write_text("sp_preopt")
        sp0 = work / "sp_preopt"
        sp0.mkdir(exist_ok=True)
        (sp0 / "linker.xyz").write_text(xyz)
        r = run_xtb_redirected(sp0, ["linker.xyz", "--sp"])
        res["sp_preopt"] = r
        if r["returncode"] != 0:
            raise ValueError(f"xtb --sp (preopt) rc={r['returncode']}")
        res["homo_sp_preopt_eV"], res["lumo_sp_preopt_eV"] = parse_homo_lumo(
            sp0 / "check.out")
        res["sp_preopt_energy_hartree"] = parse_total_energy(sp0 / "check.out")

        # geometry optimisation
        (work / "stage").write_text("opt")
        opt = work / "opt"
        opt.mkdir(exist_ok=True)
        (opt / "linker.xyz").write_text(xyz)
        r = run_xtb_redirected(opt, ["linker.xyz", "--opt"])
        res["opt"] = r
        xtbopt = opt / "xtbopt.xyz"
        if r["returncode"] != 0 or not xtbopt.exists() or not (opt / ".xtboptok").exists():
            raise ValueError(f"xtb --opt rc={r['returncode']}")
        res["opt_energy_hartree"] = parse_total_energy(opt / "check.out")

        # SP at the optimised geometry (PRIMARY value)
        (work / "stage").write_text("sp")
        sp1 = work / "sp"
        sp1.mkdir(exist_ok=True)
        (sp1 / "linker.xyz").write_bytes(xtbopt.read_bytes())
        r = run_xtb_redirected(sp1, ["linker.xyz", "--sp"])
        res["sp"] = r
        if r["returncode"] != 0:
            raise ValueError(f"xtb --sp rc={r['returncode']}")
        res["homo_eV"], res["lumo_eV"] = parse_homo_lumo(sp1 / "check.out")
        res["sp_energy_hartree"] = parse_total_energy(sp1 / "check.out")
        res["gap_eV"] = res["lumo_eV"] - res["homo_eV"]
        res["status"] = "converged"
    except subprocess.TimeoutExpired:
        res["error"] = "timeout"
    except Exception as e:  # noqa: BLE001
        res["error"] = str(e)[:300]
    res["elapsed_total_s"] = time.monotonic() - t_start
    result_path.write_text(json.dumps(res, indent=2))
    return res


def run_one_timed(std_linker: str, run_dir: Path, task_timeout: int,
                  embed_timeout: int = 240) -> dict:
    """run_one under hard wall-clock caps (stage-aware).

    RDKit distance geometry on metal macrocycles cannot be interrupted from
    Python, and multiprocessing daemon workers may not spawn children, so the
    real work runs as a plain OS grandchild (`compute-one` subcommand) watched
    by the pool worker: killed when the `stage` marker stays at "embedding"
    for more than `embed_timeout` seconds, or when total wall exceeds
    `task_timeout`.  xTB's own `timeout` wrapper caps any orphaned xTB.
    """
    key = identity_key(std_linker)
    result_path = run_dir / "linkers" / key / "result.json"
    if result_path.exists():
        try:
            old = json.loads(result_path.read_text())
            if old.get("status") == "converged":
                old["skipped"] = True
                return old
        except Exception:
            pass
    stage_path = run_dir / "linkers" / key / "stage"
    try:
        stage_path.unlink(missing_ok=True)
    except Exception:
        pass
    t0 = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "compute-one",
         "--smiles", std_linker, "--run-dir", str(run_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        env=dict(os.environ, XTB_BIN=XTB))
    rec = None
    kill_reason = None
    embedding_since = t0
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        elapsed = time.monotonic() - t0
        stage = ""
        try:
            stage = stage_path.read_text().strip()
        except Exception:
            stage = ""
        if stage and stage != "embedding":
            embedding_since = None
        if stage == "embedding" and embedding_since is not None \
                and time.monotonic() - embedding_since > embed_timeout:
            kill_reason = (f"embedding hard-timeout {embed_timeout}s "
                           f"(grandchild killed; RDKit distance geometry did "
                           f"not return on this linker)")
            break
        if elapsed > task_timeout:
            kill_reason = (f"task hard-timeout {task_timeout}s (grandchild "
                           f"killed in stage '{stage or '?'}')")
            break
        time.sleep(3)
    if kill_reason:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
    else:
        try:
            out, _ = proc.communicate(timeout=60)
            line = out.strip().splitlines()[-1] if out and out.strip() else ""
            rec = json.loads(line) if line else None
        except Exception:
            rec = None
    if rec is None:
        rec = {"std_linker": std_linker, "identity_key": key,
               "status": "failed",
               "error": kill_reason or "grandchild exited without a result",
               "elapsed_total_s": time.monotonic() - t0}
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(rec, indent=2))
        except Exception:
            pass
    return rec


def cmd_compute_one(args) -> int:
    rec = run_one(args.smiles, Path(args.run_dir))
    print(json.dumps(rec))
    return 0


def cmd_run(args) -> int:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "linkers").mkdir(exist_ok=True)
    ids = [ln.strip() for ln in Path(args.identities).read_text().splitlines()
           if ln.strip()]
    ids = sorted(set(ids))
    if args.shuffle_seed is not None:
        import random
        random.Random(args.shuffle_seed).shuffle(ids)
    if args.limit:
        ids = ids[: args.limit]
    done_path = run_dir / "results.jsonl"
    done_keys = set()
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get("status") == "converged":
                    done_keys.add(rec["identity_key"])
            except Exception:
                pass
    todo = [s for s in ids if identity_key(s) not in done_keys]
    meta = {"xtb": XTB, "timeout_s": TIMEOUT_S, "embed_seed": EMBED_SEED,
            "n_confs": N_CONFS, "n_identities": len(ids), "n_todo": len(todo),
            "workers": args.workers, "task_timeout_s": args.task_timeout,
            "embed_timeout_s": args.embed_timeout,
            "xtb_sha256": hashlib.sha256(Path(XTB).read_bytes()).hexdigest()}
    (run_dir / "campaign.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    n_done = n_fail = 0
    t0 = time.monotonic()
    worker = functools.partial(run_one_timed, run_dir=run_dir,
                               task_timeout=args.task_timeout,
                               embed_timeout=args.embed_timeout)
    with done_path.open("a") as out, Pool(args.workers) as pool:
        for rec in pool.imap_unordered(worker, todo, chunksize=1):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if rec.get("status") == "converged":
                n_done += 1
            else:
                n_fail += 1
            if (n_done + n_fail) % 200 == 0:
                rate = (n_done + n_fail) / (time.monotonic() - t0)
                print(f"progress {n_done + n_fail}/{len(todo)} "
                      f"converged={n_done} failed={n_fail} "
                      f"rate={rate:.1f}/s", flush=True)
    print(f"DONE converged={n_done} failed={n_fail} "
          f"elapsed={(time.monotonic() - t0) / 60:.1f} min")
    return 0


def cmd_collect(args) -> int:
    import pandas as pd

    run_dir = Path(args.run_dir)
    rows = []
    for rp in sorted((run_dir / "linkers").glob("*/result.json")):
        try:
            rows.append(json.loads(rp.read_text()))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    df = df.drop(columns=[c for c in ("skipped",) if c in df.columns])
    df.to_csv(args.out, index=False)
    print(f"collected {len(df)} results -> {args.out}")
    print(df["status"].value_counts().to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--identities", required=True)
    pr.add_argument("--run-dir", required=True)
    pr.add_argument("--workers", type=int, default=64)
    pr.add_argument("--limit", type=int, default=0)
    pr.add_argument("--shuffle-seed", type=int, default=None)
    pr.add_argument("--task-timeout", type=int, default=2100,
                    help="hard per-identity wall cap in seconds; the grandchild "
                         "process is killed past this (default 2100, above "
                         "xTB's own 1800s wrapper)")
    pr.add_argument("--embed-timeout", type=int, default=240,
                    help="hard cap for the RDKit embedding stage alone "
                         "(default 240s; RDKit distance geometry cannot be "
                         "interrupted once stuck)")
    pc = sub.add_parser("collect")
    pc.add_argument("--run-dir", required=True)
    pc.add_argument("--out", required=True)
    p1 = sub.add_parser("compute-one")
    p1.add_argument("--smiles", required=True)
    p1.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "compute-one":
        return cmd_compute_one(args)
    return cmd_collect(args)


if __name__ == "__main__":
    sys.exit(main())
