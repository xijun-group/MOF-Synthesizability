#!/usr/bin/env python3
"""Predict MOF synthesizability from a CIF file.

Given one CIF, this driver runs the bundle's four descriptor stacks, assembles the five
inputs of the delivered five-feature VPU+MLP[128] model, and scores them:

    1. MOFid decomposition          -> Output/MetalOxo/{linker_stats.txt, linkers.cif}
    2. sascore                      -> max rdkit SA_Score over the standardised neutral
                                       MOFid linker fragments (the delivered repair rule)
       channel-B HOMO                -> max GFN2-xTB HOMO over the same fragment set
                                       (OrganicFragment_HOMO_mofid_neutral_max_complete_[eV];
                                        defined only when every fragment converged)
    3. RMSD_[A]                     -> a 2x supercell of the CIF is decomposed with MOFid;
                                       prefer exact then stereo-compatible SA-max geometry, else the largest fragment,
                                       relax it in the gas phase and compare with
                                       the in-framework geometry, min over calculate_rmsd's four
                                       reorder methods
    4. solvent_removal_stability    -> MOFSimplify Zeo++/molSimplify descriptors -> the
                                       solvent ANN (full precision, the delivered value)
    5. Ehull                        -> UMA-s-1p1 (ODAC) MLIP relaxation + the fixed QMOF
                                       DFT reference hull (the delivered Ehull flow)

    score = sigmoid(MLP[128](z)),  z = (x - pooled_means) / pooled_population_stds

The score is an **uncalibrated operational value**, not a probability of synthesis, and the
scripts here reproduce descriptor computation only: nothing is retrained, rescaled or
recalibrated. `U` (the model's negative side) is ordinary unverified background, not
confirmed non-synthesizable MOFs.

Everything runs through the installed prefix (paths.env / the environment exported by
`activate.sh`), so no project path is needed. Where a rule is a project script, the
installed copy under <prefix>/predict/rules/ is imported, so the arithmetic is the delivered
one rather than a re-derivation.

Examples
--------
    # what would run, with which paths (no computation)
    predict.sh --plan

    # one CIF, everything
    predict.sh --cif my.cif --label my_mof

    # reuse pre-computed features (no stack is run)
    predict.sh --from-features features.csv --out scores.json

    # skip the GPU step (the score is then not defined unless --allow-missing)
    predict.sh --cif my.cif --skip ehull
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "acceleration"))
from ehull_protocol import from_environment, matches_receipt

# The five model inputs, in the trained order. These are the *checkpoint's own* feature names:
# slot 3 is the frozen `Linker_HOMO_[eV]` slot, which the B-channel model fills with
# `OrganicFragment_HOMO_mofid_neutral_max_complete_[eV]` (recorded as `linker_homo_channel` in
# the model card and printed alongside the slot name).
FEATURES = (
    "solvent_removal_stability",
    "RMSD_[A]",
    "Linker_HOMO_[eV]",
    "sascore",
    "Ehull",
)
CHANNEL_SLOT = "Linker_HOMO_[eV]"
FEATURE_ALIASES = {
    "rmsd": "RMSD_[A]",
    "rmsd_[a]": "RMSD_[A]",
    "linker_homo_[ev]": CHANNEL_SLOT,
    "channel_b": CHANNEL_SLOT,
    "b_homo_ev": CHANNEL_SLOT,
    "organicfragment_homo_mofid_neutral_max_complete_[ev]": CHANNEL_SLOT,
    "solvent_removal_stability": "solvent_removal_stability",
    "sascore": "sascore",
    "ehull": "Ehull",
}
XTB_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OMP_STACKSIZE": "256M"}


def log(msg: str) -> None:
    print(f"[predict] {msg}", flush=True)


def die(msg: str, code: int = 2) -> "NoReturn":  # noqa: F821
    print(f"[predict] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


# ----------------------------------------------------------------------------- modules
def load_module(name: str, path: Path):
    """Import a bundled project script by path (the delivered rule, not a re-write)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        die(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- paths
def read_paths_env(prefix: Path) -> dict:
    out = {}
    f = prefix / "paths.env"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def resolve(args) -> dict:
    prefix = Path(args.prefix).resolve() if args.prefix else None
    penv = read_paths_env(prefix) if prefix else {}

    def pick(key, *fallbacks):
        for c in (os.environ.get(key), penv.get(key), *fallbacks):
            if c:
                return str(c)
        return ""

    if prefix is None:
        p = pick("MOFENV_PREFIX")
        prefix = Path(p).resolve() if p else None
    ctx = {
        "prefix": prefix,
        "xtb": pick("XTB_BIN", str(prefix / "xtb/bin/xtb") if prefix else ""),
        "mofid_root": pick("MOFID_ROOT", str(prefix / "mofid-build") if prefix else ""),
        "java_home": pick("JAVA_HOME", str(prefix / "java") if prefix else ""),
        "venv_python": pick("MOFSYNTH_PYTHON", str(prefix / "mofsynth-env/bin/python") if prefix else ""),
        "sascore_python": pick("SASCORE_PYTHON", str(prefix / "mofsynth-env/bin/python") if prefix else ""),
        "mofsimplify_python": pick("MOFSIMPLIFY_PYTHON", str(prefix / "mofsimplify-env/bin/python") if prefix else ""),
        "mofsimplify_root": pick("MOFSIMPLIFY_ROOT", str(prefix / "mofsimplify-repo") if prefix else ""),
        "zeo_network": pick("ZEO_NETWORK", str(prefix / "mofsimplify-repo/zeo++-0.3/network") if prefix else ""),
        "ehull_root": pick("EHULL_ROOT", str(prefix / "qmof-ehull") if prefix else ""),
        "ehull_python": (args.ehull_python
                         or (str(prefix / "qmof-ehull/pyenv/bin/python")
                             if prefix and (prefix / "qmof-ehull/pyenv/bin/python").exists() else "")
                         or pick("EHULL_PYTHON")),
        "model_dir": Path(args.model_dir).resolve() if args.model_dir
                     else Path(pick("MODEL_DIR", str(prefix / "model") if prefix else "model")),
        "predict_dir": Path(__file__).resolve().parent,
        "fast_pbc": getattr(args, "fast_pbc", False),
        "fast_rmsd": getattr(args, "fast_rmsd", False),
        "fast_sbu_lib": (str(Path(__file__).resolve().parent.parent / "acceleration/openbabel")
                         if getattr(args, "fast_sbu", False) else ""),
        "fast_ehull": getattr(args, "fast_ehull", False),
        "ehull_solver": getattr(args, "ehull_solver", "orig"),
        "ehull_inference": getattr(args, "ehull_inference", "default"),
    }
    ctx["rules"] = ctx["predict_dir"] / "rules"
    return ctx


def check_paths(ctx: dict, needed) -> list:
    """Return the list of missing requirements among `needed`."""
    missing = []
    dev = {
        "mofid": (ctx["mofid_root"], "MOFID_ROOT"),
        "xtb": (ctx["xtb"], "XTB_BIN"),
        "venv_python": (ctx["venv_python"], "MOFSYNTH_PYTHON"),
        "mofsimplify_python": (ctx["mofsimplify_python"], "MOFSIMPLIFY_PYTHON"),
        "mofsimplify_root": (ctx["mofsimplify_root"], "MOFSIMPLIFY_ROOT"),
        "zeo": (ctx["zeo_network"], "ZEO_NETWORK"),
        "model": (ctx["model_dir"] / "model.pt", "model dir"),
        "ehull": (ctx["ehull_root"], "EHULL_ROOT"),
    }
    for key in needed:
        p, label = dev[key]
        if not p or not Path(p).exists():
            missing.append(f"{key} ({label}: {p or 'unset'})")
    return missing


# ----------------------------------------------------------------------------- process
def run(cmd, cwd=None, env=None, capture=True, timeout=None, tag=""):
    cmd = [str(c) for c in cmd]
    if tag:
        log(f"{tag}: {' '.join(cmd[:6])}{' …' if len(cmd) > 6 else ''}")
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=e, timeout=timeout,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None, text=True)
    if capture and r.returncode != 0 and tag:
        tail = "\n".join((r.stdout or "").strip().splitlines()[-12:])
        log(f"{tag} exited {r.returncode}; output tail:\n{tail}")
    return r


# ----------------------------------------------------------------------------- step 1
def step_mofid(ctx, cif: Path, work: Path) -> dict:
    """MOFid decomposition of one CIF (in-process, in the prefix's venv)."""
    out = work / "mofid"
    incomplete = out / "extraction_incomplete.json"
    if incomplete.exists():
        die(f"MOFid previously failed or was interrupted; partial files in {out} cannot be reused", 3)
    source_receipt = out / "fragment_source.json"
    output_subdir = json.loads(source_receipt.read_text())["output_subdir"] if source_receipt.exists() else "Output"
    out_dir = out / output_subdir
    stats = out_dir / "MetalOxo" / "linker_stats.txt"
    linkers = out_dir / "MetalOxo" / "linkers.cif"
    if stats.exists() and linkers.exists():
        log(f"MOFid: reusing {out_dir}")
        return {"output_dir": str(out_dir), "linker_stats": str(stats),
                "linkers_cif": str(linkers), "reused": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    incomplete.write_text(json.dumps({"state": "started", "input": str(cif)}) + "\n")
    _prepend_path(ctx, "java_home", "bin")
    _prepend_path(ctx, "mofid_root", "bin")
    from mofid.paths import bin_path, mofid_path, openbabel_path  # noqa: F401
    from mofid.run_mofid import cif2mofid
    log(f"MOFid: {cif.name} -> {out_dir}")
    identifiers = cif2mofid(str(cif), output_path=str(out_dir))
    if identifiers.get("smiles_nodes") == ['*'] or not identifiers.get("smiles_linkers"):
        die("MOFid extraction failed; partial output files are not usable", 3)
    if not stats.exists() or not linkers.exists():
        die(f"MOFid produced no linker_stats.txt / linkers.cif under {out_dir} "
            f"(is /tmp/mofid-build linked? see specs/mofid.md)", 3)
    incomplete.unlink()
    return {"output_dir": str(out_dir), "linker_stats": str(stats),
            "linkers_cif": str(linkers), "reused": False}


# libopenbabel.so.7.0.0 of the MOFid build that acceleration/openbabel/libopenbabel.so.7 relinks
# with only ring.cpp replaced.
FAST_SBU_BASE_LIB_SHA256 = "a4e983b7a8c727eaaba4c7ef92f3175f461bc0dbc652eb3051964f6cfd4abe6e"


def step_mofid_fragments(ctx, cif: Path, work: Path) -> dict:
    """The sbu fragment decomposition of step_mofid without its Systre topology calls, run against
    the SSSR-accelerated libopenbabel. For the RMSD supercell only: that leg consumes nothing but
    MetalOxo/linkers.cif, and the accelerated ring search returns the original ring list."""
    out = work / "mofid"
    incomplete = out / "extraction_incomplete.json"
    if incomplete.exists():
        die(f"MOFid previously failed or was interrupted; partial files in {out} cannot be reused", 3)
    source_receipt = out / "fragment_source.json"
    output_subdir = json.loads(source_receipt.read_text())["output_subdir"] if source_receipt.exists() else "Output"
    out_dir = out / output_subdir
    stats = out_dir / "MetalOxo" / "linker_stats.txt"
    linkers = out_dir / "MetalOxo" / "linkers.cif"
    if stats.exists() and linkers.exists():
        log(f"MOFid fragments: reusing {out_dir}")
        return {"output_dir": str(out_dir), "linker_stats": str(stats),
                "linkers_cif": str(linkers), "reused": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    _prepend_path(ctx, "java_home", "bin")
    _prepend_path(ctx, "mofid_root", "bin")
    import mofid.id_constructor as ic
    from mofid.paths import mofid_path
    from mofid.run_mofid import cif2mofid  # noqa: F401  same import side effects (BABEL_DATADIR) as step_mofid
    real_sbu = Path(mofid_path) / "bin" / "sbu"
    runtime_lib = Path(mofid_path) / "openbabel" / "build" / "lib"
    if sha256(runtime_lib / "libopenbabel.so.7") != FAST_SBU_BASE_LIB_SHA256 or not real_sbu.is_file():
        log("MOFid fragments: this runtime's libopenbabel is not the build the SSSR patch was made "
            "from; using the original MOFid call")
        return step_mofid(ctx, cif, work)
    saved_env, saved_bin = os.environ.get("LD_LIBRARY_PATH"), ic.SBU_BIN
    incomplete.write_text(json.dumps({"state": "started", "input": str(cif)}) + "\n")
    # bin_mofid/sbu is a wrapper that prepends runtime_lib and execs bin/sbu. Do the same with the
    # accelerated library first: LD_LIBRARY_PATH outranks sbu's DT_RUNPATH, and only this process sees it.
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [str(ctx["fast_sbu_lib"]), str(runtime_lib)] + ([saved_env] if saved_env else []))
    ic.SBU_BIN = str(real_sbu)
    try:
        log(f"MOFid fragments (no topology): {cif.name} -> {out_dir}")
        extracted = ic.extract_fragments(str(Path(cif).resolve()), str(out_dir.resolve()))
    finally:
        ic.SBU_BIN = saved_bin
        if saved_env is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = saved_env
    fast_failed = extracted[0] == ['*'] or not extracted[1]
    if fast_failed:
        # An accelerator may crash after writing files. Retry the original SBU
        # in a separate output tree, using its restored library environment.
        log("Accelerated MOFid extraction failed; retrying original SBU in a fresh output tree")
        out_dir = out / "Output_original"
        out_dir.mkdir(exist_ok=True)
        extracted = ic.extract_fragments(str(Path(cif).resolve()), str(out_dir.resolve()))
        stats = out_dir / "MetalOxo" / "linker_stats.txt"
        linkers = out_dir / "MetalOxo" / "linkers.cif"
        if extracted[0] == ['*'] or not extracted[1]:
            die("Original MOFid supercell extraction also failed; partial files are not usable", 3)
    if not stats.exists() or not linkers.exists():
        die(f"MOFid produced no linker_stats.txt / linkers.cif under {out_dir}", 3)
    source_receipt.write_text(json.dumps({"output_subdir": out_dir.name,
                              "accelerated_attempt_failed": fast_failed,
                              "state": "complete", "input_cif_sha256": sha256(cif)}, indent=2) + "\n")
    incomplete.unlink()
    return {"output_dir": str(out_dir), "linker_stats": str(stats),
            "linkers_cif": str(linkers), "reused": False}


def _prepend_path(ctx: dict, key: str, sub: str) -> None:
    root = ctx.get(key) or ""
    if not root:
        return
    p = str(Path(root) / sub)
    if Path(p).is_dir() and p not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")


# ----------------------------------------------------------------------------- step 2
def step_fragment_set(ctx, stats: Path, work: Path) -> dict:
    """Organic-filtered canonical MOFid linker set + the standardised neutral identities."""
    std_mod = load_module("predict_rules_std", ctx["rules"] / "standardize_linker_smiles.py")
    parsed = std_mod.standardize_mofid_linker_stats(stats)
    audit = work / "fragment_standardization.json"
    audit.write_text(json.dumps(parsed, indent=2) + "\n")
    if not parsed["complete"]:
        die(f"incomplete organic fragment set in {stats}; see {audit}", 3)
    return parsed


def score_fragments(ctx, smiles_set) -> dict:
    """rdkit SA_Score per standardised identity; the sascore feature is the max."""
    from rdkit import Chem, RDConfig, RDLogger
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    out, vals = {}, []
    for s in smiles_set:
        mol = Chem.MolFromSmiles(s)
        try:
            v = float(sascorer.calculateScore(mol)) if mol is not None else None
        except Exception:
            v = None
        out[s] = v
        if v is not None and math.isfinite(v):
            vals.append(v)
    complete = bool(smiles_set) and len(vals) == len(smiles_set)
    return {"per_fragment": out, "sascore": (max(vals) if complete else None),
            "n_scored": len(vals), "complete": complete}


# ----------------------------------------------------------------------------- step 2b
def aggregate_channel_b(per_fragment: dict, std_set) -> dict:
    """The delivered channel-B reduction: the max over the fragment set, defined only when
    every fragment of the set has a converged HOMO (objectABC_v1's completeness rule)."""
    per = {s: per_fragment.get(s) for s in std_set}
    vals = [v for v in per.values() if v is not None and math.isfinite(v)]
    complete = bool(std_set) and len(vals) == len(std_set)
    return {"per_fragment": per, "complete": complete,
            "B": (max(vals) if complete else None),
            "n_identities": len(std_set), "n_converged": len(vals),
            "reason": "" if complete else "at least one fragment has no converged HOMO"}


def step_channel_b(ctx, std_set, work: Path, workers: int, timeout: int) -> dict:
    """Per-identity GFN2-xTB HOMO for every standardised fragment; B is their max, and is
    defined only when every fragment of the set converged (the delivered completeness rule)."""
    run_dir = work / "linker_homo"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not std_set:
        return {"per_fragment": {}, "complete": False, "B": None,
                "reason": "no standardisable fragment", "n_identities": 0, "n_converged": 0}
    ids = run_dir / "identities.txt"
    ids.write_text("\n".join(sorted(std_set)) + "\n")
    mod = load_module("predict_rules_linker_qm", ctx["rules"] / "canonical_linker_qm.py")
    if ctx["xtb"]:
        mod.XTB = ctx["xtb"]
    for stage in (["run", "--identities", ids, "--run-dir", run_dir,
                   "--workers", workers, "--task-timeout", max(timeout, 300)],
                  ["collect", "--run-dir", run_dir, "--out", run_dir / "canonical_linker_homo.csv"]):
        argv, sys.argv = sys.argv, [str(ctx["rules"] / "canonical_linker_qm.py")] + [str(a) for a in stage]
        try:
            mod.main()
        finally:
            sys.argv = argv
    per = {}
    for s in std_set:
        per.setdefault(s, None)
    for rp in sorted((run_dir / "linkers").glob("*/result.json")):
        rec = json.loads(rp.read_text())
        s = rec.get("std_linker")
        if s in per and rec.get("status") == "converged" and rec.get("homo_eV") is not None:
            per[s] = float(rec["homo_eV"])
    out = aggregate_channel_b(per, std_set)
    out["xtb_sha256"] = (mod.hashlib.sha256(Path(mod.XTB).read_bytes()).hexdigest()
                         if Path(mod.XTB).exists() else None)
    return out


# ----------------------------------------------------------------------------- step 3
def step_rmsd(ctx, cif: Path, work: Path, timeout: int, supercell: int = 2,
              sa_scores=None) -> dict:
    """RMSD/DeltaE of exact/compatible SA-max geometry, or the largest fragment.

    Retain 2x extraction, existing coordinates, xTB defaults and RMSD methods.
    Identity normalization is for matching only. If neither exact nor stereo-compatible match exists, use
    the original largest-component rule; record the actual selected object.
    """
    selector = load_module("predict_rules_rmsd_target", ctx["rules"] / "rmsd_sa_target.py")
    selection = selector.select_parent(sa_scores)
    input_hash = sha256(cif)
    d = work / "rmsd"
    d.mkdir(parents=True, exist_ok=True)
    jin = d / "rmsd.json"
    if jin.exists():
        cached = json.loads(jin.read_text())
        if (cached.get("target_selection") != selection or
                cached.get("input_cif_sha256") != input_hash or
                cached.get("supercell") != supercell or
                not (d / "linker.xyz").exists() or
                cached.get("selected_xyz_sha256") != sha256(d / "linker.xyz")):
            die("RMSD cache uses a different target/protocol/input; use a fresh --work directory", 3)
        log("RMSD: reusing verified target/fallback protocol rmsd.json")
        return cached
    # Even --rmsd-supercell 1 consumes a decomposition, never the complete MOF CIF.
    base = step_mofid(ctx, cif, work)
    linkers_cif = Path(base["linkers_cif"])
    decomposition = "1x (the CIF as given)"
    if supercell and supercell > 1:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
        st = Structure.from_file(str(cif))
        sup_cif = d / f"supercell{supercell}.cif"
        CifWriter(st * supercell).write_file(str(sup_cif))
        log(f"RMSD: {supercell}x supercell, {len(st)} -> {len(st) * supercell ** 3} sites")
        if ctx.get("fast_sbu_lib"):
            sup = step_mofid_fragments(ctx, sup_cif, work / f"mofid_supercell{supercell}")
        else:
            sup = step_mofid(ctx, sup_cif, work / f"mofid_supercell{supercell}")
        linkers_cif = Path(sup["linkers_cif"])
        decomposition = f"{supercell}x supercell (the upstream MOFSynth rule)"
    venv_bin = Path(ctx["venv_python"]).parent
    obabel = str(venv_bin / "obabel")
    rmsd_bin = str(venv_bin / "calculate_rmsd")
    for p in (obabel, rmsd_bin, ctx["xtb"]):
        if not Path(p).exists():
            die(f"RMSD leg needs {p}; it is missing", 3)
    xyz = d / "linker.xyz"
    try:
        geometry = selector.extract_geometry(linkers_cif, xyz, selection)
    except ValueError as exc:
        die(f"Framework geometry selection failed: {exc}; see {d / 'geometry_selection.json'}", 3)
    log(f"RMSD geometry: {geometry['selection_mode']}, {geometry['chosen']['natoms']} atoms")
    smi_file = d / "linker.smi"
    run([obabel, xyz, "-xc", "-O", smi_file], tag="obabel -xc")
    lines = xyz.read_text().splitlines()
    natoms = int(lines[0])
    smiles = smi_file.read_text().split()[0] if smi_file.exists() and smi_file.read_text().split() else ""

    def xtb(inp: Path, sub: str, args_xtb, tag):
        wd = d / sub
        wd.mkdir(exist_ok=True)
        shutil.copy2(inp, wd / "linker.xyz")
        with (wd / "check.out").open("w") as so, (wd / "check.err").open("w") as se:
            rr = subprocess.run([str(ctx["xtb"]), "linker.xyz"] + list(args_xtb), cwd=str(wd),
                                stdout=so, stderr=se, timeout=timeout, env={**os.environ, **XTB_ENV})
        out = (wd / "check.out").read_text(errors="replace")
        e = None
        for line in out.splitlines():
            if "| TOTAL ENERGY" in line:
                try:
                    e = float(line.split()[3])
                except (ValueError, IndexError):
                    pass
        return rr.returncode, e, out, wd

    rc0, e0, _, _ = xtb(xyz, "sp0", ["--sp"], "xTB single point (pre-opt)")
    rc1, e1, opt_out, opt_dir = xtb(xyz, "opt", ["--opt"], "xTB geometry optimisation")
    converged = "GEOMETRY OPTIMIZATION CONVERGED" in opt_out
    opt_xyz = opt_dir / "xtbopt.xyz"
    if rc1 != 0 or not converged or not opt_xyz.exists():
        die("xTB optimisation of the selected framework fragment did not converge", 3)
    rc2, e2, _, _ = xtb(opt_xyz, "sp1", ["--sp"], "xTB single point (post-opt)")
    if rc0 != 0 or rc2 != 0 or any(e is None or not math.isfinite(e) for e in (e0, e1, e2)):
        die("Selected fragment energy calculation failed or has a non-finite energy", 3)
    vals = []
    if ctx.get("fast_rmsd"):
        helper = ctx["predict_dir"].parent / "acceleration/rmsd_four_methods.py"
        rr = run([ctx["venv_python"], helper, opt_xyz, xyz], timeout=900)
        if rr.returncode:
            die("RMSD method worker failed: " + rr.stdout[-1000:], 3)
        records = json.loads(rr.stdout)
        vals = [tuple(item) for item in records["values"]]
        (d / "method_details.json").write_text(json.dumps(records, indent=2) + "\n")
    else:
        for extra, tag in (([], "plain"), (["--reorder-method", "hungarian"], "hungarian"),
                           (["--reorder-method", "inertia-hungarian"], "inertia-hungarian"),
                           (["--reorder-method", "distance"], "distance")):
            rr = run([rmsd_bin, "-e", *extra, opt_xyz, xyz], timeout=300)
            try:
                vals.append((float(rr.stdout.strip().split()[0]), tag))
            except (ValueError, IndexError):
                pass
    if not vals:
        die("calculate_rmsd produced no numeric value", 3)
    v, method = min(vals, key=lambda t: t[0])
    rec = {"rmsd_a": v, "rmsd_method": method, "rmsd_n_methods": len(vals),
           "all_method_values": {t: x for x, t in vals}, "natoms": natoms, "smiles": smiles,
           "sp0_energy_hartree": e0, "opt_energy_hartree": e1, "sp1_energy_hartree": e2,
           "delta_energy_hartree": (e1 - e0) if (e0 is not None and e1 is not None) else None,
           "delta_energy_kcal_mol": (e1 - e0) * 627.51,
           "target_selection": selection, "input_cif_sha256": input_hash, "supercell": supercell,
           "selected_xyz_sha256": geometry["selected_xyz_sha256"],
           "geometry_selection": str(d / "geometry_selection.json"),
           "selected_geometry_component": geometry["chosen"],
           "selection_mode": geometry["selection_mode"],
           "fallback_used": geometry["fallback_used"],
           "fallback_reason": geometry["fallback_reason"],
           "selected_matches_sa_parent": geometry["selected_matches_sa_parent"],
           "selected_corresponds_sa_parent": geometry["selected_corresponds_sa_parent"],
           "stereo_compatibility_used": geometry["stereo_compatibility_used"],
           "n_stereo_compatible_components": geometry["n_stereo_compatible_components"],
           "method_tag": "GFN2-default-charge-spin",
           "decomposition": decomposition, "linkers_cif": str(linkers_cif),
           "linkers_cif_sha256": sha256(linkers_cif)}
    jin.write_text(json.dumps(rec, indent=2))
    log(f"RMSD: {v:.10f} A ({method}, {len(vals)} methods, {natoms} atoms)")
    return rec


# ----------------------------------------------------------------------------- step 4
def step_mofsimplify(ctx, cif: Path, label: str, work: Path, timeout: int) -> dict:
    """MOFSimplify: Zeo++/molSimplify descriptors -> the solvent ANN (full precision)."""
    d = work / "mofsimplify"
    desc, pred = d / "desc", d / "predictions.csv"
    d.mkdir(parents=True, exist_ok=True)
    cifs = d / "cifs.txt"
    cifs.write_text(str(cif) + "\n")
    desc.mkdir(exist_ok=True)
    shim = Path(__file__).resolve().parent / "mofsimplify_solvent.py"
    env = {"MOFSIMPLIFY_ROOT": ctx["mofsimplify_root"], "ZEO_NETWORK": ctx["zeo_network"]}
    featurizer = [ctx["mofsimplify_python"]]
    if ctx.get("fast_pbc"):
        featurizer.append(ctx["predict_dir"].parent / "cpu_candidate/run_mofsimplify_optimized.py")
    r = run([*featurizer, ctx["mofsimplify_root"] + "/batch/featurize_shards.py",
             "--cif-list", cifs, "--out", desc, "--shard", "0", "--nshards", "1",
             "--timeout", timeout], env=env, tag="MOFSimplify featurize")
    produced = list(desc.glob("*_descriptors.csv"))
    if not produced:
        log((r.stdout or "").strip()[-800:])
        die("MOFSimplify featurization produced no descriptor CSV", 3)
    r = run([ctx["mofsimplify_python"], shim, "--predict-slt",
             ctx["predict_dir"] / "predict_slt.py", "--desc-dir", desc, "--out", pred],
            env=env, tag="MOFSimplify solvent ANN")
    if not pred.exists():
        log((r.stdout or "").strip()[-800:])
        die("the MOFSimplify predictor produced no CSV", 3)
    row = None
    with pred.open() as fh:
        for rec in csv.DictReader(fh):
            if rec.get("name") in (label, cif.stem):
                row = rec
                break
    if row is None:
        die(f"no prediction row for {label!r} in {pred}", 3)
    val = row.get("solvent_flag")
    flag = float(val) if val not in (None, "", "nan") else None
    out = {"solvent_removal_stability": flag, "solvent_flag": flag,
           "solvent_in_train": row.get("solvent_in_train"),
           "thermal_T_C": row.get("thermal_T_C"), "water_proba": row.get("water_proba"),
           "acid_proba": row.get("acid_proba"),
           "descriptor_csv": str(produced[0]), "predictions_csv": str(pred),
           "semantics": ("MOFSimplify's solvent-removal ANN output on its own training data: "
                         "an uncalibrated operational prediction, not a measured stability")}
    log(f"solvent_removal_stability: {flag}")
    return out


# ----------------------------------------------------------------------------- step 5
def step_ehull(ctx, cif: Path, label: str, work: Path) -> dict:
    """3.1.1 combined convergence followed by the unchanged fixed reference hull."""
    root = Path(ctx["ehull_root"])
    run_dir = work / "qmof-ehull"
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = from_environment()
    model = ctx.get("ehull_model") or "uma-s-1p1"
    tag = Path(model).name.removesuffix(".pt").replace(".", "_")
    js = run_dir / "ehull" / f"{label}_{tag}_ehull.json"
    relaxation_file = run_dir / "relaxations" / label / "results.json"
    expected_input = sha256(cif)
    configuration = dict(requested_model=model, solver=ctx.get("ehull_solver", "orig"),
                         inference=ctx.get("ehull_inference", "default"), energy_type="ODAC_MLIP")

    def read_current_result():
        try:
            doc = json.loads(js.read_text())
            receipt = json.loads(relaxation_file.read_text())
        except (OSError, ValueError) as exc:
            die(f"Ehull result/receipt is incomplete: {exc}; use a fresh --work directory", 3)
        valid = (matches_receipt(receipt, settings)
                 and doc.get("convergence_protocol") == settings
                 and doc.get("relaxation_converged") is True
                 and all(x.get("input_sha256") == expected_input
                         and x.get("run_configuration") == configuration for x in (doc, receipt)))
        if not valid:
            die("Ehull cache does not certify the requested 3.1.1 combined protocol, input and settings; "
                "use a fresh --work directory to recompute. Existing files are preserved.", 3)
        return doc, receipt

    if js.exists():
        doc, relaxation = read_current_result()
        log(f"Ehull: reusing verified combined-protocol result {js}")
    else:
        command = [ctx["ehull_python"], ctx["predict_dir"].parent / "acceleration/ehull_fast.py", cif,
                   "--scripts", root / "scripts", "--model", model, "--label", label,
                   "--fmax", settings["filter_fmax"], "--atom-fmax", settings["atom_fmax_eV_A"],
                   "--stress-max-gpa", settings["stress_GPa"],
                   "--energy-window-steps", settings["energy_window_steps"],
                   "--energy-span-per-atom", settings["energy_span_eV_atom"],
                   "--consecutive-steps", settings["consecutive_steps"],
                   "--max-steps", os.environ.get("EHULL_MAX_STEPS") or "10000",
                   "--device", ctx["device"] or "cuda", "--out-dir", run_dir / "relaxations",
                   "--json-out", js, "--solver", configuration["solver"],
                   "--inference", configuration["inference"],
                   "--timing-out", run_dir / "ehull" / f"{label}_timing.json"]
        if not ctx.get("fast_ehull"):
            command.append("--no-overlap-hull")
        r = run(command, timeout=ctx["ehull_timeout"], tag="QMOF-Ehull (combined convergence)")
        if r.returncode != 0 or not js.exists():
            die(f"Ehull did not produce an accepted combined-convergence result; inspect {relaxation_file}", 3)
        doc, relaxation = read_current_result()
    out = {"Ehull": doc.get("energy_above_hull"), "json": str(js),
           "energy_eV": doc.get("energy_eV"), "formation_energy": doc.get("formation_energy"),
           "decomposition_products": doc.get("decomposition_products"),
           "relaxation_converged": relaxation["converged"], "relaxation_receipt": str(relaxation_file),
           "convergence_protocol": settings, "convergence_final": relaxation["final"],
           "input_sha256": expected_input, "run_configuration": configuration,
           "endpoint_restart_or_perturbation_check": "not executed",
           "model": doc.get("model"),
           "semantics": ("an MLIP total energy measured against a fixed QMOF DFT reference hull; "
                         "combined convergence does not certify repeatability or a global minimum; "
                         "an uncalibrated operational feature, not a synthesis probability")}
    if out["Ehull"] is None or not math.isfinite(out["Ehull"]):
        die(f"Ehull is missing or nonfinite: {js}", 3)
    log(f"Ehull: {out['Ehull']}")
    return out


# ----------------------------------------------------------------------------- scoring
def load_model(model_dir: Path) -> dict:
    w = json.loads((model_dir / "model_weights.json").read_text())
    pre = dict(np.load(model_dir / "preprocessor.npz"))
    order = tuple(w["feature_order"])
    if order != FEATURES:
        die(f"model_weights.json feature order {order} != {FEATURES}")
    return {"w": w, "pre": pre, "order": order,
            "card": json.loads((model_dir / "model_card.json").read_text())
            if (model_dir / "model_card.json").exists() else {}}


def score_rows(model: dict, rows) -> list:
    """rows: list of (label, [five floats or None]). NaN -> the training finite median."""
    pre, w = model["pre"], model["w"]
    med = np.asarray(pre["finite_medians"], dtype=np.float64)
    mean = np.asarray(pre["pooled_means"], dtype=np.float64)
    std = np.asarray(pre["pooled_population_stds"], dtype=np.float64)
    x = np.asarray(rows, dtype=np.float64)
    imputed = np.isnan(x)
    x = np.where(imputed, med[None, :], x)
    z = ((x - mean[None, :]) / std[None, :]).astype(np.float32)
    w1 = np.asarray(w["w1"], dtype=np.float32)   # (128, 5)
    b1 = np.asarray(w["b1"], dtype=np.float32)   # (128,)
    w2 = np.asarray(w["w2"], dtype=np.float32)   # (1, 128)
    b2 = np.asarray(w["b2"], dtype=np.float32)   # (1,)
    h = np.maximum(z @ w1.T + b1, 0.0)
    logit = (h @ w2.T + b2).ravel().astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-logit.astype(np.float64)))
    return [{"logit": float(a), "native_score": float(b), "imputed": [bool(v) for v in imp]}
            for a, b, imp in zip(logit, prob, imputed)]


def model_summary(model: dict) -> dict:
    card = model["card"]
    return {"model": card.get("model", "vpu_mlp_relu_h1_128"),
            "features": list(model["order"]),
            "channel": card.get("linker_homo_channel"),
            "seed": card.get("seed"),
            "training_scope": card.get("training_scope"),
            "parameter_count": card.get("parameter_count"),
            "model_pt_sha256": card.get("model_pt_sha256"),
            "score_semantics": ("uncalibrated operational score (sigmoid of the network "
                                "logit); not a probability of synthesis, not a calibrated "
                                "performance metric")}


# ----------------------------------------------------------------------------- driver
def features_from_csv(path: Path) -> list:
    rows = []
    with Path(path).open() as fh:
        rd = csv.DictReader(fh)
        lower = {(c or "").strip().lower(): c for c in (rd.fieldnames or [])}
        for rec in rd:
            label = rec.get(lower.get("name", "name")) or rec.get(lower.get("label", "label")) or "row"
            vals = []
            for f in FEATURES:
                cols = [f.lower()] + [k for k, v in FEATURE_ALIASES.items() if v == f]
                v = next((rec.get(lower[c]) for c in cols if c in lower), None)
                vals.append(None if v in (None, "", "nan", "NA") else float(v))
            rows.append((label, vals))
    return rows


def feature_labels(model: dict) -> list:
    """The five inputs with slot 3 shown as the channel that fills it."""
    card = model.get("card", {})
    chan = card.get("linker_homo_channel")
    return [f"{f} [{chan}]" if f == CHANNEL_SLOT and chan else f for f in FEATURES]


def plan(ctx, args) -> int:
    print("five-feature synthesizability prediction -- plan")
    print(f"  prefix          : {ctx['prefix']}")
    print(f"  model dir       : {ctx['model_dir']}")
    print(f"  package version : 3.1.1")
    print(f"  Ehull protocol  : {json.dumps(from_environment(), sort_keys=True)}")
    for label, path in (("MOFid build", ctx["mofid_root"]), ("java", ctx["java_home"]),
                        ("xTB", ctx["xtb"]), ("MOFSynth-QM python", ctx["venv_python"]),
                        ("sascore python", ctx["sascore_python"]),
                        ("MOFSimplify python", ctx["mofsimplify_python"]),
                        ("MOFSimplify checkout", ctx["mofsimplify_root"]),
                        ("Zeo++", ctx["zeo_network"]), ("QMOF-Ehull payload", ctx["ehull_root"]),
                        ("Ehull MLIP python", ctx["ehull_python"] or "(not found)")):
        ok = "ok " if path and Path(path).exists() else "-- "
        print(f"  [{ok}] {label:22s} {path or '(unset)'}")
    if (ctx["model_dir"] / "model.pt").exists():
        print(f"  model           : {json.dumps(model_summary(load_model(ctx['model_dir'])), ensure_ascii=False)}")
    else:
        print(f"  model           : MISSING at {ctx['model_dir']} (model.pt)")
    missing = check_paths(ctx, ["mofid", "xtb", "venv_python"] if args.from_features else
                          ["mofid", "xtb", "venv_python", "mofsimplify_python",
                           "mofsimplify_root", "zeo", "model"])
    print(f"  steps that would run: {args.plan_steps}")
    if missing:
        print("  MISSING: " + "; ".join(missing))
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cif")
    ap.add_argument("--label")
    ap.add_argument("--prefix")
    ap.add_argument("--model-dir")
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--features", help="comma-separated five values, in the model order")
    ap.add_argument("--from-features", help="CSV with the five feature columns (and 'name')")
    ap.add_argument("--skip", default="", help="comma list of features to skip (e.g. ehull)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="score even when a feature is missing, imputing the training median "
                         "(the output then lists the imputed features)")
    ap.add_argument("--ehull-python")
    ap.add_argument("--ehull-model")
    ap.add_argument("--device", default=os.environ.get("EHULL_DEVICE", "cuda"))
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--stage-jobs", type=int, choices=range(1, 5), default=1,
                    help="independent descriptor processes (1 preserves serial execution; "
                         "2-4 overlap channel-B, RMSD, MOFSimplify and Ehull)")
    ap.add_argument("--fast-pbc", action=argparse.BooleanOptionalAction, default=False,
                    help="use the equivalent vectorized periodic distance kernel")
    ap.add_argument("--fast-rmsd", action=argparse.BooleanOptionalAction, default=False,
                    help="run original RMSD CLI methods in one interpreter")
    ap.add_argument("--fast-sbu", action=argparse.BooleanOptionalAction, default=False,
                    help="RMSD supercell: sbu fragments only, with the SSSR-accelerated libopenbabel")
    ap.add_argument("--early-stages", action=argparse.BooleanOptionalAction, default=False,
                    help="start Ehull and MOFSimplify before the MOFid fragment set "
                         "(RMSD and channel-B depend on it); needs --stage-jobs > 1")
    ap.add_argument("--fast-ehull", action=argparse.BooleanOptionalAction, default=False,
                    help="load the fixed hull in a forked process while the relaxation runs")
    ap.add_argument("--ehull-solver", choices=["orig", "chol"], default="orig",
                    help="BFGS step: orig eigh, or Cholesky when the Hessian is positive definite "
                         "(combined convergence applies with either solver)")
    ap.add_argument("--ehull-inference", choices=["default", "nockpt_merge"], default="default",
                    help="UMA fp32 inference settings: default, or no activation checkpointing "
                         "with merged MoLE experts (default remains off)")
    ap.add_argument("--timeout", type=int, default=1800, help="per xTB stage, seconds")
    ap.add_argument("--rmsd-supercell", type=int, default=2,
                    help="cell repeat for RMSD geometry extraction: 2 retains the upstream "
                         "supercell size; prefer exact/compatible SA-max geometry, else largest. 1 decomposes the CIF as given")
    ap.add_argument("--ehull-timeout", type=int, default=7200)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the result JSON to stdout")
    args = ap.parse_args()

    ctx = resolve(args)
    ctx["ehull_model"] = args.ehull_model or os.environ.get("EHULL_MODEL") or ""
    ctx["device"] = args.device
    ctx["ehull_timeout"] = args.ehull_timeout
    args.plan_steps = ("all five features (MOFid, sascore, channel-B xTB HOMO, RMSD, "
                       "MOFSimplify, Ehull) then score") if not args.from_features else \
                      "score only (features supplied)"
    if args.plan:
        return plan(ctx, args)

    model = load_model(ctx["model_dir"])
    skipped = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    t0 = time.time()

    # ---- features supplied directly -------------------------------------------------
    if args.features or args.from_features:
        rows = ([("cli", [float(v) if v not in ("", "nan") else None
                          for v in args.features.split(",")])] if args.features
                else features_from_csv(Path(args.from_features)))
        out_rows = []
        for label, vals in rows:
            res = score_rows(model, [vals])[0]
            out_rows.append({"name": label, "features": dict(zip(FEATURES, vals)), **res})
        doc = {"mode": "features-supplied", "model": model_summary(model), "rows": out_rows,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if args.out:
            Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        for r in out_rows:
            print(f"{r['name']}\t{r['native_score']!r}\tlogit={r['logit']!r}")
        if args.json:
            print(json.dumps(doc, indent=2))
        return 0

    # ---- full pipeline --------------------------------------------------------------
    if not args.cif:
        die("need --cif (or --from-features / --features)", 2)
    cif = Path(args.cif)
    if not cif.exists():
        die(f"CIF not found: {cif}", 2)
    label = args.label or cif.stem
    work = Path(args.work).resolve() if args.work else (Path.cwd() / "predict-work" / label)
    work.mkdir(parents=True, exist_ok=True)
    log(f"CIF {cif}  label {label}")
    log(f"work dir {work}")

    need = ["mofid", "xtb", "venv_python"]
    if "mofsimplify" not in skipped and "solvent_removal_stability" not in skipped:
        need += ["mofsimplify_python", "mofsimplify_root", "zeo"]
    if "ehull" not in skipped:
        need += ["ehull"]
    missing = check_paths(ctx, need)
    if missing:
        die("missing requirements: " + "; ".join(missing) + "  (run --plan to inspect)", 3)

    prov = {"cif": str(cif.resolve()), "cif_sha256": sha256(cif), "label": label,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "prefix": str(ctx["prefix"])}
    values = {k: None for k in FEATURES}
    detail = {}

    def front():
        mof = step_mofid(ctx, cif, work)
        frag = step_fragment_set(ctx, Path(mof["linker_stats"]), work)
        sc = score_fragments(ctx, frag["standardised_set"])
        log(f"sascore: {sc['sascore']}  ({sc['n_scored']}/{frag['n_fragments']} scored)")
        return mof, frag, sc

    early = args.early_stages and args.stage_jobs > 1
    if early:
        # Ehull/MOFSimplify read only the CIF; RMSD and channel-B wait for the fragment set.
        import concurrent.futures
        import threading
        front_future = concurrent.futures.Future()

        def run_front():
            try:
                front_future.set_result(front())
            except BaseException as exc:  # includes die()'s SystemExit
                front_future.set_exception(exc)

        # A daemon thread, so a cancelled run does not wait for MOFid before exiting.
        threading.Thread(target=run_front, daemon=True).start()
        from stage_scheduler import collect
        try:
            stage_results = collect(ctx, args, cif.resolve(), label, work,
                                    lambda: front_future.result()[1]["standardised_set"], skipped)
        except BaseException:
            if front_future.done() and front_future.exception() is not None:
                raise front_future.exception()
            raise
        mof, frag, sc = front_future.result()
    else:
        mof, frag, sc = front()
    detail["mofid"] = mof
    detail["fragment_set"] = frag
    values["sascore"] = sc["sascore"]
    detail["sascore"] = sc

    if args.stage_jobs > 1:
        if early:
            detail.update(stage_results)
        else:
            from stage_scheduler import collect
            detail.update(collect(ctx, args, cif.resolve(), label, work,
                                  frag["standardised_set"], skipped))
        values[FEATURES[2]] = detail["channel_b"].get("B")
        values["RMSD_[A]"] = detail["rmsd"].get("rmsd_a")
        values["solvent_removal_stability"] = detail["mofsimplify"].get("solvent_removal_stability")
        values["Ehull"] = detail["ehull"].get("Ehull")
    else:
        if "b" in skipped or "channel_b" in skipped or "linker_homo" in skipped:
            detail["channel_b"] = {"skipped": True}
        else:
            detail["channel_b"] = step_channel_b(ctx, frag["standardised_set"], work,
                                                 args.workers, args.timeout)
            values[FEATURES[2]] = detail["channel_b"]["B"]
            log(f"channel-B HOMO: {detail['channel_b']['B']}  complete="
                f"{detail['channel_b']['complete']}")

        if "rmsd" in skipped:
            detail["rmsd"] = {"skipped": True}
        else:
            detail["rmsd"] = step_rmsd(ctx, cif, work, args.timeout, args.rmsd_supercell,
                                      sa_scores=sc)
            values["RMSD_[A]"] = detail["rmsd"]["rmsd_a"]

        if "mofsimplify" in skipped or "solvent_removal_stability" in skipped:
            detail["mofsimplify"] = {"skipped": True}
        else:
            detail["mofsimplify"] = step_mofsimplify(ctx, cif, label, work, args.timeout)
            values["solvent_removal_stability"] = detail["mofsimplify"]["solvent_removal_stability"]

        if "ehull" in skipped:
            detail["ehull"] = {"skipped": True}
        else:
            if not ctx["ehull_python"] or not Path(ctx["ehull_python"]).exists():
                die("the Ehull MLIP interpreter was not found: build it with "
                    "install/build_ehull_env.sh, pass --ehull-python PATH, or use --skip ehull", 3)
            detail["ehull"] = step_ehull(ctx, cif, label, work)
            values["Ehull"] = detail["ehull"]["Ehull"]

    missing_feats = [k for k in FEATURES if values[k] is None]
    vec = [values[k] for k in FEATURES]
    scored = score_rows(model, [vec])[0]
    if missing_feats and not args.allow_missing:
        log("the model inputs are incomplete: " + ", ".join(missing_feats))
        log("no score is produced for an incomplete input vector (use --allow-missing to "
            "impute the training median, which is NOT a prediction for those features)")
        scored = None

    doc = {"mode": "cif-pipeline", "package_version": "3.1.1", "provenance": prov,
           "ehull_definition": "combined-v1; see details.ehull for executed protocol",
           "model_validation_for_combined_ehull": "not executed; checkpoint unchanged",
           "rmsd_definition": "framework geometry: exact SA-max parent, then compatibility for stereo-unspecified target, then largest fragment",
           "model_validation_for_sa_max_rmsd": "not executed; checkpoint unchanged",
           "features": {k: values[k] for k in FEATURES},
           "feature_order": list(FEATURES),
           "feature_labels": feature_labels(model),
           "missing_features": missing_feats,
           "scored": scored is not None,
           "logit": scored["logit"] if scored else None,
           "native_score": scored["native_score"] if scored else None,
           "imputed": scored["imputed"] if scored else None,
           "model": model_summary(model),
           "details": detail,
           "elapsed_s": round(time.time() - t0, 1),
           "semantics": ("descriptor computation + an uncalibrated operational score; no "
                         "retraining, rescaling or calibration is performed here")}
    out = Path(args.out).resolve() if args.out else work / "prediction.json"
    out.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    log(f"wrote {out}")

    print()
    print("=" * 68)
    print(f"MOF: {label}")
    labels = feature_labels(model)
    for i, k in enumerate(FEATURES):
        v = values[k]
        print(f"  {labels[i]:58s} {v if v is not None else 'MISSING'}")
    if scored:
        print(f"  {'mlp logit':58s} {scored['logit']!r}")
        print(f"  {'mlp raw native score (uncalibrated)':58s} {scored['native_score']!r}")
        if any(scored["imputed"]):
            print(f"  imputed inputs: {[FEATURES[i] for i, x in enumerate(scored['imputed']) if x]}")
    else:
        print("  no score: " + ", ".join(missing_feats) + " missing")
    print("=" * 68)
    if args.json:
        print(json.dumps(doc, indent=2, default=str))
    return 0


def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
