#!/usr/bin/env python3
"""Launch the additive optimization overlay against an existing bundle runtime."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def value(argv, name):
    found = None
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            found = argv[i + 1]
        elif token.startswith(name + "="):
            found = token.split("=", 1)[1]
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
            epilog="Scientific arguments are forwarded: --cif FILE --label NAME --work DIR --plan --skip ehull --from-features FILE. Run bash install.sh first, or use --reuse with an existing runtime.")
    config = json.loads((ROOT / "runtime.json").read_text()) if (ROOT / "runtime.json").exists() else {}
    ap.add_argument("--prefix", default=os.environ.get("MOFENV_PREFIX") or config.get("prefix") or str(ROOT / "runtime"))
    ap.add_argument("--baseline", action="store_true", help="original serial stages and distance kernel")
    ap.add_argument("--fast-hull", action=argparse.BooleanOptionalAction, default=True,
                    help="isolate the verified lazy-import and fixed-hull cache overlay")
    ap.add_argument("--fast-bfgs", action=argparse.BooleanOptionalAction, default=False,
                    help="same full LAPACK EVD through the verified SciPy implementation")
    ap.add_argument("--zeo-network", help="override the Zeo++ executable")
    ap.add_argument("--cpu-list", help="allowed logical CPUs, e.g. 472-479")
    ap.add_argument("--cpus", type=int, default=8, help="CPU affinity budget when --cpu-list is omitted")
    ap.add_argument("--gpu-id", help="CUDA device index or UUID; otherwise preserve CUDA_VISIBLE_DEVICES")
    a, rest = ap.parse_known_args()
    prefix = Path(a.prefix).expanduser().resolve()
    python = prefix / "mofsynth-env/bin/python"
    if not python.exists():
        raise SystemExit(f"Runtime not installed at {prefix}. Run bash install.sh, or bash install.sh --reuse --prefix /existing/runtime --ehull-python /existing/gpu-env/bin/python")
    bwrap = shutil.which("bwrap") or (str(ROOT / "tools/bwrap") if (ROOT / "tools/bwrap").exists() else None)
    if not bwrap:
        raise SystemExit("bubblewrap is required; install the OS bubblewrap package before using this Linux bundle")

    env = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    for line in (prefix / "paths.env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            env[key.strip()] = val.strip()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
               OMP_STACKSIZE="256M", PYTHONDONTWRITEBYTECODE="1", LC_ALL="C",
               MOFENV_PREFIX=str(prefix), MPLCONFIGDIR="/tmp/mplconfig",
               FAIRCHEM_CACHE_DIR="/tmp/fairchem", CUDA_MPS_PIPE_DIRECTORY="/tmp/no-mps")
    env.setdefault("PYTHONHASHSEED", "0")
    if a.zeo_network:
        env["ZEO_NETWORK"] = str(Path(a.zeo_network).expanduser().resolve())
    elif not a.baseline:
        env["ZEO_NETWORK"] = str(ROOT / "acceleration/network")
    if a.fast_bfgs and not a.baseline:
        env["PYTHONPATH"] = str(ROOT / "acceleration")
    env["PATH"] = ":".join(str(prefix / d) for d in
                    ("xtb/bin", "mofsynth-env/bin", "java/bin", "mofid-build/bin")) + ":" + env.get("PATH", "")
    if Path(bwrap).parent == ROOT / "tools":
        env["PATH"] = str(ROOT / "tools") + ":" + env["PATH"]
        env["LD_LIBRARY_PATH"] = str(ROOT / "tools/lib")
    if a.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = a.gpu_id
    if a.cpu_list:
        cpus = set()
        for part in a.cpu_list.split(","):
            ends = [int(x) for x in part.split("-")]
            cpus.update(range(ends[0], ends[-1] + 1))
        if not cpus or not cpus <= os.sched_getaffinity(0):
            raise SystemExit("--cpu-list must be a nonempty subset of the current allowed CPUs")
        os.sched_setaffinity(0, cpus)
    else:
        if a.cpus < 1:
            raise SystemExit("--cpus must be positive")
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-a.cpus:])

    # Resolve all supplied file paths before the private home and cwd are created.
    read_paths, write_paths = [ROOT], []
    path_flags = {"--cif", "--work", "--out", "--from-features", "--model-dir", "--ehull-python"}
    i = 0
    while i < len(rest):
        flag = rest[i].split("=", 1)[0]
        if flag in path_flags:
            inline = "=" in rest[i]
            raw = rest[i].split("=", 1)[1] if inline else rest[i + 1]
            # Resolving a venv's Python symlink would bypass that venv.
            path = (Path(raw).expanduser().absolute() if flag == "--ehull-python"
                    else Path(raw).expanduser().resolve())
            if inline:
                rest[i] = flag + "=" + str(path)
            else:
                rest[i + 1] = str(path)
            if flag == "--work":
                path.mkdir(parents=True, exist_ok=True)
                write_paths.append(path)
            elif flag == "--out":
                path.parent.mkdir(parents=True, exist_ok=True)
                write_paths.append(path.parent)
            else:
                read_paths.append(path if path.is_dir() else path.parent)
            i += 1 if inline else 2
        else:
            i += 1

    if value(rest, "--cif") and not value(rest, "--work"):
        label = value(rest, "--label") or Path(value(rest, "--cif")).stem
        if Path(label).name != label or label in (".", ".."):
            raise SystemExit("label must be a single path component")
        work = ROOT / "runs" / label
        work.mkdir(parents=True, exist_ok=True)
        rest += ["--work", str(work)]
        write_paths.append(work)
    if not value(rest, "--ehull-python"):
        # The installed optional pyenv is CPU-only on XJ2. Preserve explicit
        # caller selection, otherwise use the independently verified GPU env.
        gpu_python = Path(config.get("ehull_python") or str(prefix.parent / "gpu-env/bin/python"))
        requested = os.environ.get("EHULL_PYTHON")
        selected = Path(requested).expanduser().absolute() if requested and Path(requested).is_file() else gpu_python
        if selected.exists():
            rest += ["--ehull-python", str(selected)]
            read_paths.append(selected.parent.parent)

    defaults = ["--workers", "2"]
    if not a.baseline:
        defaults += ["--stage-jobs", "3", "--fast-pbc", "--fast-rmsd",
                     "--fast-sbu", "--early-stages", "--fast-ehull", "--ehull-solver", "chol"]
    driver = ROOT / ("baseline" if a.baseline else "predict") / "predict_synthesizability.py"
    command = [str(python), str(driver),
               "--prefix", str(prefix), *defaults, *rest]
    ns = [bwrap, "--die-with-parent", "--unshare-pid", "--ro-bind", "/", "/",
          "--dev-bind", "/dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
          "--tmpfs", str(Path.home())]
    # Hide obsolete source-machine paths before restoring the selected runtime.
    # This also works when the recipient's own home is /home/cheyue.
    for owner_path in ("/home/cheyue", "/share/data/cheyue"):
        if Path(owner_path).exists() and Path(owner_path) != Path.home():
            ns += ["--tmpfs", owner_path]
    ns += ["--ro-bind", str(prefix.parent), str(prefix.parent)]
    # Keep the CUDA driver's JIT cache that the private home would hide. GPUs whose kernels are
    # partly shipped as PTX (e.g. sm_120) otherwise recompile them in every run, which more than
    # doubled the per-step relaxation time on an RTX 5090. Cached binaries are the same compilation.
    nv_cache = Path(env.get("CUDA_CACHE_PATH") or Path.home() / ".nv/ComputeCache")
    try:
        nv_cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    if nv_cache.is_dir() and os.access(nv_cache, os.W_OK):
        env["CUDA_CACHE_PATH"] = str(nv_cache)
        ns += ["--bind", str(nv_cache), str(nv_cache)]
    for path in dict.fromkeys(read_paths):
        ns += ["--ro-bind", str(path), str(path)]
    for path in dict.fromkeys(write_paths):
        ns += ["--bind", str(path), str(path)]
    ns += ["--ro-bind", str(prefix / "mofid-build"), "/tmp/mofid-build"]
    if a.fast_hull and not a.baseline:
        for filename in ("__init__.py", "hull.py"):
            src = ROOT / "ehull_candidate/qmof-ehull/src/qmof_thermo" / filename
            dst = prefix / "qmof-ehull/src/qmof_thermo" / filename
            ns += ["--ro-bind", str(src), str(dst)]
    if a.fast_bfgs and not a.baseline:
        ns += ["--ro-bind", str(ROOT / "acceleration/mlip.py"),
               str(prefix / "qmof-ehull/src/qmof_thermo/mlip.py")]
    ns += ["--chdir", str(ROOT), "--", *command]
    os.execvpe(ns[0], ns, env)


if __name__ == "__main__":
    main()
