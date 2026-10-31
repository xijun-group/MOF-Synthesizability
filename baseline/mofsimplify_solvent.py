#!/usr/bin/env python3
"""Run the project's MOFSimplify predictor against the installed MOFSimplify checkout.

`predict_slt.py` is the project's own full-precision variant of MOFSimplify's batch
predictor: it reports `solvent_flag` as a float (the value the delivered feature column
carries) where `predict_batch.py` rounds it to two decimals. It is shipped verbatim with
the bundle and reads the checkout root from its module-level constant MOFSIMPLIFY, which
on the source machine is `~/MOFSimplify/`.

This shim is the only new code: it imports that file unchanged, overrides the constant
with MOFSIMPLIFY_ROOT (the checkout the bundle installed) and calls its main() with the
given arguments. No prediction code is re-implemented here.

Usage
-----
    python mofsimplify_solvent.py --predict-slt PATH/predict_slt.py \
        --desc-dir DESC --out predictions.csv [--shard 0 --nshards 1]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys


def load(path: str):
    spec = importlib.util.spec_from_file_location("bundle_predict_slt", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bundle_predict_slt"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predict-slt", required=True)
    ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--root", default=os.environ.get("MOFSIMPLIFY_ROOT", ""))
    a = ap.parse_args()

    if not a.root:
        raise SystemExit("MOFSIMPLIFY_ROOT is not set and --root was not given")
    root = a.root if a.root.endswith(os.sep) else a.root + os.sep

    mod = load(a.predict_slt)
    before = getattr(mod, "MOFSIMPLIFY", None)
    mod.MOFSIMPLIFY = root
    print(f"[mofsimplify_solvent] MOFSIMPLIFY {before!r} -> {root!r}", flush=True)

    argv, sys.argv = sys.argv, [a.predict_slt, "--desc-dir", a.desc_dir, "--out", a.out,
                                "--shard", str(a.shard), "--nshards", str(a.nshards)]
    try:
        rc = mod.main()
    finally:
        sys.argv = argv
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
