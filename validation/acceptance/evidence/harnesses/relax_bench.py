"""One production Ehull relaxation (qmof_thermo.relax_mof + get_energy_above_hull) with optional variants.

usage: relax_bench.py <cif> <label> <outdir> [--solver orig|chol] [--infer default|nockpt|nockpt_merge]

orig/default is exactly the production call chain of scripts/predict_ehull.py (run_single.sh settings:
uma-s-1p1, ODAC task, fmax 0.03, max_steps 10000, cuda, FrechetCellFilter + ASE BFGS).
"""
import argparse
import json
import sys
import time
from pathlib import Path

t_start = time.perf_counter()
import numpy as np
import torch
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from ase.optimize import BFGS
from ase._4.optimize.bfgs import BFGSMethod
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import UMATask, inference_settings_default
from qmof_thermo import get_energy_above_hull, relax_mof
from qmof_thermo.mlip import _resolve_local_model

STATS = {"chol": 0, "eigh_fallback": 0}


class CholBFGSMethod(BFGSMethod):
    """-H^-1 g by Cholesky when H is positive definite (then V|w|^-1 V^T = H^-1); original eigh otherwise."""

    def compute_step(self, gradient):
        if np.isfinite(self.hessian).all() and np.isfinite(gradient).all():
            try:
                factor = cho_factor(self.hessian, lower=True, check_finite=False)
            except LinAlgError:
                pass
            else:
                STATS["chol"] += 1
                return -cho_solve(factor, gradient, check_finite=False)
        STATS["eigh_fallback"] += 1
        return super().compute_step(gradient)


class CholBFGS(BFGS):
    def update(self, pos, forces, pos0, forces0):
        super().update(pos, forces, pos0, forces0)
        if not isinstance(self.state, CholBFGSMethod):
            self.state = CholBFGSMethod(self.state.hessian)

    def read(self):
        super().read()
        self.state = CholBFGSMethod(self.state.hessian)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif", type=Path)
    ap.add_argument("label")
    ap.add_argument("out", type=Path)
    ap.add_argument("--solver", choices=["orig", "chol"], default="orig")
    ap.add_argument("--infer", choices=["default", "nockpt", "nockpt_merge"], default="default")
    ap.add_argument("--model", default="uma-s-1p1")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t_import = time.perf_counter() - t_start

    sys.path.insert(0, str(Path(__import__("qmof_thermo").__file__).resolve().parents[2] / "scripts"))
    from predict_ehull import read_structure, resolve_model
    atoms = read_structure(a.cif)
    model = resolve_model(a.model)

    calculator = None
    if a.infer != "default":
        s = inference_settings_default()
        s.activation_checkpointing = False
        s.merge_mole = a.infer == "nockpt_merge"
        calculator = FAIRChemCalculator.from_model_checkpoint(
            name_or_path=_resolve_local_model(model), task_name=UMATask.ODAC,
            inference_settings=s, device="cuda")
    t0 = time.perf_counter()
    energy = relax_mof(atoms, label=a.label, model=model, fmax=0.03, max_steps=10000, device="cuda",
                       out_dir=a.out / "relaxations", optimizer=CholBFGS if a.solver == "chol" else BFGS,
                       calculator=calculator)
    t_relax = time.perf_counter() - t0
    t0 = time.perf_counter()
    thermo = get_energy_above_hull(atoms, energy, energy_type="ODAC_MLIP")
    t_hull = time.perf_counter() - t0
    summary = json.loads((a.out / "relaxations" / a.label / "results.json").read_text())
    doc = {"label": a.label, "solver": a.solver, "infer": a.infer, "energy_eV": float(energy), **thermo,
           "relax": summary, "natoms": len(atoms), "energy_per_atom": float(energy) / len(atoms),
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "solver_stats": dict(STATS),
           "wall": {"import_s": t_import, "relax_s": t_relax, "hull_s": t_hull,
                    "total_s": time.perf_counter() - t_start}}
    (a.out / "bench.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print(json.dumps({k: doc[k] for k in ("label", "solver", "infer", "energy_per_atom", "gpu")}
                     | {"ehull": thermo.get("energy_above_hull"), "nsteps": summary["nsteps"], "wall": doc["wall"]}, default=str))


if __name__ == "__main__":
    main()
