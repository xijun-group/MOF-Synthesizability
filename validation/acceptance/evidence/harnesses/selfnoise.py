"""Intrinsic run-to-run noise of the original UMA call, and the size of the Cholesky-vs-eigh step change.
usage: selfnoise.py <cif> <opt.traj of an original relaxation>"""
import sys
from pathlib import Path
import numpy as np, torch
from ase.io import read
from ase.filters import FrechetCellFilter
from ase._4.optimize.bfgs import BFGSMethod
from scipy.linalg import cho_factor, cho_solve
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import UMATask
MODEL = str(Path.home() / "syn_perf_20261023/v3/local/mofsynth-fast/runtime/qmof-ehull/UMA/checkpoints/uma-s-1p1.pt")
atoms = read(sys.argv[1])
res = []
for k in range(2):  # two independent calculator instances, as two separate runs would have
    calc = FAIRChemCalculator.from_model_checkpoint(MODEL, task_name=UMATask.ODAC, device="cuda")
    for rep in range(3):
        a = atoms.copy(); a.calc = calc
        res.append((a.get_potential_energy(), a.get_forces().copy(), a.get_stress().copy()))
e = np.array([r[0] for r in res]); F = np.array([r[1] for r in res]); S = np.array([r[2] for r in res])
fs = np.abs(F).max()
print(f"{len(atoms)} atoms: self-noise over 6 calls: dE/atom max {np.ptp(e)/len(atoms):.2e} eV, dF max {np.ptp(F,axis=0).max():.2e} eV/A "
      f"(|F|max {fs:.2f}), dS max {np.ptp(S,axis=0).max():.2e}")
# Replay the original trajectory's Hessian and compare the two step formulas at every checkpoint.
traj = read(sys.argv[2], index=":")
xs = []
for t in traj:
    f = FrechetCellFilter(t); xs.append((f.get_positions().ravel(), -f.get_forces().ravel()))
H = BFGSMethod(70.0 * np.eye(len(xs[0][0]))); worst = 0.0; pd = 0
for i, ((x0, g0), (x1, g1)) in enumerate(zip(xs, xs[1:])):
    H.update(x1, g1, x0, g0)
    if i % 10: continue
    w, V = np.linalg.eigh(H.hessian); s_e = -V @ (g1 @ V / np.fabs(w))
    try:
        s_c = -cho_solve(cho_factor(H.hessian, lower=True), g1); pd += 1
        worst = max(worst, np.abs(s_c - s_e).max() / np.abs(s_e).max())
    except Exception:
        pass
print(f"Cholesky vs eigh step: max relative difference {worst:.2e} over {pd} positive-definite checkpoints of {len(xs)-1} steps")
