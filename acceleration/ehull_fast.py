"""Ehull with the 3.1.1 combined convergence protocol and optional hull overlap.

Default: Cholesky BFGS, unchanged default fp32 inference, variable-cell relaxation.
All force/stress/energy gates must hold for five consecutive optimizer steps.
No accepted Ehull JSON is emitted when the step cap is reached without convergence.
--no-overlap-hull disables preloading only; it retains the same stopping protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

from ehull_protocol import protocol
from ehull_convergence import relax_mof

T0 = time.perf_counter()


def _hull_child(conn) -> None:
    try:
        from qmof_thermo import hull
        loader = getattr(hull, "_load_phase_diagram", None)
        if loader is not None:
            loader(hull._DEFAULT_PD_JSON)  # warms the fixed-hull cache when the overlay provides one
        request = conn.recv()
        if request is None:
            return
        atoms, energy, energy_type = request
        conn.send(("ok", hull.get_energy_above_hull(atoms, energy, energy_type=energy_type)))
    except BaseException as exc:  # reported to the parent, which raises it
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _cholesky_bfgs():
    import numpy as np
    from scipy.linalg import LinAlgError, cho_factor, cho_solve
    from ase.optimize import BFGS
    from ase._4.optimize.bfgs import BFGSMethod

    class CholBFGSMethod(BFGSMethod):
        def compute_step(self, gradient):
            if np.isfinite(self.hessian).all() and np.isfinite(gradient).all():
                try:
                    factor = cho_factor(self.hessian, lower=True, check_finite=False)
                except LinAlgError:
                    pass
                else:
                    return -cho_solve(factor, gradient, check_finite=False)
            return super().compute_step(gradient)

    class CholBFGS(BFGS):
        def update(self, pos, forces, pos0, forces0):
            super().update(pos, forces, pos0, forces0)
            if not isinstance(self.state, CholBFGSMethod):
                self.state = CholBFGSMethod(self.state.hessian)

        def read(self):
            super().read()
            self.state = CholBFGSMethod(self.state.hessian)

    return CholBFGS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cif", type=Path)
    parser.add_argument("--scripts", type=Path, required=True, help="qmof-ehull/scripts directory")
    parser.add_argument("--model", default="uma-s-1p1")
    parser.add_argument("--energy-type", choices=["DFT", "ODAC_MLIP"], default="ODAC_MLIP")
    parser.add_argument("--label", default=None)
    parser.add_argument("--fmax", type=float, default=0.03, help="filtered generalized-force limit")
    parser.add_argument("--atom-fmax", type=float, default=0.01)
    parser.add_argument("--stress-max-gpa", type=float, default=0.05)
    parser.add_argument("--energy-window-steps", type=int, default=5)
    parser.add_argument("--energy-span-per-atom", type=float, default=1e-5)
    parser.add_argument("--consecutive-steps", type=int, default=5)
    parser.add_argument("--overlap-hull", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/relaxations"))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--fix-cell", action="store_true")
    parser.add_argument("--solver", choices=["orig", "chol"], default="chol")
    parser.add_argument("--inference", choices=["default", "nockpt_merge"], default="default")
    parser.add_argument("--timing-out", type=Path, default=None, help="write phase wall times (JSON)")
    args = parser.parse_args()
    if args.fix_cell:
        parser.error("combined convergence requires a variable cell and stress; --fix-cell is unavailable")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    try:
        settings = protocol(atom_fmax_eV_A=args.atom_fmax, filter_fmax=args.fmax,
                            stress_GPa=args.stress_max_gpa, energy_window_steps=args.energy_window_steps,
                            energy_span_eV_atom=args.energy_span_per_atom, consecutive_steps=args.consecutive_steps)
    except ValueError as exc:
        parser.error(str(exc))
    timing = {}

    def mark(name):
        timing[name] = round(time.perf_counter() - T0, 3)

    parent_conn = child = None
    if args.overlap_hull:
        parent_conn, child_conn = mp.Pipe()
        child = mp.get_context("fork").Process(target=_hull_child, args=(child_conn,), daemon=True)
        child.start()
        child_conn.close()
    try:
        sys.path.insert(0, str(args.scripts.resolve()))
        from predict_ehull import read_structure, resolve_model
        from qmof_thermo import set_log_level
        from qmof_thermo.mlip import BFGS as OriginalBFGS

        set_log_level("INFO")
        mark("imports")
        atoms = read_structure(args.cif)
        label = args.label or args.cif.stem
        model = resolve_model(args.model)
        extra = {"optimizer": OriginalBFGS}
        if args.solver == "chol":
            extra["optimizer"] = _cholesky_bfgs()
        # The calculator relax_mof would build (same checkpoint, task, device), built here so model
        # loading is timed apart from the relaxation; nockpt_merge only changes inference settings.
        import torch
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit.api.inference import UMATask, inference_settings_default
        from qmof_thermo.mlip import _resolve_local_model
        inference_settings = inference_settings_default()
        if args.inference == "nockpt_merge":
            inference_settings.activation_checkpointing = False
            inference_settings.merge_mole = True
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        resolved = _resolve_local_model(model)
        extra["calculator"] = FAIRChemCalculator.from_model_checkpoint(
            name_or_path=resolved, task_name=UMATask.ODAC if "uma" in str(resolved).lower() else None,
            inference_settings=inference_settings, device=device)
        mark("model_loaded")
        energy = relax_mof(atoms, label=label, model=model, fmax=args.fmax, max_steps=args.max_steps,
                           device=args.device, out_dir=args.out_dir, fix_cell=args.fix_cell,
                           convergence_protocol=settings, **extra)
        mark("relaxed")

        relaxed = atoms.copy()  # composition is what the hull reads; the calculator stays here
        if child is None:
            from qmof_thermo.hull import get_energy_above_hull
            thermo = get_energy_above_hull(relaxed, energy, energy_type=args.energy_type)
        else:
            parent_conn.send((relaxed, energy, args.energy_type))
            status, thermo = parent_conn.recv()
            if status != "ok":
                raise RuntimeError("hull evaluation failed: " + thermo)
        mark("hull")
    except BaseException:
        if child is not None and child.is_alive():
            child.kill()
        raise
    finally:
        if child is not None:
            child.join(timeout=60)
        if parent_conn is not None:
            parent_conn.close()

    receipt_path = args.out_dir / label / "results.json"
    receipt = json.loads(receipt_path.read_text())
    provenance = dict(input_sha256=hashlib.sha256(args.cif.read_bytes()).hexdigest(),
                      run_configuration=dict(requested_model=args.model, solver=args.solver,
                                             inference=args.inference, energy_type=args.energy_type))
    receipt.update(provenance)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    result = {"cif": str(args.cif), "label": label, "model": model, "energy_eV": float(energy), **thermo,
              "package_version": "3.1.1", "relaxation_converged": True,
              "convergence_protocol": settings, **provenance}
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    if args.timing_out is not None:
        nsteps = json.loads((args.out_dir / label / "results.json").read_text()).get("nsteps")
        timing.update(nsteps=nsteps, solver=args.solver, inference=args.inference)
        if nsteps:
            timing["relax_s_per_step"] = round((timing["relaxed"] - timing["model_loaded"]) / nsteps, 4)
        args.timing_out.write_text(json.dumps(timing, indent=2) + "\n")


if __name__ == "__main__":
    main()
