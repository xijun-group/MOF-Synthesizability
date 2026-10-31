"""
Module for MLIP handling.
"""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from ase.filters import FrechetCellFilter
from ase.io import write
from scipy_evd_bfgs import SciPyEVDBFGS as BFGS
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import UMATask
from monty.serialization import dumpfn

from qmof_thermo.constants import QMOF_ODAC_COMPATIBLE_ELEMENTS, UMA_ODAC_ELEMENTS

if TYPE_CHECKING:
    from typing import Literal

    from ase import Atoms
    from ase.optimize.optimize import Optimizer

LOGGER = getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_MODEL_ALIASES = {
    "uma-s-1p1": _REPO_ROOT / "UMA/checkpoints/uma-s-1p1.pt",
    "uma-s-1p2": _REPO_ROOT / "UMA/checkpoints/uma-s-1p2.pt",
    "uma-s-1": _REPO_ROOT / "UMA/checkpoints/uma-s-1.pt",
    "uma-m-1p1": _REPO_ROOT / "UMA/checkpoints/uma-m-1p1.pt",
    "esen-full": _REPO_ROOT / "OADC25/checkpoints/esen_sm_odac25_full.pt",
    "esen-filtered": _REPO_ROOT / "OADC25/checkpoints/esen_sm_odac25_filtered.pt",
}


def _resolve_local_model(model: str | Path) -> str | Path:
    if not isinstance(model, str):
        return model

    model_path = _LOCAL_MODEL_ALIASES.get(model)
    if model_path is None or not model_path.is_file():
        return model

    return model_path


def relax_mof(
    atoms: Atoms,
    label: str = "output",
    model: str | Path = "uma-s-1p1",
    uma_task_name: UMATask | None = UMATask.ODAC,
    fmax: float = 0.01,
    max_steps: int = 10000,
    optimizer: type[Optimizer] = BFGS,
    device: Literal["cpu", "cuda"] | None = None,
    out_dir: Path | str = Path("data/relaxations"),
    fix_cell: bool = False,
    scratch_dir: Path | str | None = None,
    calculator: FAIRChemCalculator | None = None,
) -> float:
    """
    Relax an ASE Atoms structure using an ODAC-trained MLIP.

    By default performs a full relaxation of both atomic positions and cell
    parameters using the BFGS optimizer with a FrechetCellFilter. Set
    ``fix_cell=True`` to relax only atomic positions while keeping the cell
    fixed. Outputs are saved to ``<out_dir>/<label>/`` including optimization
    log, trajectory, relaxed CIF file, and a JSON summary of results.

    Parameters
    ----------
    atoms
        ASE Atoms object to be relaxed.
    label
        Unique identifier for the relaxation job. Used to name output files.
    model
        Model name or path to checkpoint file. For UMA models, pass the model
        name (e.g., ``"uma-s-1p1"``) to load from Hugging Face, or provide a
        local checkpoint path. For eSEN models, pass the local checkpoint
        file path.
    uma_task_name
        Task name for UMA models. Set to "odac" for UMA models, or
        ``None`` for non-UMA models like eSEN.
    fmax
        Convergence criteria, set maximum force on atoms in in eV/Å. Optimization
        stops when maximum force on any atom falls below this value.
    max_steps
        Maximum number of optimization steps allowed.
    device
        Device to run calculation on, e.g., "cpu" or "cuda".
    out_dir
        Base directory for output files. Subdirectory ``<label>``
        created and stores all relaxation specific outputs.
    fix_cell
        If True, keep the unit cell fixed and relax only atomic positions.
        This is useful for models that do not implement stress, such as the
        eSEN checkpoints in this repo.
    scratch_dir
        Optional base directory for intermediate relaxation outputs. When
        provided, files are written to ``<scratch_dir>/<label>/`` instead of
        ``<out_dir>/<label>/``. The caller is responsible for copying the
        completed ``<scratch_dir>/<label>/`` directory to ``<out_dir>/<label>/``.
        This avoids writing step-by-step I/O to shared filesystems such as
        ``/fs1``. If ``None``, outputs are written directly to ``<out_dir>/<label>/``.
    calculator
        Optional pre-built calculator to reuse across multiple relaxations. When
        provided, the function skips building a new calculator from ``model``,
        which avoids repeated checkpoint loading. This is useful for long-running
        workers that process many structures on the same GPU. The caller is
        responsible for ensuring the calculator is on the correct ``device``.

    Returns
    -------
    float
        The final relaxed total energy in eV.

    Notes
    -----
    Following files are written to ``<out_dir>/<label>/`` (or
    ``<scratch_dir>/<label>/`` when ``scratch_dir`` is provided):
        - ``opt.log``: Optimization log
        - ``opt.traj``: Trajectory file with all optimization steps
        - ``<label>.cif``: Final relaxed structure in CIF format
        - ``results.json``: Summary including energy, volume, forces, and steps
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _resolve_local_model(model)

    mol_elements = set(atoms.get_chemical_symbols())

    unsupported = mol_elements - UMA_ODAC_ELEMENTS
    if unsupported:
        LOGGER.warning(
            "Structure contains elements not in the ODAC dataset: "
            f"{sorted(unsupported)}. MLIP predictions for these elements may be unreliable."
        )

    incompatible = mol_elements - QMOF_ODAC_COMPATIBLE_ELEMENTS
    if incompatible:
        LOGGER.warning(
            "Structure contains elements whose ODAC "
            f"pseudopotentials do not match QMOF's: {sorted(incompatible)}. "
            "The results may not be compatible between the MLIP and the QMOF-Thermo Database."
        )

    if calculator is None:
        atoms.calc = FAIRChemCalculator.from_model_checkpoint(
            name_or_path=model,
            task_name=uma_task_name if "uma" in str(model).lower() else None,
            device=device,
        )
    else:
        atoms.calc = calculator

    opt_subject = atoms if fix_cell else FrechetCellFilter(atoms)

    if scratch_dir is not None:
        out_dir = (Path(scratch_dir) / label).resolve()
    else:
        out_dir = (Path(out_dir) / label).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "opt.traj"

    opt = optimizer(
        opt_subject,  # type: ignore
        trajectory=traj_path,
    )
    converged = bool(opt.run(fmax=fmax, steps=max_steps))

    final_forces = atoms.get_forces()
    final_fmax = float(np.max(np.linalg.norm(final_forces, axis=1)))
    nsteps = opt.get_number_of_steps()
    final_volume = atoms.get_volume()
    final_energy = atoms.get_potential_energy()
    LOGGER.info(
        f"Energy: {final_energy}, Volume: {final_volume}, fmax: {final_fmax}, steps: {nsteps}"
    )

    cif_path = out_dir / f"{label}.cif"
    write(cif_path, atoms)
    LOGGER.info(f"Final relaxed structure written to: {cif_path}")

    summary = {
        "id": label,
        "model": model,
        "fmax_target": fmax,
        "max_steps": max_steps,
        "fix_cell": fix_cell,
        "converged": converged,
        "nsteps": nsteps,
        "final_energy": final_energy,
        "final_volume": final_volume,
        "final_fmax": final_fmax,
    }
    summary_path = out_dir / "results.json"
    dumpfn(summary, summary_path)
    LOGGER.info(f"Summary written to: {summary_path}")

    return final_energy
