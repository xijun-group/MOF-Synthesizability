"""
Module for calculating energy above hull.
"""

from __future__ import annotations

import json
from functools import lru_cache
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from ase import Atoms
from monty.json import MontyDecoder
from monty.serialization import loadfn
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry
from pymatgen.io.ase import AseAtomsAdaptor

from qmof_thermo.constants import QMOF_ELEMENTS, QMOF_ODAC_COMPATIBLE_ELEMENTS

_DEFAULT_PD_JSON = Path(__file__).parent.resolve() / "patched_phase_diagram.json"

LOGGER = getLogger(__name__)

if TYPE_CHECKING:
    from pymatgen.core import Composition, Structure


class HullOutput(TypedDict):
    energy_above_hull: float
    energy_below_reference_hull: float
    formation_energy: float
    decomposition_products: dict[str, float]


@lru_cache(maxsize=2)
def _load_phase_diagram_version(
    serialized_phase_diagram: str,
    file_identity: tuple[int, int, int, int, int],
) -> PatchedPhaseDiagram:
    # ``file_identity`` is deliberately part of the cache key. A replacement or
    # in-place edit must not reuse the previous reference hull.
    try:
        return loadfn(serialized_phase_diagram)
    except KeyError as exc:
        if exc.args != ("computed_data",):
            raise

        with Path(serialized_phase_diagram).open() as f:
            data = json.load(f)

        if data.get("@class") != "PatchedPhaseDiagram" or "all_entries" not in data:
            raise

        decoder = MontyDecoder()
        entries = [decoder.process_decoded(entry) for entry in data["all_entries"]]
        return PatchedPhaseDiagram(entries)


def _load_phase_diagram(serialized_phase_diagram: Path | str) -> PatchedPhaseDiagram:
    """Reuse a fixed reference hull within one process, invalidating on edits."""
    path = Path(serialized_phase_diagram).resolve()
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return _load_phase_diagram_version(str(path), identity)


def _get_hull_output(
    ppd: PatchedPhaseDiagram, composition: Composition, energy: float
) -> HullOutput:
    """Evaluate a candidate against a fixed reference hull.

    ``energy_above_hull`` retains the raw signed offset from the fixed DFT
    reference hull. A negative value therefore means that the model-predicted
    candidate lies below that fixed hull. ``energy_below_reference_hull`` stores
    the same downward offset as a positive convenience field.
    """
    entry = PDEntry(composition, energy)
    decomp, reference_hull_offset = ppd.get_decomp_and_e_above_hull(
        entry, allow_negative=True
    )

    if reference_hull_offset is None:
        msg = f"Could not compute energy above hull for composition {composition.reduced_formula}."
        raise ValueError(msg)

    return {
        "energy_above_hull": float(reference_hull_offset),
        "energy_below_reference_hull": float(max(-reference_hull_offset, 0.0)),
        "formation_energy": float(ppd.get_form_energy_per_atom(entry)),
        "decomposition_products": {
            decomp_entry.composition.reduced_formula: float(amount)
            for decomp_entry, amount in decomp.items()
        },
    }


def get_energy_above_hull(
    struct: Structure | Atoms,
    energy: float,
    *,
    energy_type: Literal["DFT", "ODAC_MLIP"],
    serialized_phase_diagram: Path | str = _DEFAULT_PD_JSON,
) -> HullOutput:
    """
    Calculate the energy above hull for a structure with a given total energy.

    Parameters
    ----------
    struct
        Input structure as either a pymatgen Structure or ASE Atoms object.
        If an Atoms object is provided, it will be converted to a Structure.
    energy
        Total relaxed energy of the structure in eV.
    energy_type
        How the energy data was sourced, either from "DFT" or "ODAC_MLIP"
    serialized_phase_diagram
        Path to the directory containing the precomputed PatchedPhaseDiagram.
        Default is the phase diagram bundled with the package.

    Returns
    -------
    dict
        A dictionary containing energy above the convex hull, formation energy, and decomposition products.
    """

    energy_type = energy_type.lower()
    if energy_type not in ("dft", "odac_mlip"):
        raise ValueError(
            f"Unsupported energy_type of {energy_type} from one of 'DFT' or 'ODAC_MLIP'"
        )

    if isinstance(struct, Atoms):
        struct = AseAtomsAdaptor.get_structure(struct)

    mol_elements = {str(el) for el in struct.composition.elements}

    out_of_chemical_space = mol_elements - QMOF_ELEMENTS
    if out_of_chemical_space:
        LOGGER.warning(
            "Your structure contains elements that are not present in the QMOF chemical space: "
            f"{sorted(out_of_chemical_space)}. The convex hull phase diagram will be incomplete."
        )

    if energy_type == "odac_mlip" and (
        incompatible := mol_elements - QMOF_ODAC_COMPATIBLE_ELEMENTS
    ):
        LOGGER.warning(
            "Structure contains elements whose ODAC "
            f"pseudopotentials do not match QMOF's: {sorted(incompatible)}. "
            "If `energy` is obtained from an ODAC MLIP, the results will likely not be compatible with "
            "the QMOF DFT reference data."
        )

    ppd = _load_phase_diagram(serialized_phase_diagram)

    return _get_hull_output(ppd, struct.composition, energy)
