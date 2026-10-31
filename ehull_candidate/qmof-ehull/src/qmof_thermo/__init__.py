from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qmof_thermo.hull import get_energy_above_hull
    from qmof_thermo.mlip import relax_mof
    from qmof_thermo.phase_diagram import setup_phase_diagrams

__all__ = [
    "get_energy_above_hull",
    "relax_mof",
    "set_log_level",
    "setup_phase_diagrams",
]

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    """Load only the scientific stack requested by the caller.

    Hull-only postprocessing must not load the independent Torch/Fairchem stack.
    The exported function objects themselves are unchanged.
    """
    modules = {
        "get_energy_above_hull": "hull",
        "relax_mof": "mlip",
        "setup_phase_diagrams": "phase_diagram",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{modules[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def set_log_level(level=logging.INFO):
    logger.setLevel(level)
