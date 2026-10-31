"""Same full DSYEVD BFGS solve through SciPy for the pinned ASE 3.29 stack.

The eigenvector layout and both original matrix-vector operations are retained.
Nonfinite inputs use ASE's original implementation and its original behavior.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from ase.optimize import BFGS
try:
    from ase._4.optimize.bfgs import BFGSMethod
except ImportError:
    BFGSMethod = None

BFGS_ACCEL_AVAILABLE = BFGSMethod is not None
if BFGSMethod is None:
    # Existing-prefix recipients may have an older ASE. Keep their original
    # optimizer rather than applying an incompatible private API patch.
    SciPyEVDBFGS = BFGS
else:
    class SciPyEVDBFGSMethod(BFGSMethod):
        def compute_step(self, gradient: np.ndarray) -> np.ndarray:
            if not (np.isfinite(self.hessian).all() and np.isfinite(gradient).all()):
                return super().compute_step(gradient)
            omega, vectors = eigh(self.hessian, lower=True, driver='evd',
                                  check_finite=False, overwrite_a=False)
            vectors = np.asarray(vectors, order='C')
            return -vectors @ (gradient @ vectors / np.fabs(omega))


    class SciPyEVDBFGS(BFGS):
        def update(self, pos, forces, pos0, forces0):
            super().update(pos, forces, pos0, forces0)
            if not isinstance(self.state, SciPyEVDBFGSMethod):
                self.state = SciPyEVDBFGSMethod(self.state.hessian)

        def read(self):
            super().read()
            self.state = SciPyEVDBFGSMethod(self.state.hessian)
