#!/usr/bin/env python3
"""Run the delivered MOFSimplify driver with exact PBC and local RAC reuse.

The package stores the PBC kernel alongside this launcher and the single
changed RAC function in ../acceleration/MOF_descriptors.py. Both patches
live in the child process; installed scientific environments remain intact.
"""
import importlib.util
from pathlib import Path
import runpy
import sys

from pbc_distance_optimized import compute_distance_matrix2
from molSimplify.Informatics.MOF import MOF_descriptors, PBC_functions


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Expected original MOFSimplify driver path')
    driver = str(Path(sys.argv[1]).resolve())
    PBC_functions.compute_distance_matrix2 = compute_distance_matrix2
    MOF_descriptors.compute_distance_matrix2 = compute_distance_matrix2
    patch = Path(__file__).resolve().parent.parent/'acceleration/MOF_descriptors.py'
    spec = importlib.util.spec_from_file_location('syn_rac_sbu_acceleration', patch)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load local RAC acceleration: '+str(patch))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # All other descriptor functions and their globals remain the installed
    # originals; only reuse of invocation-local invariants is introduced.
    MOF_descriptors.make_MOF_SBU_RACs = module.make_MOF_SBU_RACs
    sys.argv = [driver] + sys.argv[2:]
    sys.path.insert(0, str(Path(driver).parent))
    runpy.run_path(driver, run_name='__main__')


if __name__ == '__main__':
    main()
