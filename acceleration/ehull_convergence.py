"""Combined convergence, retaining the selected BFGS step implementation.

Stress is the maximum absolute principal Cauchy stress at zero external pressure.
Requiring convergence is separate from finite restart/perturbation diagnostics.
"""
import json
import time
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.units import GPa

from ehull_protocol import DEFAULTS, evaluate, protocol


def combined_optimizer(base):
    class CombinedBFGS(base):
        def __init__(self, atoms, *, convergence_protocol, metrics_path, **kwargs):
            self.convergence_protocol = protocol(**convergence_protocol)
            self.history = []
            self.streak = 0
            self.started = time.monotonic()
            self.metrics_file = Path(metrics_path).open('w')
            super().__init__(atoms, **kwargs)

        def gradient_converged(self, gradient):
            if self.history and self.history[-1]['step'] == self.nsteps:
                return self.history[-1]['combined_converged']
            atoms = self.atoms.atoms
            energy = float(atoms.get_potential_energy())
            stress = atoms.get_stress(voigt=False)
            forces = atoms.get_forces()
            row = dict(step=self.nsteps, energy_eV=energy, energy_per_atom_eV=energy/len(atoms),
                       atom_fmax_eV_A=float(np.linalg.norm(forces, axis=1).max()),
                       filter_fmax=float(self.optimizable.gradient_norm(gradient)),
                       stress_max_abs_principal_GPa=float(np.abs(np.linalg.eigvalsh(stress)).max()/GPa),
                       stress_tensor_eV_A3=stress.tolist(), volume_A3=float(atoms.get_volume()),
                       elapsed_s=time.monotonic()-self.started)
            if not np.isfinite([energy, row['atom_fmax_eV_A'], row['filter_fmax'],
                                row['stress_max_abs_principal_GPa'], row['volume_A3']]).all():
                raise FloatingPointError('Nonfinite Ehull convergence metric')
            self.history.append(row)
            gates, span = evaluate(self.history, self.convergence_protocol)
            self.streak = self.streak + 1 if all(gates.values()) else 0
            row.update(gates=gates, energy_window_span_eV_atom=span,
                       consecutive_passes=self.streak,
                       combined_converged=self.streak >= self.convergence_protocol['consecutive_steps'])
            self.metrics_file.write(json.dumps(row, allow_nan=False)+'\n')
            self.metrics_file.flush()
            return row['combined_converged']

    return CombinedBFGS


def relax_mof(atoms, *, calculator, optimizer, label='output', model='uma-s-1p1',
              out_dir=Path('data/relaxations'), fmax=0.03, max_steps=10000,
              convergence_protocol=None, fix_cell=False, **unused):
    if fix_cell:
        raise ValueError('Combined Ehull convergence requires variable-cell relaxation and stress; --fix-cell is not supported by this protocol.')
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    settings = protocol(**(convergence_protocol or dict(DEFAULTS, filter_fmax=fmax)))
    if fmax != settings['filter_fmax']:
        raise ValueError('fmax and convergence_protocol.filter_fmax disagree')
    directory = (Path(out_dir)/label).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    atoms.calc = calculator
    cls = combined_optimizer(optimizer)
    opt = cls(FrechetCellFilter(atoms), convergence_protocol=settings,
              metrics_path=directory/'convergence.jsonl',
              trajectory=directory/'opt.traj', logfile=str(directory/'opt.log'))
    try:
        converged = bool(opt.run(fmax=fmax, steps=max_steps))
    finally:
        # ASE 3.29 manages its path-backed trajectory and Log per write.
        opt.metrics_file.close()
    last = opt.history[-1]
    write(directory/f'{label}.cif', atoms)
    result = dict(id=label, model=str(model), package_version='3.1.1',
                  convergence_protocol=settings, fmax_target=fmax,
                  atom_fmax_target=settings['atom_fmax_eV_A'], max_steps=max_steps,
                  fix_cell=False, converged=converged,
                  termination_reason='combined_converged' if converged else 'max_steps',
                  nsteps=opt.nsteps, final_energy=last['energy_eV'],
                  final_volume=last['volume_A3'], final_fmax=last['atom_fmax_eV_A'],
                  final_stress_max_abs_principal_GPa=last['stress_max_abs_principal_GPa'],
                  final_filter_fmax=last['filter_fmax'], final=last,
                  endpoint_restart_or_perturbation_check='not executed')
    (directory/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if not converged:
        raise RuntimeError(f'Ehull did not meet combined convergence in {max_steps} steps; see {directory / "results.json"}. No accepted Ehull is emitted.')
    return last['energy_eV']
