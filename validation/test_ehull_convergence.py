"""3.1.1 convergence and integration regressions; CPU-only, no model inference."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.units import GPa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'acceleration'))
from ehull_protocol import DEFAULTS, evaluate, matches_receipt, protocol
from ehull_convergence import combined_optimizer, relax_mof


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


driver = module('combined_predict_driver', ROOT / 'predict/predict_synthesizability.py')
batch = module('combined_batch_driver', ROOT / 'predict_batch.py')


class ConvergenceTests(unittest.TestCase):
    def test_invalid_parameters(self):
        for values in [dict(stress_GPa=0), dict(atom_fmax_eV_A=float('nan')),
                       dict(energy_window_steps=0), dict(consecutive_steps=2.5)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                protocol(**values)

    def test_window_detects_excursion_and_requires_five_updates(self):
        row = dict(atom_fmax_eV_A=.005, filter_fmax=.005,
                   stress_max_abs_principal_GPa=.01, energy_per_atom_eV=-1.)
        self.assertFalse(evaluate([dict(row) for _ in range(5)], DEFAULTS)[0]['energy'])
        history = [dict(row) for _ in range(6)]
        self.assertTrue(all(evaluate(history, DEFAULTS)[0].values()))
        history[2]['energy_per_atom_eV'] += .001
        self.assertFalse(evaluate(history, DEFAULTS)[0]['energy'])

    def test_tensor_stress_and_vector_force_not_component_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            atoms = bulk('Cu', cubic=True)
            stress = np.array([[.04, .04, 0], [.04, .04, 0], [0, 0, 0]]) * GPa
            forces = np.tile([.009, .009, 0], (len(atoms), 1))
            atoms.calc = SinglePointCalculator(atoms, energy=0., forces=forces, stress=stress)
            opt = combined_optimizer(BFGS)(FrechetCellFilter(atoms), convergence_protocol=DEFAULTS,
                  metrics_path=Path(folder)/'metrics.jsonl', logfile=None)
            opt.gradient_converged(np.zeros(3*(len(atoms)+3)))
            opt.metrics_file.close()
            self.assertFalse(opt.history[-1]['gates']['stress'])
            self.assertFalse(opt.history[-1]['gates']['atom_force'])

    def test_persistence_counts_steps_not_repeated_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            atoms = bulk('Cu', cubic=True)
            atoms.set_cell(atoms.cell*1.02, scale_atoms=True)
            atoms.rattle(.01, seed=123)
            atoms.calc = EMT()
            opt = combined_optimizer(BFGS)(FrechetCellFilter(atoms), convergence_protocol=DEFAULTS,
                  metrics_path=Path(folder)/'metrics.jsonl', logfile=None,
                  trajectory=str(Path(folder)/'opt.traj'))
            self.assertTrue(opt.run(fmax=.03, steps=500))
            length, streak = len(opt.history), opt.streak
            opt.converged(); opt.converged()
            self.assertEqual((len(opt.history), opt.streak), (length, streak))
            self.assertEqual(streak, 5)
            self.assertGreaterEqual(opt.nsteps, 9)
            with Trajectory(str(Path(folder)/'opt.traj')) as trajectory:
                self.assertEqual(len(trajectory), length)
            opt.metrics_file.close()

    def test_step_cap_writes_failure_and_does_not_return_energy(self):
        with tempfile.TemporaryDirectory() as folder:
            atoms = bulk('Cu', cubic=True)
            atoms.set_cell(atoms.cell*1.1, scale_atoms=True)
            with self.assertRaisesRegex(RuntimeError, 'No accepted Ehull'):
                relax_mof(atoms, calculator=EMT(), optimizer=BFGS, out_dir=folder, max_steps=1)
            receipt = json.loads((Path(folder)/'output/results.json').read_text())
            self.assertFalse(receipt['converged'])
            self.assertEqual(receipt['termination_reason'], 'max_steps')
            self.assertFalse(matches_receipt(receipt, DEFAULTS))

    def test_fixed_cell_cannot_claim_full_combined_protocol(self):
        with self.assertRaisesRegex(ValueError, 'variable-cell'):
            relax_mof(bulk('Cu'), calculator=EMT(), optimizer=BFGS, fix_cell=True)

    def test_legacy_ehull_cache_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder); cif=work/'input.cif'; cif.write_text('input')
            out=work/'qmof-ehull/ehull';out.mkdir(parents=True)
            (out/'test_uma-s-1p1_ehull.json').write_text('{"energy_above_hull":0.1}')
            receipt=work/'qmof-ehull/relaxations/test';receipt.mkdir(parents=True)
            (receipt/'results.json').write_text('{"converged":true}')
            with patch.object(driver, 'run') as run, self.assertRaises(SystemExit) as exc:
                driver.step_ehull(dict(ehull_root='/unused'), cif, 'test', work)
            self.assertEqual(exc.exception.code, 3)
            run.assert_not_called()

    def test_serial_hull_route_still_forwards_combined_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder);cif=work/'input.cif';cif.write_text('input')
            ctx=dict(ehull_root='/payload', ehull_python='/python', predict_dir=ROOT/'predict',
                     device='cpu', ehull_timeout=10, fast_ehull=False, ehull_solver='orig')
            with patch.object(driver,'run') as run, self.assertRaises(SystemExit):
                run.return_value.returncode=1
                driver.step_ehull(ctx,cif,'test',work)
            command=run.call_args.args[0]
            self.assertIn('--no-overlap-hull',command)
            self.assertIn('--consecutive-steps',command)
            self.assertIn('--stress-max-gpa',command)
            self.assertTrue(any(str(x).endswith('ehull_fast.py') for x in command))

    def test_batch_legacy_cache_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cif=Path(folder)/'input.cif';cif.write_text('input')
            self.assertFalse(batch.reusable_prediction({'features':{'Ehull':.1}},cif,[]))
            self.assertTrue(batch.reusable_prediction({},cif,['--baseline']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
