"""Target-selection regressions; no xTB or model scoring."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'predict/rules'))
import rmsd_sa_target as selector

spec = importlib.util.spec_from_file_location('samax_driver', ROOT / 'predict/predict_synthesizability.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def score(values):
    return {'complete': True, 'per_fragment': values, 'sascore': max(values.values())}


class TargetTests(unittest.TestCase):
    def test_highest_sa_not_longest_identity(self):
        result = selector.select_parent(score({'CCCCCCCCCCCC': 1.0, 'C1CC1': 3.0}))
        self.assertEqual(result['target_parent'], 'C1CC1')
        self.assertEqual(result['target_sa'], 3.0)

    def test_exact_tie_is_order_independent(self):
        a, b = 'C1CC1', 'CCO'
        r1 = selector.select_parent(score({a: 2.0, b: 2.0}))
        r2 = selector.select_parent(score({b: 2.0, a: 2.0}))
        self.assertEqual(r1, r2)
        self.assertEqual(r1['target_parent'], min(a, b))
        self.assertEqual(r1['sa_tied_parents'], sorted([a, b]))

    def test_incomplete_and_nonfinite_rejected(self):
        for bad in [None, {}, {'complete': False},
                    {'complete': True, 'per_fragment': {'CC': None}},
                    {'complete': True, 'per_fragment': {'CC': float('nan')}},
                    {'complete': True, 'per_fragment': {'CC': float('inf')}},
                    {'complete': True, 'per_fragment': {'CC': 2}, 'sascore': 1}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                selector.select_parent(bad)

    def test_geometry_copy_only_within_exact_target(self):
        records = [dict(component_index=0, standardized_parent='other', status='ok', natoms=100),
                   dict(component_index=1, standardized_parent='target', status='ok', natoms=8),
                   dict(component_index=2, standardized_parent='target', status='ok', natoms=10),
                   dict(component_index=3, standardized_parent='target', status='ok', natoms=10)]
        self.assertEqual(selector.choose_component(records, 'target')['component_index'], 2)
        self.assertEqual(selector.choose_component(records, 'missing', 0)['component_index'], 0)

    def test_fallback_includes_unstandardizable_geometry_and_ties(self):
        records = [dict(component_index=2, standardized_parent=None, status='failed', natoms=20),
                   dict(component_index=1, standardized_parent='other', status='ok', natoms=20),
                   dict(component_index=0, standardized_parent='other', status='ok', natoms=10)]
        self.assertEqual(selector.choose_component(records, 'missing', 1)['component_index'], 1)
        with self.assertRaises(ValueError):
            selector.choose_component(records, 'missing', 0 if records[0]['natoms'] < 20 else 99)
        records[0]['natoms'] = 21
        self.assertEqual(selector.choose_component(records, 'missing', 2)['component_index'], 2)
        with self.assertRaises(ValueError):
            selector.choose_component([], 'missing')

    def test_legacy_cache_cannot_override_new_target(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            cif = work / 'input.cif'
            cif.write_text('test input')
            (work / 'rmsd').mkdir()
            (work / 'rmsd/rmsd.json').write_text(json.dumps({'rmsd_a': 0.1}))
            with patch.object(driver, 'step_mofid') as decompose:
                with self.assertRaises(SystemExit) as exc:
                    driver.step_rmsd({'rules': ROOT / 'predict/rules'}, cif, work, 10,
                                     sa_scores=score({'CCO': 2.0}))
                self.assertEqual(exc.exception.code, 3)
                decompose.assert_not_called()

    def test_partial_decomposition_files_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            out = work / 'mofid'
            (out / 'Output/MetalOxo').mkdir(parents=True)
            for filename in ['linker_stats.txt', 'linkers.cif']:
                (out / 'Output/MetalOxo' / filename).write_text('partial output')
            (out / 'extraction_incomplete.json').write_text('{"state":"started"}')
            for method in [driver.step_mofid, driver.step_mofid_fragments]:
                with self.subTest(method=method.__name__), self.assertRaises(SystemExit) as exc:
                    method({}, work / 'input.cif', work)
                self.assertEqual(exc.exception.code, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
