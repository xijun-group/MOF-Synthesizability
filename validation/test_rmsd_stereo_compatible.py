"""Tier priority and chemistry-boundary regressions; no QM calculations."""
from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'predict/rules'))
import rmsd_sa_target as selector

def rec(i,s,n=10):
    return dict(component_index=i,standardized_parent=s,status='ok',natoms=n)

class StereoTests(unittest.TestCase):
    def test_exact_precedes_larger_compatible_and_fallback(self):
        rows=[rec(0,'CC(O)F',8),rec(1,'C[C@H](O)F',20),rec(2,'CCCCCCCC',30)]
        self.assertEqual(selector.choose_component(rows,'CC(O)F',2)['component_index'],0)

    def test_compatible_precedes_larger_unrelated_and_uses_copy_tie(self):
        rows=[rec(0,'C[C@H](O)F',9),rec(1,'C[C@@H](O)F',10),rec(2,'CCCCCCCC',30),rec(3,'C[C@H](O)F',10)]
        self.assertEqual(selector.choose_component(rows,'CC(O)F',2)['component_index'],1)
        exact,compatible=selector.matching_components(rows,'CC(O)F')
        self.assertFalse(exact);self.assertEqual(len(compatible),3)

    def test_double_bond_unspecified_is_compatible(self):
        rows=[rec(0,'F/C=C/F'),rec(1,'F/C=C\\F')]
        self.assertEqual(len(selector.matching_components(rows,'FC=CF')[1]),2)

    def test_specified_stereo_is_not_relaxed(self):
        for target,candidate in [('C[C@H](O)F','C[C@@H](O)F'),('F/C=C/F','F/C=C\\F'),
                                 ('C[C@H](O)F','CC(O)F'),
                                 ('C[C@H](O)C(F)Cl','C[C@H](O)[C@H](F)Cl')]:
            with self.subTest(target=target,candidate=candidate):
                self.assertEqual(selector.matching_components([rec(0,candidate)],target),([],[]))
                self.assertEqual(selector.choose_component([rec(0,candidate)],target,0)['component_index'],0)

    def test_isotopes_charge_hydrogen_and_bonds_not_relaxed(self):
        for target,candidate in [('CC(O)F','[13CH3][C@H](O)F'),('CC(O)F','C[C@H]([O-])F'),
                                 ('CC(O)F','C[C@H](O)Cl'),('CC(O)F','CC(=O)F'),
                                 ('CC(O)F','C[C@](O)([2H])F')]:
            with self.subTest(candidate=candidate):
                self.assertEqual(selector.matching_components([rec(0,candidate)],target),([],[]))
        self.assertEqual(len(selector.matching_components([rec(0,'[13CH3][C@H](O)F')],'[13CH3]C(O)F')[1]),1)

if __name__=='__main__':unittest.main(verbosity=2)
