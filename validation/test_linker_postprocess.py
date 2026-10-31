#!/usr/bin/env python3
"""Chemistry and pipeline regressions; run with the runtime RDKit Python."""
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "predict/rules"))
import standardize_linker_smiles as std
import canonical_linker_qm as qm
from rdkit import Chem, RDConfig
sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer

spec = importlib.util.spec_from_file_location("driver", ROOT / "predict/predict_synthesizability.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

TETRAZOLES = [
    "c1ccc2c([C]3N=NN=N3)cc(C3=NN=N[N]3)c-2cc1",
    "c1ccc2c([C]3N=NN=N3)cc(C3=N[N]N=N3)c-2cc1",
    "c1ccc2c(C3=N[N]N=N3)cc(C3=NN=N[N]3)c-2cc1",
    "c1ccc2c(C3=N[N]N=N3)cc(C3=N[N]N=N3)c-2cc1",
]
TETRAZOLE_PARENT = "c1ccc2c(-c3nn[nH]n3)cc(-c3nn[nH]n3)c-2cc1"
FAMILIES = {
    "benzoic_acid": ["O=C(O)c1ccccc1", "[O-]C(=O)c1ccccc1", "[O]C(=O)c1ccccc1", "[O][C]([O])c1ccccc1"],
    "terephthalic_acid": ["O=C(O)c1ccc(C(=O)O)cc1", "[O]C(=O)c1ccc(C([O])=O)cc1", "[O-]C(=O)c1ccc(C(=O)[O-])cc1"],
    "oxalic_acid": ["O=C(O)C(=O)O", "[O]C(=O)C([O])=O", "[O][C]([O])[C]([O])[O]"],
    "tetrazole": ["Cc1nnn[nH]1", "Cc1nn[nH]n1", "Cc1nnn[n-]1", "C[C]1N=NN=N1", "CC1=N[N]N=N1"],
    "imidazole": ["Cc1ncc[nH]1", "Cc1ncc[n-]1", "CC1=NC=C[N]1"],
    "pyrazole": ["Cc1ccn[nH]1", "Cc1ccn[n-]1", "CC1=CC=N[N]1"],
    "triazole": ["c1nnc[nH]1", "c1nnc[n-]1", "C1=NN=C[N]1"],
    "triazole_adjacent_N": ["c1cn[nH]n1", "c1cn[n-]n1", "C1=CN=N[N]1"],
    "pyrrole": ["c1cc[nH]c1", "c1cc[n-]c1", "C1=CC=C[N]1"],
    "benzimidazole": ["c1ccc2[nH]cnc2c1", "c1ccc2[n-]cnc2c1"],
    "malformed_pyrazolate": ["c1[n-][nH]cc1", "c1cn[nH]c1", "c1cn[n-]c1"],
    "phenol": ["Oc1ccccc1", "[O-]c1ccccc1", "[O]c1ccccc1"],
    "thiophenol": ["Sc1ccccc1", "[S-]c1ccccc1", "[S]c1ccccc1"],
    "aniline": ["Nc1ccccc1", "[N]c1ccccc1"],
    "pyridine": ["c1ccncc1", "c1cc[nH+]cc1"],
    "bipyridine": ["c1ccncc1-c1ccncc1", "c1cc[nH+]cc1-c1cc[nH+]cc1"],
    "pyridone": ["O=c1cccc[nH]1", "Oc1ccccn1"],
    "sulfonic_acid": ["O=S(=O)(O)c1ccccc1", "[O-]S(=O)(=O)c1ccccc1"],
}


class LinkerTests(unittest.TestCase):
    def assert_family(self, forms, expected=None):
        parents = [std.standardize_fragment_smiles(s) for s in forms]
        self.assertNotIn(None, parents, list(zip(forms, parents)))
        self.assertEqual(len(set(parents)), 1, list(zip(forms, parents)))
        if expected:
            self.assertEqual(parents[0], expected)
        for parent in parents:
            self.assertEqual(std.standardize_fragment_smiles(parent), parent)
            self.assertFalse(any(a.GetNumRadicalElectrons() for a in Chem.MolFromSmiles(parent).GetAtoms()))
        return parents[0]

    def test_four_observed_tetrazole_forms(self):
        parent = self.assert_family(TETRAZOLES, TETRAZOLE_PARENT)
        values = [sascorer.calculateScore(Chem.MolFromSmiles(std.standardize_fragment_smiles(s))) for s in TETRAZOLES]
        self.assertEqual(len(set(values)), 1)
        self.assertAlmostEqual(values[0], 2.4680365646222064)
        self.assertEqual(len({qm.identity_key(std.standardize_fragment_smiles(s)) for s in TETRAZOLES}), 1)

    def test_ligand_families(self):
        for name, forms in FAMILIES.items():
            with self.subTest(family=name):
                self.assert_family(forms)

    def test_atom_order_and_kekule_invariance(self):
        rng = random.Random(20261024)
        for raw in TETRAZOLES + [forms[-1] for forms in FAMILIES.values()]:
            expected = std.standardize_fragment_smiles(raw)
            mol = Chem.MolFromSmiles(raw, sanitize=False)
            Chem.SanitizeMol(mol, sanitizeOps=(Chem.SanitizeFlags.SANITIZE_CLEANUP |
                             Chem.SanitizeFlags.SANITIZE_PROPERTIES |
                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                             Chem.SanitizeFlags.SANITIZE_FINDRADICALS))
            for _ in range(20):
                order = list(range(mol.GetNumAtoms()))
                rng.shuffle(order)
                smi = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
                self.assertEqual(std.standardize_fragment_smiles(smi), expected, smi)
        for raw in ["Cc1nn[nH]n1", "Cc1ncc[nH]1"]:
            mol = Chem.MolFromSmiles(raw)
            self.assertEqual(std.standardize_fragment_smiles(Chem.MolToSmiles(mol, kekuleSmiles=True)),
                             std.standardize_fragment_smiles(raw))

    def test_charge_and_isotope_preservation(self):
        for raw in ["C[N+](C)(C)C", "[O-][n+]1ccccc1", "O=[N+]([O-])c1ccccc1", "Cn1ccnc1"]:
            self.assertEqual(std.standardize_fragment_smiles(raw), Chem.MolToSmiles(Chem.MolFromSmiles(raw)))
        self.assertIn("13C", std.standardize_fragment_smiles("[13CH3]c1nn[nH]n1"))
        self.assertIn("13C", std.standardize_fragment_smiles("[13CH3]c1nn[nH]n1", False))
        self.assertIn("[2H]", std.standardize_fragment_smiles("[2H]C([2H])([2H])c1ccccc1"))

    def test_stereochemistry_and_atom_maps(self):
        a, b = "N[C@H](C)Cc1ccccc1", "N[C@@H](C)Cc1ccccc1"
        self.assertNotEqual(std.standardize_fragment_smiles(a), std.standardize_fragment_smiles(b))
        self.assertEqual(std.standardize_fragment_smiles(a, False), std.standardize_fragment_smiles(b, False))
        self.assertEqual(std.standardize_fragment_smiles("[CH3:7][OH:3]"), std.standardize_fragment_smiles("CO"))
        self.assertEqual(std.standardize_fragment_smiles("[H]OC(=O)c1ccccc1"), std.standardize_fragment_smiles("O=C(O)c1ccccc1"))
        self.assertNotEqual(std.standardize_fragment_smiles("OC(=O)/C=C/C(=O)O"),
                            std.standardize_fragment_smiles("OC(=O)/C=C\\C(=O)O"))
        self.assertNotEqual(std.standardize_fragment_smiles("Cc1ncc[nH]1"),
                            std.standardize_fragment_smiles("Cc1ccn[nH]1"))

    def test_unsupported_inputs_and_no_partial_sets(self):
        for raw in [None, "", "invalid(", "C(C)(C)(C)(C)C", "[Zn]", "*c1ccccc1", "CC.O", "[C]1N=NN=N1"]:
            with self.subTest(raw=raw):
                self.assertIsNone(std.standardize_fragment_smiles(raw))
                self.assertTrue(std.standardize_fragment_record(raw)["reason"])
        self.assertEqual(std.standardize_linker_set("CC || invalid("), (None, None, 2, 1))
        a = std.standardize_linker_set("CC || C(C) || O")
        b = std.standardize_linker_set("O || CC")
        self.assertEqual(a[:2], b[:2])

    def test_incomplete_enumeration_is_rejected(self):
        factory = std.rdMolStandardize.TautomerEnumerator
        # Use a proxy to force a truncated result without mocking its outcome.
        class Limited:
            def __init__(self): self.e = factory(); self.e.SetMaxTautomers(1)
            def SetMaxTautomers(self, n): pass
            def SetMaxTransforms(self, n): self.e.SetMaxTransforms(n)
            def Enumerate(self, m): return self.e.Enumerate(m)
        with patch.object(std.rdMolStandardize, "TautomerEnumerator", Limited):
            r = std.standardize_fragment_record("CC(=O)CC(=O)C")
        self.assertEqual(r["status"], "failed")
        self.assertIn("incomplete_tautomer_enumeration", r["reason"])

    def test_raw_stats_parser_and_driver_audit(self):
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            stats = work / "linker_stats.txt"
            def row(s): return "key\t2\t1\tinchi\tbackbone\t" + s + "\tunused\n"
            stats.write_text(row("c1[n-][nH]cc1") + row("c1cn[nH]c1") + row("[Zn]"))
            ctx = {"rules": ROOT / "predict/rules"}
            result = driver.step_fragment_set(ctx, stats, work)
            self.assertTrue(result["complete"])
            self.assertEqual(result["n_standardised"], 1)
            self.assertEqual(len(result["excluded_fragments"]), 1)
            self.assertEqual(len(result["fragment_records"]), 2)
            self.assertTrue((work / "fragment_standardization.json").exists())
            stats.write_text(row("CC") + row("C(C)(C)(C)(C)C"))
            with self.assertRaises(SystemExit): driver.step_fragment_set(ctx, stats, work)
            self.assertFalse(json.loads((work / "fragment_standardization.json").read_text())["complete"])
            stats.write_text(row("CC") + "broken row\n")
            self.assertFalse(std.standardize_mofid_linker_stats(stats)["complete"])

    def test_complete_aggregation_only(self):
        self.assertFalse(driver.score_fragments({}, ["CC", "invalid("])["complete"])
        self.assertIsNone(driver.score_fragments({}, ["CC", "invalid("])["sascore"])
        for missing in [None, math.nan, math.inf]:
            result = driver.aggregate_channel_b({"a": -9., "b": missing}, ["a", "b"])
            self.assertFalse(result["complete"])
            self.assertIsNone(result["B"])

    def test_identical_xyz_for_equivalent_inputs(self):
        parents = [std.standardize_fragment_smiles(s) for s in TETRAZOLES]
        a, _ = qm.embed_canonical(parents[0])
        b, _ = qm.embed_canonical(parents[-1])
        self.assertEqual(a, b)

    def test_optimized_energy_uses_final_summary(self):
        with tempfile.TemporaryDirectory() as name:
            log = Path(name) / "check.out"
            log.write_text(":: total energy -53.007036042955 Eh ::\n"
                           "GEOMETRY OPTIMIZATION CONVERGED\n"
                           ":: total energy -53.047652245955 Eh ::\n")
            self.assertEqual(qm.parse_total_energy(log), -53.047652245955)


if __name__ == "__main__":
    unittest.main(verbosity=2)
