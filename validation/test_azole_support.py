from pathlib import Path
import sys,unittest,random,importlib.util
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
HERE=Path(__file__).resolve().parent
# Also runnable after deployment in the package's validation directory.
RULES=HERE.parent/'predict/rules'
sys.path.insert(0,str(RULES if RULES.is_dir() else HERE))
import standardize_linker_smiles as std
RAW='[N]1N=[C]C(=[C]1)c1ccc(cc1)C1=[C][N]N=[C]1'
PARENT='c1cc(-c2cn[nH]c2)ccc1-c1cn[nH]c1'
N_BAD='[O-]C(=O)c1ccc(cc1)C1=NC(=[N]=C([N]1)c1ccc(cc1)C(=O)[O-])c1ccc(cc1)C(=O)[O-]'
B_BAD='Nc1nnnn1[BH2]n1nnnc1N'

class AzoleSupport(unittest.TestCase):
 def test_observed_parent_and_formula(self):
  rec=std.standardize_fragment_record(RAW)
  self.assertEqual(rec['standardized_smiles'],PARENT)
  self.assertEqual(rec['repairs'].count('azole_sp2_C_radical_to_CH:2'),2)
  mol=Chem.MolFromSmiles(PARENT)
  self.assertEqual(rdMolDescriptors.CalcMolFormula(mol),'C12H10N4')
  self.assertEqual(Chem.GetFormalCharge(mol),0)
  self.assertEqual(sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),0)
  self.assertEqual(std.standardize_fragment_smiles(PARENT),PARENT)

 def test_degree_two_CH_restoration_family(self):
  for s in ['[N]1N=[C]C=C1','[N]1N=CC=[C]1','[N]1N=[C][C]=[C]1']:
   with self.subTest(s=s):
    self.assertEqual(std.standardize_fragment_smiles(s),std.standardize_fragment_smiles('c1cn[nH]c1'))

 def test_substituted_carbon_resonance_equivalence(self):
  # Alternative MOFid/Open Babel representation from the actual geometry.
  raw='N1=[C][C]([C]=N1)C1=[C][C]=C(C=C1)C1=[C][N]N=[C]1'
  self.assertEqual(std.standardize_fragment_smiles(raw),PARENT)
  self.assertEqual(std.standardize_fragment_smiles('C[C]1[C]=NN=[C]1'),
                   std.standardize_fragment_smiles('Cc1cn[nH]c1'))

 def test_atom_order_invariance(self):
  mol=Chem.MolFromSmiles(RAW,sanitize=False)
  Chem.SanitizeMol(mol,sanitizeOps=(Chem.SanitizeFlags.SANITIZE_CLEANUP|Chem.SanitizeFlags.SANITIZE_PROPERTIES|Chem.SanitizeFlags.SANITIZE_SYMMRINGS|Chem.SanitizeFlags.SANITIZE_FINDRADICALS))
  rng=random.Random(20261025)
  for i in range(100):
   order=list(range(mol.GetNumAtoms()));rng.shuffle(order)
   raw=Chem.MolToSmiles(Chem.RenumberAtoms(mol,order),canonical=False)
   self.assertEqual(std.standardize_fragment_smiles(raw),PARENT,raw)

 def test_connectivity_preserved(self):
  a=Chem.MolFromSmiles(RAW,sanitize=False);b=Chem.MolFromSmiles(PARENT)
  def graph(m):
   rw=Chem.RWMol(m)
   for bond in rw.GetBonds():bond.SetBondType(Chem.BondType.SINGLE);bond.SetIsAromatic(False)
   for atom in rw.GetAtoms():atom.SetIsAromatic(False);atom.SetNumExplicitHs(0);atom.SetNoImplicit(True);atom.SetNumRadicalElectrons(0);atom.SetFormalCharge(0)
   return Chem.MolToSmiles(rw)
  self.assertEqual(graph(a),graph(b))

 def test_reject_ambiguous_and_cumulene_patterns(self):
  for raw in ['[C]1N=NN=N1','[N]1[C]=C=NC1','[N]1N=[C]CC1','[N]1N=[C]C=[C-]1','[N]1N(C)=[C]C=[C]1','[N]1N=[C]C2=CC=CC=C12']:
   with self.subTest(raw=raw):self.assertIsNone(std.standardize_fragment_smiles(raw))

 def test_isotope_retained(self):
  raw=RAW.replace('N=[C]','N=[13C]',1)
  parent=std.standardize_fragment_smiles(raw)
  self.assertIsNotNone(parent)
  mol=Chem.MolFromSmiles(parent)
  self.assertEqual([(a.GetAtomicNum(),a.GetIsotope()) for a in mol.GetAtoms() if a.GetIsotope()],[(6,13)])
  self.assertEqual(std.standardize_fragment_smiles(parent),parent)

 def test_overvalent_inputs_not_blindly_repaired(self):
  for raw in (N_BAD,B_BAD):
   r=std.standardize_fragment_record(raw)
   self.assertEqual(r['status'],'failed')
   self.assertIn('Explicit valence',r['reason'])
   self.assertEqual(r['repairs'],[])

if __name__=='__main__':unittest.main(verbosity=2)
