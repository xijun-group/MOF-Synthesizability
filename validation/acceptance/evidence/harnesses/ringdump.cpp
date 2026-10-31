// Dump SSSR and LSSR ring paths (in order) for a CIF read the way MOFid's importCIF reads it.
#include <openbabel/obconversion.h>
#include <openbabel/mol.h>
#include <openbabel/ring.h>
#include <openbabel/generic.h>
#include <openbabel/obiter.h>
#include <openbabel/atom.h>
#include <iostream>
using namespace OpenBabel;
int main(int argc, char **argv) {
  OBConversion conv; conv.SetInFormat("mmcif");
  conv.AddOption("p", OBConversion::INOPTIONS);
  OBMol mol;
  if (!conv.ReadFile(&mol, argv[1])) return 2;
  OBUnitCell *uc = (OBUnitCell*)mol.GetData(OBGenericDataType::UnitCell);
  if (uc) uc->FillUnitCell(&mol);
  mol.ConnectTheDots();
  std::vector<OBRing*> s = mol.GetSSSR();
  std::cout << "atoms " << mol.NumAtoms() << " bonds " << mol.NumBonds() << " sssr " << s.size() << "\n";
  for (size_t i = 0; i < s.size(); ++i) { for (int a : s[i]->_path) std::cout << a << ' '; std::cout << '\n'; }
  std::vector<OBRing*> l = mol.GetLSSR();
  std::cout << "lssr " << l.size() << "\n";
  for (size_t i = 0; i < l.size(); ++i) { for (int a : l[i]->_path) std::cout << a << ' '; std::cout << '\n'; }
  int arom = 0; FOR_ATOMS_OF_MOL(a, mol) arom += a->IsAromatic();
  std::cout << "aromatic_atoms " << arom << "\n";
}
