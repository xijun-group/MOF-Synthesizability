#!/usr/bin/env python3
"""Reusable standardisation of MOFid-extracted organic-fragment SMILES.

Why this exists
---------------
MOFid cuts a MOF into SBU nodes and organic linkers, then converts each linker
fragment to SMILES.  The fragment carries dangling valences wherever a
metal-carboxylate (or similar) bond was cut.  Bond perception resolves those
dangling valences inconsistently, so the SAME linker chemistry is emitted as
different SMILES depending on the source / converter:

    terephthalic acid, four representations actually observed in this project
      [O]C(=O)c1ccc(C([O])=O)cc1     under-valent radical O   (MOFTransformer, boyd_woo)
      [O-]C(=O)c1ccc(C(=O)[O-])cc1   carboxylate dianion      (qmof, CoRE)
      [O-]C(=O)c1ccc(C(=O)O)cc1      monoanion                (mixed)
      O=C(O)c1ccc(C(=O)O)cc1         neutral parent           (rare)

Those four strings are four different `canonical_linker_set` / `linker_set_id`
values, and they score differently under RDKit SA_Score (1.297 / 2.203 / 3.109 /
2.861 for exactly the same molecule).  Two consequences:

  1. linker identity is polluted: identical linkers do not group together, so
     `linker_count`, `linker_set_id` and every "same linker" search are biased;
  2. descriptor values (sascore, Linker_HOMO/LUMO) encode the extraction
     artefact rather than molecular chemistry, and the artefact is
     source-dependent, which is exactly the axis the paper must not confuse
     with a chemical signal.

What this module does
---------------------
`standardize_fragment_smiles` maps any of those representations onto one
canonical neutral parent, by

  1. capping every radical centre with an explicit hydrogen (the cut bond is
     replaced by H), then
  2. neutralising surplus charge with RDKit's Uncharger (anionic carboxylate /
     alkoxide / thiolate -> acid; cations deprotonated where possible; a
     genuinely quaternary N+ has no removable H and is left charged), then
  3. re-sanitising and writing the canonical SMILES.

`standardize_linker_set` applies the same rule to a whole `' || '`-joined
linker set and returns the re-sorted, de-duplicated canonical set.

Both are pure functions of the input string; nothing is imputed and no row is
dropped.  Unparsable fragments are reported, never silently repaired.
"""

from __future__ import annotations

import hashlib

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

SEPARATOR = " || "

_UNCHARGER = rdMolStandardize.Uncharger()

__all__ = [
    "SEPARATOR",
    "standardize_fragment_smiles",
    "standardize_linker_set",
    "linker_set_id",
    "fragment_provenance",
]


def _cap_radicals_with_h(mol: Chem.Mol) -> int:
    """Add one H per unpaired electron.  Returns the number of capped centres."""
    n_capped = 0
    for atom in mol.GetAtoms():
        n_rad = atom.GetNumRadicalElectrons()
        if n_rad:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + n_rad)
            atom.SetNumRadicalElectrons(0)
            n_capped += 1
    return n_capped


def standardize_fragment_smiles(smiles, isomeric=True):
    """Return the canonical neutral parent SMILES, or None if unparsable.

    With `isomeric=False` stereochemistry is dropped as well, which is the form
    to use when *comparing* whether two pipelines saw the same molecule: one
    pipeline routinely perceives E/Z and R/S where the other does not, and that
    is a perception difference, not a different molecule.

    The transformation is idempotent: standardising an already-standardised
    SMILES returns the same string.
    """
    if smiles is None:
        return None
    smiles = smiles.strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RWMol(mol)
    _cap_radicals_with_h(mol)
    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        mol = _UNCHARGER.uncharge(mol)
    except Exception:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if not isomeric:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def fragment_provenance(smiles):
    """Describe what was wrong with a fragment SMILES before standardisation.

    Returns a dict with the flags used by the audit, or None if unparsable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_radical_atoms = sum(1 for a in mol.GetAtoms() if a.GetNumRadicalElectrons())
    n_radical_electrons = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    charge = Chem.GetFormalCharge(mol)
    std = standardize_fragment_smiles(smiles)
    return {
        "n_radical_atoms": n_radical_atoms,
        "n_radical_electrons": n_radical_electrons,
        "formal_charge": charge,
        "has_radical": n_radical_atoms > 0,
        "has_charge": charge != 0,
        "needs_standardization": std is not None and std != Chem.MolToSmiles(mol),
        "standardized_smiles": std,
    }


def linker_set_id(canonical_linker_set: str) -> str:
    return hashlib.sha256(canonical_linker_set.encode()).hexdigest()[:16]


def standardize_linker_set(canonical_linker_set):
    """Standardise a ' || '-joined linker set.

    Returns (standardized_set, standardizer_id, n_fragments, n_unparsable).
    `standardizer_id` is None when no fragment could be standardised.
    """
    if not isinstance(canonical_linker_set, str) or not canonical_linker_set.strip():
        return None, None, 0, 0
    fragments = [f for f in canonical_linker_set.split(SEPARATOR) if f.strip()]
    out, n_unparsable = set(), 0
    for frag in fragments:
        std = standardize_fragment_smiles(frag)
        if std is None:
            n_unparsable += 1
        else:
            out.add(std)
    if not out:
        return None, None, len(fragments), n_unparsable
    joined = SEPARATOR.join(sorted(out))
    return joined, linker_set_id(joined), len(fragments), n_unparsable


if __name__ == "__main__":
    _CASES = [
        ("[O]C(=O)c1ccc(C([O])=O)cc1", "MOFTransformer radical form"),
        ("[O-]C(=O)c1ccc(cc1)C(=O)[O-]", "qmof dianion form"),
        ("[O-]C(=O)c1ccc(C(=O)O)cc1", "monoanion form"),
        ("O=C(O)c1ccc(C(=O)O)cc1", "neutral parent"),
    ]
    print("terephthalic acid, four observed representations ->")
    for smi, label in _CASES:
        std = standardize_fragment_smiles(smi)
        print(f"  {label:28s} {smi:32s} -> {std}")
    targets = {standardize_fragment_smiles(s) for s, _ in _CASES}
    assert len(targets) == 1, f"standardisation failed to unify: {targets}"
    assert standardize_fragment_smiles(next(iter(targets))) == next(iter(targets)), "not idempotent"
    print("OK: all four collapse to one canonical neutral parent, idempotent")
