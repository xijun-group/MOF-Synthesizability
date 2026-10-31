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
The v3 rule repairs recognised carboxyl multiradicals and azole ring radicals
BEFORE H capping, then selects a canonical tautomer AFTER neutralisation.
This defines a descriptor parent, not the energetically preferred tautomer.
The per-fragment audit and standardizer_protocol() record the convention.
In an isolated conjugated C/N five-ring with one delocalised ring radical,
degree-two sp2 carbon monoradicals may additionally represent missing C-H
caps. The ring radical must reside on N or a substituted degree-three C.
Only that restricted pattern is supported; arbitrary multiradicals fail.

`standardize_fragment_smiles` maps supported representations onto one parent, by

  1. repairing recognised bond-perception artefacts, then capping remaining
     radical centres with an explicit hydrogen (the cut bond is
     replaced by H), then
  2. neutralising surplus charge with RDKit's Uncharger (anionic carboxylate /
     alkoxide / thiolate -> acid; cations deprotonated where possible; a
     genuinely quaternary N+ has no removable H and is left charged), then
  3. re-sanitising and selecting the canonical tautomer after complete
     enumeration. Permanent charges and isotopes are preserved.

`standardize_linker_set` applies the same rule to a whole `' || '`-joined
linker set and returns the re-sorted, de-duplicated canonical set.

Results depend on the recorded RDKit version. No connectivity is inferred.
Disconnected, dummy and unsupported inputs fail explicitly. A set containing
a failed fragment has no defined aggregate; it is never silently truncated.
"""

from __future__ import annotations

import hashlib

from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

SEPARATOR = " || "
STANDARDIZER_VERSION = "mofid-neutral-parent-v3-azole-carbon-H-20261025"
ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"}

_UNCHARGER = rdMolStandardize.Uncharger(canonicalOrder=True)


def standardizer_protocol():
    return {"version": STANDARDIZER_VERSION, "rdkit_version": rdBase.rdkitVersion,
            "tautomer_max_tautomers": 1000, "tautomer_max_transforms": 1000,
            "tautomer_selection": "RDKit default score, canonical SMILES tie-break",
            "azole_carbon_H_rule": "isolated neutral C/N five-ring; two double bonds; "
                                   "one N or substituted-C ring radical plus "
                                   "degree-two valence-three carbon monoradicals only",
            "remaining_radicals": "legacy one-H-per-radical-electron capping",
            "stereochemistry": "retain unless changed by RDKit tautomer enumeration"}


def _repair_carboxyl_multiradicals(mol, repairs):
    """Restore [O][C]([O])R before capping can turn it into a geminal diol."""
    for carbon in mol.GetAtoms():
        if (carbon.GetAtomicNum() != 6 or carbon.GetDegree() != 3 or
                carbon.GetFormalCharge() or carbon.GetNumRadicalElectrons() != 1 or
                carbon.GetTotalNumHs() or
                any(b.GetBondType() != Chem.BondType.SINGLE for b in carbon.GetBonds())):
            continue
        oxygens = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8 and
                   a.GetDegree() == 1 and a.GetFormalCharge() in (0, -1)]
        radicals = [a for a in oxygens if a.GetNumRadicalElectrons() == 1 and
                    a.GetFormalCharge() == 0 and not a.GetTotalNumHs()]
        if len(oxygens) != 2 or not radicals:
            continue
        ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
        oxygen = min(radicals, key=lambda a: (ranks[a.GetIdx()], a.GetIdx()))
        mol.GetBondBetweenAtoms(carbon.GetIdx(), oxygen.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
        carbon.SetNumRadicalElectrons(0)
        oxygen.SetNumRadicalElectrons(0)
        for other in oxygens:
            if other.GetIdx() != oxygen.GetIdx() and other.GetNumRadicalElectrons() == 1:
                other.SetNumRadicalElectrons(0)
                other.SetFormalCharge(-1)
        repairs.append("carboxyl_multiradical_to_carboxylate")
    mol.UpdatePropertyCache(strict=False)


def _repair_azoles(mol, repairs):
    rings = list(mol.GetRingInfo().AtomRings())
    charge_sites = []
    for ring in rings:
        if len(ring) != 5:
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if any(a.GetAtomicNum() not in (6, 7) for a in atoms):
            continue
        nitrogens = [a for a in atoms if a.GetAtomicNum() == 7]
        if not nitrogens:
            continue
        # Only remove surplus [n-] when a pyrrolic donor already exists.
        # A blanket string replacement would damage valid azolate anions.
        anions = [a for a in nitrogens if a.GetFormalCharge() == -1]
        donors = [a for a in nitrogens if a.GetFormalCharge() == 0 and
                  (a.GetTotalNumHs() == 1 or a.GetDegree() == 3)]
        if (all(a.GetIsAromatic() for a in atoms) and len(anions) == 1 and
                len(donors) == 1 and anions[0].GetDegree() == 2 and
                not anions[0].GetTotalNumHs()):
            anions[0].SetFormalCharge(0)
            repairs.append("azole_surplus_pyridine_N_charge")
        # Conjugated, isolated C/N five-ring with no existing N-H or
        # N-substitution. Do not repair fused/ambiguous radical rings.
        if any(set(ring) & set(other) for other in rings if other != ring):
            continue
        bonds = [mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % 5]) for i in range(5)]
        if (any(b is None for b in bonds) or
                sum(b.GetBondType() == Chem.BondType.DOUBLE for b in bonds) != 2 or
                any(b.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE) for b in bonds) or
                any(a.GetFormalCharge() for a in atoms) or
                any(a.GetDegree() != 2 or a.GetTotalNumHs() for a in nitrogens)):
            continue
        ring_set = set(ring)
        if any(b.GetBondType() != Chem.BondType.SINGLE for a in atoms for b in a.GetBonds()
               if b.GetOtherAtomIdx(a.GetIdx()) not in ring_set):
            continue
        # AFOYOK-type output has missing C-H atoms written as [C], plus one
        # remaining ring radical. The latter may reside on N or on the
        # substituted carbon in another resonance representation already
        # supported by the single-radical azolate rule.
        carbon_H_sites = [a for a in atoms if a.GetAtomicNum() == 6 and
                          a.GetNumRadicalElectrons() == 1 and a.GetDegree() == 2 and
                          a.GetTotalNumHs() == 0 and
                          a.GetValence(Chem.ValenceType.EXPLICIT) == 3]
        cap_indices = {a.GetIdx() for a in carbon_H_sites}
        remaining = [a for a in atoms if a.GetNumRadicalElectrons() and
                     a.GetIdx() not in cap_indices]
        if carbon_H_sites and sum(a.GetNumRadicalElectrons() for a in remaining) == 1:
            donor = remaining[0]
            if not (donor.GetAtomicNum() == 7 or
                    (donor.GetAtomicNum() == 6 and donor.GetDegree() == 3 and
                     donor.GetTotalNumHs() == 0 and
                     donor.GetValence(Chem.ValenceType.EXPLICIT) == 3)):
                continue
            # Apart from a substituted-C donor, each carbon already needs
            # one ring pi bond. This excludes cumulenes/saturated sites; adding H
            # must not silently turn those into a different aromatic system.
            if any(sum(b.GetBondType() == Chem.BondType.DOUBLE
                       for b in a.GetBonds()
                       if b.GetOtherAtomIdx(a.GetIdx()) in ring_set) !=
                   (0 if a.GetIdx() == donor.GetIdx() else 1)
                   for a in atoms if a.GetAtomicNum() == 6):
                continue
            for atom in carbon_H_sites:
                atom.SetNumExplicitHs(1)
                atom.SetNumRadicalElectrons(0)
            mol.UpdatePropertyCache(strict=False)
            repairs.append(f"azole_sp2_C_radical_to_CH:{len(carbon_H_sites)}")
        if sum(a.GetNumRadicalElectrons() for a in atoms) != 1:
            continue
        # The remaining radical is treated as delocalised deprotonation.
        for atom in atoms:
            atom.SetIsAromatic(True)
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(atom.GetAtomicNum() == 7)
        for bond in bonds:
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)
        charge_sites.append([a.GetIdx() for a in nitrogens])
        repairs.append("azole_ring_radical_to_delocalised_azolate")
    mol.UpdatePropertyCache(strict=False)
    if charge_sites:
        # Rank after all ring radical sites have been removed.
        ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
        for candidates in charge_sites:
            mol.GetAtomWithIdx(min(candidates, key=lambda i: (ranks[i], i))).SetFormalCharge(-1)
    mol.UpdatePropertyCache(strict=False)


def standardize_fragment_record(smiles, isomeric=True):
    """Return the parent and explicit audit; never hide sanitisation failures.

    This is a versioned descriptor convention, not an energy-ranked tautomer.
    Atom connectivity is never inferred. Incomplete tautomer enumeration is
    a failure, because selecting its partial set can depend on atom order.
    """
    record = {"raw_smiles": smiles, "standardized_smiles": None, "status": "failed",
              "reason": "", "repairs": [], "standardizer_version": STANDARDIZER_VERSION}
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("empty_or_nonstring_smiles")
        params = Chem.SmilesParserParams()
        params.sanitize = False
        params.removeHs = False
        params.parseName = False
        mol = Chem.MolFromSmiles(smiles.strip(), params)
        if mol is None or not mol.GetNumAtoms():
            raise ValueError("invalid_smiles_syntax")
        if any(a.GetSymbol() not in ALLOWED_ELEMENTS for a in mol.GetAtoms()):
            raise ValueError("unsupported_element_or_attachment_dummy")
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("disconnected_fragment_requires_explicit_membership")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        mol = Chem.RemoveHs(mol, sanitize=False)
        # Assign radicals and rings WITHOUT kekulising malformed azolates.
        Chem.SanitizeMol(mol, sanitizeOps=(Chem.SanitizeFlags.SANITIZE_CLEANUP |
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES |
                         Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                         Chem.SanitizeFlags.SANITIZE_FINDRADICALS))
        record.update(n_radical_atoms=sum(bool(a.GetNumRadicalElectrons()) for a in mol.GetAtoms()),
                      n_radical_electrons=sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
                      formal_charge=Chem.GetFormalCharge(mol))
        record["canonical_raw_smiles"] = Chem.MolToSmiles(mol)
        _repair_carboxyl_multiradicals(mol, record["repairs"])
        _repair_azoles(mol, record["repairs"])
        for ring in mol.GetRingInfo().AtomRings():
            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            if (len(ring) == 5 and any(a.GetAtomicNum() == 7 for a in atoms) and
                    all(a.GetAtomicNum() in (6, 7) for a in atoms) and
                    any(a.GetNumRadicalElectrons() for a in atoms)):
                raise ValueError("unsupported_azole_radical_pattern")
        n_capped = _cap_radicals_with_h(mol)
        if n_capped:
            record["repairs"].append(f"H_cap_remaining_radical_centres:{n_capped}")
        Chem.SanitizeMol(mol)
        mol = _UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
        if not isomeric:
            Chem.RemoveStereochemistry(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(1000)
        enumerator.SetMaxTransforms(1000)
        tautomers = enumerator.Enumerate(mol)
        record["tautomer_status"] = str(tautomers.status)
        record["n_tautomers"] = len(tautomers)
        if tautomers.status != rdMolStandardize.TautomerEnumeratorStatus.Completed:
            raise ValueError("incomplete_tautomer_enumeration:" + str(tautomers.status))
        parent = enumerator.PickCanonical(tautomers)
        Chem.SanitizeMol(parent)
        if any(a.GetNumRadicalElectrons() for a in parent.GetAtoms()):
            raise ValueError("residual_radical_after_standardization")
        if not isomeric:
            Chem.RemoveStereochemistry(parent)
        record.update(status="ok", standardized_smiles=Chem.MolToSmiles(parent),
                      standardized_formal_charge=Chem.GetFormalCharge(parent))
    except Exception as exc:
        record["reason"] = str(exc)
    return record

__all__ = [
    "SEPARATOR",
    "standardize_fragment_smiles",
    "standardize_linker_set",
    "linker_set_id",
    "fragment_provenance",
    "standardize_fragment_record",
    "standardize_mofid_linker_stats",
    "standardizer_protocol",
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
    return standardize_fragment_record(smiles, isomeric)["standardized_smiles"]


def fragment_provenance(smiles):
    """Describe what was wrong with a fragment SMILES before standardisation.

    Returns a dict with the flags used by the audit, or None if unparsable.
    """
    record = standardize_fragment_record(smiles)
    if "n_radical_atoms" not in record:
        return None
    record.update(has_radical=bool(record["n_radical_atoms"]), has_charge=bool(record["formal_charge"]),
                  needs_standardization=(record["standardized_smiles"] is not None and
                                         record["standardized_smiles"] != record.get("canonical_raw_smiles")))
    return record


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
    if n_unparsable or not out:
        return None, None, len(fragments), n_unparsable
    joined = SEPARATOR.join(sorted(out))
    return joined, linker_set_id(joined), len(fragments), n_unparsable


def standardize_mofid_linker_stats(path):
    """Read raw column 6 before sanitisation; audit exclusions and failures."""
    records, excluded = [], []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) < 6 or not columns[5].strip():
            records.append({"line": line_no, "raw_smiles": None, "status": "failed",
                            "standardized_smiles": None, "reason": "malformed_linker_stats_row"})
            continue
        raw = columns[5].strip()
        mol = Chem.MolFromSmiles(raw, sanitize=False)
        if mol is not None and any(a.GetSymbol() not in ALLOWED_ELEMENTS for a in mol.GetAtoms()):
            excluded.append({"line": line_no, "raw_smiles": raw,
                             "reason": "outside_existing_organic_element_allowlist"})
            continue
        record = standardize_fragment_record(raw)
        record["line"] = line_no
        records.append(record)
    failed = [r for r in records if r["status"] != "ok"]
    parents = sorted({r["standardized_smiles"] for r in records if r["status"] == "ok"})
    return {"canonical_set": sorted({r["canonical_raw_smiles"] for r in records if r.get("canonical_raw_smiles")}),
            "raw_set": sorted({r["raw_smiles"] for r in records if r["raw_smiles"]}),
            "standardised_set": parents, "n_fragments": len(records),
            "n_standardised": len(parents), "complete": bool(records) and not failed,
            "unstandardisable": failed, "excluded_fragments": excluded,
            "fragment_records": records, "standardizer": standardizer_protocol(),
            "linker_parse_status": "ok" if records and not failed else "incomplete_organic_set"}


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
