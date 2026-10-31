"""Select a complete SA-max parent and its existing framework geometry.

Prefer an exact SA-max parent match, then stereo-compatible correspondence
for a wholly stereo-unspecified target, then the original largest component. No embedding, atom addition,
or replacement of geometry by neutral descriptor parent coordinates.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

PROTOCOL = "rmsd-deltae-sa-stereo-compatible-v4-20261025"


def _standardizer():
    spec = importlib.util.spec_from_file_location(
        "rmsd_target_standardizer", Path(__file__).with_name("standardize_linker_smiles.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_parent(sa_scores):
    """Exact SA ties use lexicographically smallest canonical parent SMILES."""
    if not sa_scores or not sa_scores.get("complete"):
        raise ValueError("RMSD requires a complete finite SA fragment set")
    values = sa_scores.get("per_fragment", {})
    if not values or any(v is None or not math.isfinite(v) for v in values.values()):
        raise ValueError("RMSD requires a complete finite SA fragment set")
    maximum = max(values.values())
    if sa_scores.get("sascore") != maximum:
        raise ValueError("SA aggregate disagrees with per-fragment maximum")
    tied = sorted(s for s, value in values.items() if value == maximum)
    return {"protocol": PROTOCOL, "target_parent": tied[0], "target_sa": maximum,
            "sa_tied_parents": tied, "sa_tie_rule": "exact maximum; lexicographic canonical SMILES",
            "geometry_copy_rule": "exact matches first, then compatible matches; most explicit atoms within tier, first component on tie",
            "stereo_compatibility_rule": "only wholly stereo-unspecified targets; equality after removing stereo only; preserve isotopes, charge, H and bond graph",
            "geometry_fallback_rule": "if neither exact nor compatible match, original Open Babel StripSalts (-r) largest component, including native tie handling",
            "standardizer": _standardizer().standardizer_protocol()}


def matching_components(records, parent):
    """Return exact and compatible tiers without altering either stored identity.

    Compatibility is deliberately limited to wholly stereo-unspecified targets.
    Any specified target stereo (including partially specified molecules) disables
    the second tier. Do not equate conflicting specified stereoisomers.
    """
    from rdkit import Chem
    exact = [r for r in records if r.get("status") == "ok" and
             r.get("standardized_parent") == parent]
    target = Chem.MolFromSmiles(parent) if isinstance(parent, str) else None
    if target is None:
        return exact, []
    if (any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in target.GetAtoms()) or
            any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in target.GetBonds())):
        return exact, []
    Chem.RemoveStereochemistry(target)
    key = Chem.MolToSmiles(target, isomericSmiles=True)
    cache, compatible = {}, []
    for record in records:
        raw = record.get("standardized_parent")
        if record.get("status") != "ok" or not raw or raw == parent:
            continue
        if raw not in cache:
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                cache[raw] = None
            else:
                Chem.RemoveStereochemistry(mol)
                # Keep isotope information; no neutralization or tautomer edits here.
                cache[raw] = Chem.MolToSmiles(mol, isomericSmiles=True)
        if cache[raw] == key:
            compatible.append(record)
    return exact, compatible


def choose_component(records, parent, largest_component_index=None):
    exact, compatible = matching_components(records, parent)
    matches = exact or compatible
    if not records:
        raise ValueError("No connected framework geometry is available")
    if matches:
        return min(matches, key=lambda r: (-r["natoms"], r["component_index"]))
    # Use the native -r selection, including its tie behavior, not a Python tie rule.
    maximum = max(r["natoms"] for r in records)
    for record in records:
        if record["component_index"] == largest_component_index and record["natoms"] == maximum:
            return record
    raise ValueError("Native largest-component selection missing or inconsistent")


def extract_geometry(linkers_cif, xyz_path, selection):
    """Use the same Open Babel CIF reader as the old obabel -r geometry branch.

    Every connected component is identified by the shared standardizer. Identity
    normalization is for matching only; the emitted XYZ retains exactly the
    selected component's explicit atoms and coordinates. Copies with missing
    explicit hydrogens can have the same descriptor parent, so prefer the most
    atom-complete available copy without adding hydrogens or changing its state.
    """
    from openbabel import openbabel as ob
    std = _standardizer()
    linkers_cif, xyz_path = Path(linkers_cif), Path(xyz_path)
    audit_path = xyz_path.parent / "geometry_selection.json"
    conv = ob.OBConversion()
    conv.SetInFormat("cif")
    mol = ob.OBMol()
    if not conv.ReadFile(mol, str(linkers_cif)) or not mol.NumAtoms():
        raise ValueError("Open Babel could not read linker geometry CIF")
    can = ob.OBConversion()
    can.SetOutFormat("can")
    xyz = ob.OBConversion()
    xyz.SetOutFormat("xyz")
    components = list(mol.Separate())
    records, cache = [], {}
    for index, component in enumerate(components):
        tokens = can.WriteString(component).split()
        raw = tokens[0] if tokens else ""
        if raw not in cache:
            cache[raw] = std.standardize_fragment_record(raw)
        result = cache[raw]
        records.append({"component_index": index, "natoms": component.NumAtoms(),
                        "raw_geometry_smiles": raw, "raw_geometry_formula": component.GetFormula(),
                        "raw_geometry_formal_charge": component.GetTotalCharge(),
                        "standardized_parent": result["standardized_smiles"],
                        "status": result["status"], "reason": result["reason"],
                        "repairs_for_identity_only": result["repairs"]})
    audit = {**selection, "openbabel_version": ob.OBReleaseVersion(),
             "linkers_cif": str(linkers_cif),
             "linkers_cif_sha256": hashlib.sha256(linkers_cif.read_bytes()).hexdigest(),
             "components": records, "n_components": len(records),
             "coordinates": "original parsed framework geometry; no embedding or H addition",
             "identity_normalization_changes_geometry": False}
    try:
        largest_index = None
        exact, compatible = matching_components(records, selection["target_parent"])
        if not exact and not compatible:
            # This is the same operation used by obabel -r in the upstream workflow.
            largest = ob.OBMol(mol)
            largest.StripSalts()
            largest_rows = xyz.WriteString(largest).splitlines()[2:]
            largest_index = next((i for i, part in enumerate(components)
                                  if xyz.WriteString(part).splitlines()[2:] == largest_rows), None)
        chosen = choose_component(records, selection["target_parent"], largest_index)
    except ValueError as exc:
        audit.update(status="failed", reason=str(exc))
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        raise
    matched = chosen["status"] == "ok" and chosen["standardized_parent"] == selection["target_parent"]
    compatible_match = any(r["component_index"] == chosen["component_index"] for r in compatible)
    corresponds = matched or compatible_match
    mode = ("exact_sa_parent" if matched else "stereo_compatible_sa_parent"
            if compatible_match else "largest_fragment_fallback")
    component = components[chosen["component_index"]]
    block = xyz.WriteString(component)
    if not block.strip() or int(block.splitlines()[0]) != chosen["natoms"]:
        raise ValueError("invalid selected geometry XYZ")
    xyz_path.write_text(block)
    audit.update(status="ok", chosen=chosen,
                 selection_mode=mode,
                 fallback_used=not corresponds,
                 fallback_reason=None if corresponds else "no exact or stereo-compatible SA-max parent match",
                 selected_corresponds_sa_parent=corresponds,
                 stereo_compatibility_used=compatible_match,
                 n_stereo_compatible_components=len(compatible),
                 selected_matches_sa_parent=matched,
                 selected_parent=chosen["standardized_parent"],
                 n_matching_components=sum(r["standardized_parent"] == selection["target_parent"]
                                           and r["status"] == "ok" for r in records),
                 selected_xyz_sha256=hashlib.sha256(xyz_path.read_bytes()).hexdigest())
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit
