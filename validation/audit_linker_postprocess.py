#!/usr/bin/env python3
"""Read-only sample audit of real raw SMILES, with legacy/new outputs."""
import argparse
import csv
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "predict/rules"))
import standardize_linker_smiles as std
from rdkit import Chem, rdBase


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", type=Path, required=True)
    ap.add_argument("--unparsable", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("legacy_std", ROOT / "baseline/rules/standardize_linker_smiles.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    all_rows = list(csv.DictReader(args.catalogue.open()))
    ordered = sorted(all_rows, key=lambda r: (-int(r["n_mof_rows"]), r["smiles"]))
    rng = random.Random(20261024)
    selection = {r["smiles"]: "top_256_frequency" for r in ordered[:256]}
    for r in rng.sample(sorted(all_rows, key=lambda r: r["smiles"]), 512):
        selection.setdefault(r["smiles"], "seeded_random_512")
    for token in ("[n-]", "[N]", "[C]", "[O]", "[S]", "[nH]", "[n+]", "[NH+]", "[P", "[B"):
        for r in [r for r in ordered if token in r["smiles"]][:100]:
            selection.setdefault(r["smiles"], "motif:" + token)
    for r in csv.DictReader(args.unparsable.open()):
        selection.setdefault(r["smiles"], "all_historical_unparsable")
    records, counts, reasons, repairs = [], Counter(), Counter(), Counter()
    violations = []
    for idx, (raw, scope) in enumerate(sorted(selection.items()), 1):
        r = std.standardize_fragment_record(raw)
        old = legacy.standardize_fragment_smiles(raw)
        parent = r["standardized_smiles"]
        counts["sampled"] += 1
        counts[r["status"]] += 1
        counts["changed"] += int(old != parent)
        counts["rescued"] += int(old is None and parent is not None)
        counts["previously_valued_now_rejected"] += int(old is not None and parent is None)
        if parent:
            idem = std.standardize_fragment_smiles(parent) == parent
            mol = Chem.MolFromSmiles(raw, sanitize=False)
            Chem.SanitizeMol(mol, sanitizeOps=(Chem.SanitizeFlags.SANITIZE_CLEANUP |
                             Chem.SanitizeFlags.SANITIZE_PROPERTIES |
                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                             Chem.SanitizeFlags.SANITIZE_FINDRADICALS))
            order = list(range(mol.GetNumAtoms()))
            rng.shuffle(order)
            reordered = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
            invariant = std.standardize_fragment_smiles(reordered) == parent
            counts["idempotent_checked"] += 1
            counts["atom_reorder_checked"] += 1
            if not idem or not invariant:
                violations.append({"raw": raw, "parent": parent, "idempotent": idem,
                                   "atom_reorder_invariant": invariant, "reordered": reordered})
        else:
            reasons[r["reason"].split(":")[0]] += 1
        repairs.update(r["repairs"])
        records.append({"raw_smiles": raw, "sample_scope": scope, "legacy_parent": old,
                        "new_parent": parent, "status": r["status"], "reason": r["reason"],
                        "repairs": " || ".join(r["repairs"]), "n_tautomers": r.get("n_tautomers")})
        if idx % 250 == 0:
            print(json.dumps({"processed": idx, "total": len(selection)}), flush=True)
    with (args.out / "real_fragment_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"state": "PASS" if not violations else "FAIL", "seed": 20261024,
               "scope": "stratified sample, not a full population reprocessing",
               "catalogue_rows": len(all_rows), "counts": dict(counts),
               "failure_reasons": dict(reasons), "repair_counts": dict(repairs),
               "violations": violations, "standardizer": std.standardizer_protocol(),
               "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in (args.catalogue, args.unparsable)]}
    (args.out / "real_fragment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return bool(violations)


if __name__ == "__main__":
    raise SystemExit(main())
