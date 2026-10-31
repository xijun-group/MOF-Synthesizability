#!/usr/bin/env python3
"""Two fresh real xTB runs of equivalent MOFid inputs, no model scoring."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from test_linker_postprocess import TETRAZOLES, TETRAZOLE_PARENT, driver, std, qm


def main():
    out = ROOT / "validation/linkerfix/qm"
    out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_linker_qm.py /absolute/path/to/xtb")
    qm.XTB = sys.argv[1]
    parents = [std.standardize_fragment_smiles(s) for s in TETRAZOLES]
    assert set(parents) == {TETRAZOLE_PARENT}
    values = []
    for label, parent in (("raw_A", parents[0]), ("raw_D", parents[3])):
        work = out / label
        if work.exists():
            raise RuntimeError("verification requires a fresh directory: " + str(work))
        work.mkdir()
        record = qm.run_one_timed(parent, work, 300)
        homo = record.get("homo_eV") if record["status"] == "converged" else None
        values.append(driver.aggregate_channel_b({parent: homo}, [parent]))
    recs = [next((out / name / "linkers").glob("*/result.json"))
            for name in ("raw_A", "raw_D")]
    checks = {"same_normalized_parent": len(set(parents)) == 1,
              "both_complete": all(v["complete"] for v in values),
              "same_homo": values[0]["B"] is not None and values[0]["B"] == values[1]["B"]}
    for relative in ("linker.xyz", "opt/xtbopt.xyz", "sp/linker.xyz"):
        checks["same_" + relative] = (recs[0].parent / relative).read_bytes() == (recs[1].parent / relative).read_bytes()
    result = {"state": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "homo_eV": [v["B"] for v in values], "runs": [str(p) for p in recs],
              "xtb": sys.argv[1], "xtb_sha256": hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
              "standardizer": std.standardizer_protocol()}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
