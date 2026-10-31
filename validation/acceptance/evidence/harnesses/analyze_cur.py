"""Analyse bench_cur.sh output: timings and equivalence of the current default to the original.

usage: analyze_cur.py <bench dir> <matrix analysis json> <out json>

CPU four features and their intermediates must be bitwise equal to the original run on the same host.
Ehull and the native score are compared with the original's own spread for that MOF, measured by the
original relaxations of the GPU matrix (4090/5090/A800 and the v2 XJ2 panel) widened by this host's
own original result.
"""
import hashlib
import json
import sys
from pathlib import Path

MOFS = ["reference58", "gen_49320289d14b3171", "gen_eed3ec90c711abe1",
        "gen_2aed42338cd9ddfb", "gen_5b466e79428854e0"]
CPU4 = ["solvent_removal_stability", "RMSD_[A]", "Linker_HOMO_[eV]", "sascore"]
RMSD_KEYS = ("rmsd_a", "rmsd_method", "all_method_values", "natoms", "smiles", "sp0_energy_hartree",
             "opt_energy_hartree", "sp1_energy_hartree", "linkers_cif_sha256")


def desc(work):
    return [hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((work / "mofsimplify/desc").glob("*_descriptors.csv"))]


def main():
    bench, matrix, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    runs = json.loads(matrix.read_text())["runs"]
    rows, totals = [], {"orig": 0.0, "curr": 0.0}
    for m in MOFS:
        t = {mode: json.loads((bench / m / mode / "trial.json").read_text()) for mode in ("orig", "curr")}
        p = {mode: json.loads((bench / m / mode / "work/prediction.json").read_text()) for mode in t}
        orig_samples = [r for r in runs if r["mof"] == m and r["variant"] == "orig/default"]
        eh = [r["ehull"] for r in orig_samples] + [p["orig"]["features"]["Ehull"]]
        window_ehull = max(eh) - min(eh)
        row = {"label": m, "rc": {k: v["rc"] for k, v in t.items()},
               "wall_s": {k: round(v["wall_s"], 2) for k, v in t.items()}}
        for mode in ("curr",):
            a, b = p["orig"], p[mode]
            da, db = a["details"], b["details"]
            ehull_dev = abs(a["features"]["Ehull"] - b["features"]["Ehull"])
            row[mode] = {
                "cpu4_exact": all(a["features"][k] == b["features"][k] for k in CPU4),
                "rmsd_details_exact": all(da["rmsd"].get(k) == db["rmsd"].get(k) for k in RMSD_KEYS),
                "fragments_homo_sascore_exact": (da["fragment_set"] == db["fragment_set"]
                                                 and da["channel_b"]["per_fragment"] == db["channel_b"]["per_fragment"]
                                                 and da["sascore"] == db["sascore"]),
                "descriptor_bytes_exact": desc(bench / m / "orig/work") == desc(bench / m / mode / "work"),
                "relaxation_converged": db["ehull"]["relaxation_converged"] is True,
                "ehull_abs_dev": ehull_dev,
                "ehull_within_original_window": ehull_dev <= window_ehull,
                "score_abs_dev": abs(a["native_score"] - b["native_score"]),
                "speedup_vs_original": t["orig"]["wall_s"] / t[mode]["wall_s"],
            }
        row["original_window_ehull"] = window_ehull
        row["original_window_samples"] = len(eh)
        rows.append(row)
        for k in totals:
            totals[k] += t[k]["wall_s"]
    gates = ("cpu4_exact", "rmsd_details_exact", "fragments_homo_sascore_exact", "descriptor_bytes_exact",
             "relaxation_converged", "ehull_within_original_window")
    summary = {"pass": sum(all(r["curr"][g] for g in gates) for r in rows),
               "aggregate_speedup_vs_original": totals["orig"] / totals["curr"],
               "median_speedup_vs_original": sorted(r["curr"]["speedup_vs_original"] for r in rows)[2]}
    doc = {"bench": str(bench), "rows": rows, "totals_s": totals, "summary": summary, "gates": gates}
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{'MOF':22s} {'original':>9s} {'current':>8s}  x     gates      dEhull     window    dScore")
    for r in rows:
        g = "ok" if all(r["curr"][x] for x in gates) else "FAIL:" + ",".join(x for x in gates if not r["curr"][x])
        w = r["wall_s"]
        print(f"{r['label']:22s} {w['orig']:9.1f} {w['curr']:8.1f} {r['curr']['speedup_vs_original']:5.2f}  {g:10s} "
              f"{r['curr']['ehull_abs_dev']:.2e}  {r['original_window_ehull']:.2e}  {r['curr']['score_abs_dev']:.2e}")
    print(f"total {totals['orig']:.1f} / {totals['curr']:.1f} s; aggregate x{summary['aggregate_speedup_vs_original']:.2f}, "
          f"median x{summary['median_speedup_vs_original']:.2f}; pass {summary['pass']}/5")


if __name__ == "__main__":
    main()
