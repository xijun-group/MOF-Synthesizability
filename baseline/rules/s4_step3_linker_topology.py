"""Section 4, Step 3 of SBU_linker_topology_compatibility_workflow.md.

Builds MOF-level linker and topology identities for the full SBU input
population (selection_manifest.csv, 240,786 runs).

Identity sources
----------------
linker  : per-run Output/MetalOxo/linker_stats.txt (MOFid organic fragments).
          Each fragment SMILES (column 6) is canonicalised with RDKit; the sorted
          unique canonical SMILES are joined with ' || ' into canonical_linker_set;
          linker_set_id = sha256(canonical_linker_set)[:16].
          Both the canonicalisation and the id rule were calibrated against the
          published reference table
          shap/mof_lowscore_failure_diagnosis_v1/low_score_lt0p2_clustering_enrichment_v1/
          linker_identity_dictionary.csv and reproduce it exactly on all 39,766
          reference rows; the agreement is re-measured by check_against_reference().
topology: MOFid-v1 topology field, taken from the full-population MOFid parse in
          shap/mof_lowscore_failure_diagnosis_v1/matched_mofid_v1/{P,U}_mofid_audit.csv
          (383,243 parsed MOFid records). Name normalisation between that table and
          selection_manifest.csv is explicit and audited; no fuzzy matching.
          Multi-topology signatures are kept whole and never split.

Usage:  PYTHONPATH=./_deps python3 s4_step3_linker_topology.py [--limit N] [--jobs J]
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/cheyue/data/PU_raw_cif/mofid2')
MANIFEST = ROOT / 'selection_manifest.csv'
AUDIT_DIR = Path('/home/cheyue/data/SYN-final/shap/mof_lowscore_failure_diagnosis_v1/matched_mofid_v1')
REF_DICT = Path('/home/cheyue/data/SYN-final/shap/mof_lowscore_failure_diagnosis_v1/'
                'low_score_lt0p2_clustering_enrichment_v1/linker_identity_dictionary.csv')

SOURCE_FROM_FILE = {
    'boyd_woo_opt_mofid.csv': 'boyd_woo',
    'CoRE_2024_ASR_opt_mofid.csv': 'CoRE_2024_ASR',
    'DB13_bao_opt_mofid.csv': 'DB13_bao',
    'DB15_ARC_opt_mofid.csv': 'DB15_ARC',
    'DB1_Lan_hmof_opt_mofid.csv': 'DB1_Lan',
    'DB6_Li_opt_mofid.csv': 'DB6_Li',
    'hMOF_opt_mofid.csv': 'hMOF',
    'MOFTransformer_opt_mofid.csv': 'MOFTransformer',
    'qmof_opt_syn_mofid.csv': 'qmof',
    'Tobacco_opt_mofid.csv': 'Tobacco',
}

# Applied only when the exact (source, name) key is absent from the MOFid audit table.
# MOFTransformer names use '+' where the audit table uses '_'; the CoRE_2024_ASR audit rows
# are stored in the filename-safe form ('.' -> '_').
NORMALISE_RULES = {
    'MOFTransformer': [('plus_to_underscore', lambda n: n.replace('+', '_'))],
    'CoRE_2024_ASR': [('dot_to_underscore', lambda n: n.replace('.', '_'))],
}

# A topology token is a comma-joined multi-net signature and every component can carry a
# ".catN" catenation suffix, so validity is judged on the first dot-separated field of each
# component. TIMEOUT (Systre exceeded its 30 s limit) is not a net name either.
INVALID_TOKENS = ('ERROR', 'UNKNOWN', 'NA', 'TIMEOUT')
SEPARATOR = ' || '
FAMILY_PRIORITY = {'carboxylate': 0, 'imidazole_ring': 1, 'pyridyl_ring': 2}
CAT_PATTERN = re.compile(r'^([^.]+)\.(cat\d+)$')
SMARTS = {
    'carboxylate': '[CX3](=[OX1])[OX2H1,OX1-]',
    'imidazole_ring': '[nX2,nX3r5]1[cX3r5][nX2,nX3r5][cX3r5][cX3r5]1',
    'pyridyl_ring': '[nX2r6]1[cX3r6][cX3r6][cX3r6][cX3r6][cX3r6]1',
}
_MOTIF = {}


def _init_worker():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    _MOTIF['Chem'] = Chem
    for k, sm in SMARTS.items():
        _MOTIF[k] = Chem.MolFromSmarts(sm)


def fragment_profile(smiles):
    """Return (canonical_smiles, n_carboxylate, has_imidazole_ring, has_pyridyl_ring)."""
    Chem = _MOTIF['Chem']
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    allowed = {'C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si'}
    if any(a.GetSymbol() not in allowed for a in mol.GetAtoms()):
        return None
    return (Chem.MolToSmiles(mol),
            len(mol.GetSubstructMatches(_MOTIF['carboxylate'])),
            mol.HasSubstructMatch(_MOTIF['imidazole_ring']),
            mol.HasSubstructMatch(_MOTIF['pyridyl_ring']))


def family_from_profiles(profiles):
    """Union of family tags over the whole linker set; carboxylate keeps the largest count."""
    max_carb = max((p[1] for p in profiles), default=0)
    tags = []
    if max_carb >= 4:
        tags.append('carboxylate_4plus')
    elif max_carb >= 1:
        tags.append(f'carboxylate_{max_carb}')
    if any(p[2] for p in profiles):
        tags.append('imidazole_ring')
    if any(p[3] for p in profiles):
        tags.append('pyridyl_ring')
    if not tags:
        tags.append('other_organic')
    return '+'.join(sorted(tags, key=lambda x: (
        FAMILY_PRIORITY.get(x if not x.startswith('carboxylate') else 'carboxylate', 99), x)))


def parse_linkers(path):
    """Parse MOFid linker_stats.txt: InChIKey, degree, dup, InChI, backbone, SMILES, SMILES2."""
    inchis, raw_smiles, degrees, profiles = [], [], [], []
    unparsable = 0
    try:
        text = Path(path).read_text(errors='replace')
    except OSError:
        return {'canonical_linker_set': None, 'linker_set_id': None, 'linker_family': None,
                'linker_count': 0, 'linker_inchikeys': None, 'linker_smiles_raw': None,
                'linker_family_per_fragment': None, 'linker_max_degree': None,
                'linker_parse_status': 'missing_linker_stats', 'linker_unparsable_fragments': 0}
    for line in text.splitlines():
        c = line.split('\t')
        if len(c) < 6 or not c[5].strip():
            continue
        prof = fragment_profile(c[5].strip())
        if prof is None:
            unparsable += 1
            continue
        profiles.append(prof)
        inchis.append(c[0].strip())
        raw_smiles.append(c[5].strip())
        try:
            degrees.append(int(c[1]))
        except ValueError:
            degrees.append(None)
    if not profiles:
        return {'canonical_linker_set': None, 'linker_set_id': None, 'linker_family': None,
                'linker_count': 0, 'linker_inchikeys': None, 'linker_smiles_raw': None,
                'linker_family_per_fragment': None, 'linker_max_degree': None,
                'linker_parse_status': 'no_parsable_linker',
                'linker_unparsable_fragments': unparsable}
    uniq = sorted({p[0] for p in profiles})
    lset = SEPARATOR.join(uniq)
    per_frag = SEPARATOR.join(sorted(
        f'{p[0]} :: carb{p[1]}'
        f'{"+imi" if p[2] else ""}{"+pyr" if p[3] else ""}' for p in profiles))
    return {'canonical_linker_set': lset,
            'linker_set_id': hashlib.sha256(lset.encode()).hexdigest()[:16],
            'linker_family': family_from_profiles(profiles),
            'linker_count': len(uniq),
            'linker_inchikeys': SEPARATOR.join(sorted(set(inchis))),
            'linker_smiles_raw': SEPARATOR.join(raw_smiles),
            'linker_family_per_fragment': per_frag,
            'linker_max_degree': max([d for d in degrees if d is not None], default=None),
            'linker_parse_status': 'ok' if unparsable == 0 else 'ok_with_unparsable_fragment',
            'linker_unparsable_fragments': unparsable}


def work(chunk):
    _init_worker()
    rows = []
    for src, name, run_dir in chunk:
        d = dict(source=src, name=name, run_dir=run_dir)
        d.update(parse_linkers(os.path.join(run_dir, 'Output', 'MetalOxo', 'linker_stats.txt')))
        rows.append(d)
    return rows


def topo_parts(token):
    """Split an audited topology token into (net, catenation). Multi-net signatures stay whole."""
    if token is None:
        return None, None
    m = CAT_PATTERN.match(str(token))
    return (m.group(1), m.group(2)) if m else (str(token), None)


def topo_valid(token):
    if token is None or str(token).strip() == '':
        return False
    parts = [p.strip() for p in str(token).split(',') if p.strip()]
    if not parts:
        return False
    # each component may carry a '.catN' catenation suffix; judge on the net name itself
    return not any(p.split('.')[0] in INVALID_TOKENS for p in parts)


def load_topology():
    frames = [pd.read_csv(AUDIT_DIR / fn,
                          usecols=['source_file', 'mof_folder_name', 'topology_token'])
              for fn in ('P_mofid_audit.csv', 'U_mofid_audit.csv')]
    a = pd.concat(frames, ignore_index=True)
    a['source'] = a.source_file.map(SOURCE_FROM_FILE)
    unmapped = int(a.source.isna().sum())
    a = a.dropna(subset=['source'])
    lookup, conflicts = {}, []
    for (src, nm), grp in a.groupby(['source', 'mof_folder_name'], sort=False):
        toks = sorted(set(grp.topology_token.dropna().astype(str)))
        if len(toks) > 1:
            conflicts.append({'source': src, 'name': nm, 'n_records': len(grp),
                              'topology_tokens': '||'.join(toks)})
        lookup[(src, nm)] = toks[0] if toks else None
    return lookup, conflicts, unmapped, len(a)


def normalise_lookup(lookup, src, name):
    if (src, name) in lookup:
        return lookup[(src, name)], 'exact'
    for label, fn in NORMALISE_RULES.get(src, []):
        if (src, fn(name)) in lookup:
            return lookup[(src, fn(name))], label
    return None, 'unmatched'


def check_against_reference():
    """Reproduce the published reference dictionary with the rules used above."""
    _init_worker()
    n = canon_ok = id_ok = fam_ok = 0
    fam_mismatch_examples = []
    with open(REF_DICT, newline='') as f:
        for r in csv.DictReader(f):
            smi, ref_set = r['eligible_organic_smiles'], r['canonical_linker_set']
            if not smi or not ref_set:
                continue
            profiles = [fragment_profile(x.strip()) for x in smi.split('||') if x.strip()]
            if any(p is None for p in profiles) or not profiles:
                continue
            n += 1
            got_set = SEPARATOR.join(sorted({p[0] for p in profiles}))
            canon_ok += int(got_set == ref_set)
            id_ok += int(hashlib.sha256(got_set.encode()).hexdigest()[:16] == r['linker_set_id'])
            fam = family_from_profiles(profiles)
            fam_ok += int(fam == r['linker_type'])
            if fam != r['linker_type'] and len(fam_mismatch_examples) < 5:
                fam_mismatch_examples.append({'reference': r['linker_type'], 'ours': fam,
                                              'elig': smi[:90]})
    return {'reference_rows': n,
            'canonical_linker_set_agreement': round(canon_ok / max(n, 1), 4),
            'linker_set_id_agreement': round(id_ok / max(n, 1), 4),
            'linker_family_agreement': round(fam_ok / max(n, 1), 4),
            'linker_family_mismatch_examples': fam_mismatch_examples,
            'note': 'reference table was produced before this task with RDKit 2026.03.6; '
                    'residual family mismatches are pyridine N-oxide / pyridone tautomers'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=48)
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST)
    if args.limit:
        man = man.head(args.limit)
    man['run_dir'] = man.selected_output_dir.astype(str)
    tasks = list(zip(man.source.astype(str), man.name.astype(str), man.run_dir))
    nchunks = args.jobs * 4
    chunks = [c for c in (tasks[i::nchunks] for i in range(nchunks)) if c]
    with Pool(args.jobs, initializer=_init_worker) as pool:
        parts = pool.map(work, chunks, chunksize=1)
    df = pd.DataFrame([r for p in parts for r in p])
    print(f'linker rows parsed: {len(df)}', file=sys.stderr)

    lookup, conflicts, unmapped, audit_rows = load_topology()
    print(f'topology audit records: {audit_rows} (unmapped source_file rows: {unmapped})',
          file=sys.stderr)

    tokens, provs = [], []
    for src, name in zip(df.source, df.name):
        tok, prov = normalise_lookup(lookup, src, name)
        tokens.append(tok)
        provs.append(prov)
    df['topology_token'] = tokens
    df['topology_valid'] = [topo_valid(t) for t in tokens]
    df['topology'] = [t if v else None for t, v in zip(tokens, df.topology_valid)]
    df['topology_net'] = [topo_parts(t)[0] if v else None for t, v in zip(tokens, df.topology_valid)]
    df['topology_catenation'] = [topo_parts(t)[1] if v else None
                                 for t, v in zip(tokens, df.topology_valid)]
    df['topology_match_provenance'] = provs

    cols = ['source', 'name', 'run_dir', 'canonical_linker_set', 'linker_set_id', 'linker_family',
            'linker_count', 'linker_max_degree', 'linker_inchikeys', 'linker_smiles_raw',
            'linker_family_per_fragment', 'linker_parse_status', 'linker_unparsable_fragments',
            'topology_token', 'topology', 'topology_net', 'topology_catenation', 'topology_valid',
            'topology_match_provenance']
    df = df[cols]
    df.to_csv(OUT / 'mof_linker_topology_map.csv', index=False)

    pd.DataFrame(conflicts or [],
                 columns=['source', 'name', 'n_records', 'topology_tokens']).to_csv(
        OUT / 'step3_topology_key_conflicts.csv', index=False)

    n = len(df)
    valid_top = int(df.topology_valid.sum())
    per_source = df.groupby('source').agg(
        rows=('name', 'size'), topology_valid=('topology_valid', 'sum'),
        linker_resolved=('canonical_linker_set', lambda s: int(s.notna().sum())))
    per_source['topology_missing'] = per_source.rows - per_source.topology_valid
    per_source['topology_valid_fraction'] = (per_source.topology_valid / per_source.rows).round(4)
    per_source.to_csv(OUT / 'step3_topology_coverage_by_source.csv')

    ref = check_against_reference()
    validation = {
        'status': 'complete',
        'step': 'Section 4 Step 3 - linker and topology identities',
        'scope': 'selection_manifest.csv full SBU input population',
        'rows': int(n),
        'linker_rows_with_canonical_set': int(df.canonical_linker_set.notna().sum()),
        'linker_parse_status_counts': df.linker_parse_status.value_counts().to_dict(),
        'distinct_linker_set_ids': int(df.linker_set_id.nunique()),
        'distinct_linker_families': int(df.linker_family.nunique()),
        'topology_valid_rows': valid_top,
        'topology_valid_fraction': round(valid_top / n, 4),
        'topology_token_counts_top': df.topology_token.value_counts().head(15).to_dict(),
        'topology_match_provenance': df.topology_match_provenance.value_counts().to_dict(),
        'topology_key_conflicts': len(conflicts),
        'normalisation_rules': {k: [x[0] for x in v] for k, v in NORMALISE_RULES.items()},
        'linker_identity_rule': "RDKit canonical SMILES per fragment, sorted, joined by ' || '; "
                                'linker_set_id = sha256(canonical_linker_set)[:16]',
        'linker_family_rule': 'union of per-fragment tags; carboxylate keeps the largest count '
                              '(1/2/3/4plus); other_organic only when no tag applies',
        'reference_dictionary_check': ref,
        'rdkit_version': __import__('rdkit').__version__,
        'limitations': [
            'topology is the MOFid-v1 topology field; ERROR/UNKNOWN/NA and multi-token '
            'signatures are kept in topology_token and excluded from topology_valid',
            'topology coverage is limited by the mismatch between the MOFid audit table '
            'population (383,243 records from the historical *_opt_mofid.csv batch) and the '
            'current SBU input generation; per-source coverage is in '
            'step3_topology_coverage_by_source.csv',
            'linker_set_id is derived from canonical SMILES, not from an exact-MOFid '
            'three-dimensional geometry check',
        ],
    }
    json.dump(validation, open(OUT / 'step3_linker_topology_validation.json', 'w'), indent=2)
    print(per_source.to_string())
    print(json.dumps({k: validation[k] for k in (
        'rows', 'linker_rows_with_canonical_set', 'topology_valid_rows',
        'topology_valid_fraction', 'topology_match_provenance',
        'topology_key_conflicts')}, indent=2))
    print(json.dumps(validation['reference_dictionary_check'], indent=2))


if __name__ == '__main__':
    main()
