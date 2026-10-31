"""Real mixed-linker geometry/QM verification, isolated from model scoring.

Usage: bash validation/verify_rmsd_sa_max.sh RUNTIME_PREFIX
Requires fresh validation/rmsd_sa_max output. Fixture inputs are copied once
from the supplied selected-structure export and included in the delivery.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation/rmsd_sa_max'
PREFIX = Path(sys.argv[1]).absolute()
SID = 'DB1-Cu2O8N2-irmof14_A-irmof6_A_No1'
SOURCE = Path('/home/cheyue/data/ehull-vs-fivefeatures-syn')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

driver = load('sa_rmsd_driver', ROOT / 'predict/predict_synthesizability.py')
selector = load('sa_rmsd_selector', ROOT / 'predict/rules/rmsd_sa_target.py')
std = load('sa_rmsd_std', ROOT / 'predict/rules/standardize_linker_smiles.py')
scheduler = load('sa_rmsd_scheduler', ROOT / 'predict/stage_scheduler.py')

def main():
    if (OUT / 'serial').exists() or (OUT / 'parallel').exists():
        raise RuntimeError('Fresh validation directories required')
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = OUT / 'fixtures'
    if not fixtures.exists():
        fixtures.mkdir()
        shutil.copy2(SOURCE / 'selected_cif' / (SID + '_opt_charge.cif'), fixtures / 'input.cif')
        for filename in ['linker_stats.txt', 'linkers.cif']:
            shutil.copy2(SOURCE / 'mofid' / SID / 'Output/MetalOxo' / filename, fixtures / filename)
    fixture_hashes = {str(p.name): sha(p) for p in fixtures.iterdir()}
    # Same runtime environment as the launcher, scoped to this process and its children.
    for key, value in driver.read_paths_env(PREFIX).items():
        os.environ[key] = value
    os.environ.update(driver.XTB_ENV)
    os.environ['PATH'] = ':'.join(str(PREFIX/p) for p in ['xtb/bin','mofsynth-env/bin','java/bin','mofid-build/bin']) + ':' + os.environ['PATH']
    ctx = {'prefix': PREFIX, 'rules': ROOT / 'predict/rules', 'predict_dir': ROOT / 'predict',
           'model_dir': PREFIX / 'model', 'xtb': str(PREFIX / 'xtb/bin/xtb'),
           'venv_python': str(PREFIX / 'mofsynth-env/bin/python'),
           'java_home': str(PREFIX / 'java'), 'mofid_root': str(PREFIX / 'mofid-build'),
           'fast_sbu_lib': str(ROOT / 'acceleration/openbabel'), 'fast_rmsd': True,
           'ehull_python': ''}
    parsed = std.standardize_mofid_linker_stats(fixtures / 'linker_stats.txt')
    assert parsed['complete'] and len(parsed['standardised_set']) == 3
    scores = driver.score_fragments(ctx, parsed['standardised_set'])
    selection = selector.select_parent(scores)
    checks = {'three_complete_distinct_parents': True}
    probe = OUT / 'one_cell_geometry'
    probe.mkdir(exist_ok=True)
    try:
        geom = selector.extract_geometry(fixtures / 'linkers.cif', probe / 'linker.xyz', selection)
        checks['one_cell_matches_or_explicitly_stops'] = geom['chosen']['standardized_parent'] == selection['target_parent']
    except ValueError:
        checks['one_cell_matches_or_explicitly_stops'] = not (probe / 'linker.xyz').exists()
    # Missing exact target now follows the user-authorized largest-fragment fallback.
    negative = OUT / 'missing_target'
    negative.mkdir(exist_ok=True)
    fallback = selector.extract_geometry(fixtures / 'linkers.cif', negative / 'linker.xyz',
                                        selector.select_parent(driver.score_fragments(ctx, ['CCO'])))
    checks['missing_target_uses_largest_geometry'] = (
        fallback['fallback_used'] and not fallback['selected_matches_sa_parent'] and
        fallback['chosen']['natoms'] == max(r['natoms'] for r in fallback['components']))
    work = OUT / 'serial'
    dest = work / 'mofid/Output/MetalOxo'
    dest.mkdir(parents=True)
    for filename in ['linker_stats.txt', 'linkers.cif']:
        shutil.copy2(fixtures / filename, dest / filename)
    serial = driver.step_rmsd(ctx, fixtures / 'input.cif', work, 600, 2, sa_scores=scores)
    geometry = json.loads((work / 'rmsd/geometry_selection.json').read_text())
    checks['default_two_cell_exact_target_match'] = geometry['chosen']['standardized_parent'] == selection['target_parent']
    checks['selected_is_smaller_than_largest_component'] = geometry['chosen']['natoms'] < max(r['natoms'] for r in geometry['components'])
    # Independently compare the serialized XYZ to the selected OB component.
    from openbabel import openbabel as ob
    conv = ob.OBConversion(); conv.SetInFormat('cif')
    original = ob.OBMol(); assert conv.ReadFile(original, serial['linkers_cif'])
    component = list(original.Separate())[geometry['chosen']['component_index']]
    atoms = list(ob.OBMolAtomIter(component))
    xyz_lines = (work / 'rmsd/linker.xyz').read_text().splitlines()[2:]
    expected = [(ob.GetSymbol(a.GetAtomicNum()), a.GetX(), a.GetY(), a.GetZ()) for a in atoms]
    actual = [(v[0], *map(float,v[1:4])) for v in (line.split() for line in xyz_lines) if v]
    checks['same_explicit_atoms_no_addition'] = [v[0] for v in actual] == [v[0] for v in expected]
    max_error = max(abs(a-b) for x,y in zip(actual,expected) for a,b in zip(x[1:],y[1:]))
    checks['original_coordinates_preserved_to_xyz_precision'] = max_error < 1e-5
    checks['target_is_sa_max'] = serial['target_selection']['target_sa'] == max(scores['per_fragment'].values())
    checks['deltae_is_same_fragment_opt_minus_initial_sp'] = serial['delta_energy_hartree'] == serial['opt_energy_hartree'] - serial['sp0_energy_hartree']
    checks['deltae_kcal_conversion'] = serial['delta_energy_kcal_mol'] == serial['delta_energy_hartree'] * 627.51
    checks['rmsd_is_minimum_alignment_result'] = serial['rmsd_a'] == min(serial['all_method_values'].values())
    # A second fresh QM execution through the real scheduler; only decompositions reused.
    parallel = OUT / 'parallel'
    parallel.mkdir()
    for folder in ['mofid','mofid_supercell2']:
        shutil.copytree(work / folder, parallel / folder)
    args = SimpleNamespace(timeout=600, workers=1, rmsd_supercell=2, stage_jobs=2)
    res = scheduler.collect(ctx, args, fixtures / 'input.cif', SID, parallel,
                            lambda: parsed['standardised_set'], {'ehull','mofsimplify','b'})['rmsd']
    checks['scheduler_receives_fragment_identities'] = json.loads((parallel/'stage_logs/rmsd.task.json').read_text())['identities'] == parsed['standardised_set']
    checks['serial_parallel_same_target'] = res['target_selection'] == serial['target_selection']
    checks['serial_parallel_same_initial_xyz'] = sha(work/'rmsd/linker.xyz') == sha(parallel/'rmsd/linker.xyz')
    checks['serial_parallel_same_optimized_xyz'] = sha(work/'rmsd/opt/xtbopt.xyz') == sha(parallel/'rmsd/opt/xtbopt.xyz')
    checks['serial_parallel_same_rmsd_and_deltae'] = all(res[k] == serial[k] for k in ['rmsd_a','delta_energy_hartree','sp0_energy_hartree','opt_energy_hartree','sp1_energy_hartree'])
    cached = driver.step_rmsd(ctx, fixtures/'input.cif', work, 600, 2, sa_scores=scores)
    checks['same_protocol_cache_valid'] = cached == serial
    changed = dict(scores, per_fragment={s: 1.0 for s in scores['per_fragment']}, sascore=1.0)
    try:
        driver.step_rmsd(ctx, fixtures/'input.cif', work, 600, 2, sa_scores=changed)
        checks['changed_target_cache_rejected'] = False
    except SystemExit as exc:
        checks['changed_target_cache_rejected'] = exc.code == 3
    checks['fixture_inputs_unchanged'] = all(sha(fixtures/name)==value for name,value in fixture_hashes.items())
    result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
              'structure_id': SID, 'selection': selection, 'sa_scores': scores,
              'selected_natoms': geometry['chosen']['natoms'],
              'largest_component_natoms': max(r['natoms'] for r in geometry['components']),
              'coordinate_serialization_max_error_A': max_error,
              'serial_result': serial, 'parallel_result': res, 'fixture_sha256': fixture_hashes,
              'xtb_sha256': sha(ctx['xtb']), 'new_geometry_optimizations': 2,
              'new_homo_campaigns': 0, 'model_fits_or_scores': 0, 'new_ehull': 0}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert all(checks.values()), checks

if __name__ == '__main__':
    main()
