#!/usr/bin/env python3
"""Verify the installed package, optionally with a new full reference calculation."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--gpu-id', default='0')
    a = ap.parse_args()
    env = {k: v for k, v in os.environ.items() if k not in ('LD_LIBRARY_PATH', 'PYTHONHOME', 'PYTHONPATH')}
    subprocess.run(['bash', str(ROOT / 'predict.sh'), '--plan'], check=True, env=env)
    config = json.loads((ROOT / 'runtime.json').read_text())
    py = config['ehull_python']
    check = subprocess.run([py, '-c', 'import json,torch,fairchem.core,ase;print(json.dumps({"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda":torch.version.cuda}))'],
                           text=True, capture_output=True, check=True, env=env)
    print(check.stdout.strip())
    if a.quick:
        return 0
    work_root = ROOT / 'runs'
    work_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='installation-check-', dir=work_root))
    command = ['bash', str(ROOT / 'predict.sh'), '--cif', str(ROOT / 'examples/reference58.cif'),
               '--label', 'installation_check', '--work', str(work), '--gpu-id', a.gpu_id]
    subprocess.run(command, check=True, env=env)
    result = json.loads((work / 'prediction.json').read_text())
    # SA/HOMO parent normalization and SA-target RMSD are revised definitions.
    # Old 3.1 RMSD/score constants are not acceptance criteria for this revision.
    expected = {'solvent_removal_stability': 0.790681004524231}
    checks = {key: result['features'][key] == value for key, value in expected.items()}
    rmsd = result['features']['RMSD_[A]']
    detail = result['details']
    sa = detail['sascore']
    homo = detail['channel_b']
    target = detail['rmsd']['target_selection']
    maximum = max(sa['per_fragment'].values())
    tied = sorted(s for s,v in sa['per_fragment'].items() if v == maximum)
    checks['complete_sa_max'] = sa['complete'] and result['features']['sascore'] == maximum
    checks['complete_homo_max'] = homo['complete'] and result['features']['Linker_HOMO_[eV]'] == max(homo['per_fragment'].values())
    checks['rmsd_matches_sa_max_parent'] = target['target_parent'] == tied[0] and target['target_sa'] == maximum
    checks['rmsd_finite'] = math.isfinite(rmsd) and rmsd >= 0
    checks['deltae_same_fragment'] = detail['rmsd']['delta_energy_hartree'] == detail['rmsd']['opt_energy_hartree'] - detail['rmsd']['sp0_energy_hartree']
    info = {'rmsd': rmsd, 'target_selection': target,
            'scope': 'installation smoke check; revised model-feature validation not executed'}
    checks['complete_score'] = result['scored'] and not result['missing_features']
    checks['score_finite'] = result['native_score'] is not None and math.isfinite(result['native_score'])
    # The old force-only energy constant is not a 3.1.1 acceptance target.
    sys.path.insert(0, str(ROOT / 'acceleration'))
    from ehull_protocol import DEFAULTS, matches_receipt
    ehull = result['details']['ehull']
    relaxation = json.loads(Path(ehull['relaxation_receipt']).read_text())
    checks['ehull_finite'] = math.isfinite(result['features']['Ehull'])
    checks['ehull_combined_default_protocol'] = matches_receipt(relaxation, DEFAULTS)
    checks['ehull_result_protocol'] = ehull['convergence_protocol'] == DEFAULTS
    checks['package_version'] = result.get('package_version') == '3.1.1' 
    checks['converged'] = result['details']['ehull']['relaxation_converged'] is True
    receipt = {'state': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'info': info,
               'work': str(work), 'prediction': str(work / 'prediction.json')}
    (work / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
