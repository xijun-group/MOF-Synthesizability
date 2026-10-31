#!/usr/bin/env python3
"""Configure an existing runtime for the public lightweight package (reuse only)."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
OLD_ROOT = '/home/sun07ao/syn_release_review_20261011'
BASE_NAME = 'sascore-mofsynth-mofsimplify-ehull-env-v3'
BASE_SHA256 = '3990b8f289519d4427cbd448de352b99fc7864ecc86467454d50cd0c65c58b32'
PY311 = 'cpython-3.11.16-linux-x86_64-gnu/bin/python3.11'
PY313 = 'cpython-3.13.15-linux-x86_64-gnu/bin/python3.13'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, document):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def absolute(path):
    # Retain the venv entrypoint: resolving its Python symlink bypasses the venv.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def absent(path):
    return not path.exists() and not path.is_symlink()


def inspect(args):
    prefix = absolute(args.prefix or ROOT / 'runtime')
    local_bwrap = ROOT / 'tools/bwrap'
    bwrap = str(local_bwrap) if os.access(local_bwrap, os.X_OK) else shutil.which('bwrap')
    tools = {name: shutil.which(name) for name in ('bash', 'tar', 'timeout', 'sed', 'grep', 'ldd', 'sha256sum')}
    tools['bwrap'] = bwrap
    errors = [f'Missing executable: {name}' for name, path in tools.items() if not path]
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        errors.append('The bundled native payload requires Linux x86_64.')
    if sys.version_info < (3, 10):
        errors.append('The installer needs Python 3.10 or newer.')
    payloads = {name: ROOT / 'payload' / name for name in ('base-bundle.tar.gz', 'engine-components.tar.gz')}
    engine_python = ROOT / 'engines/gpu-env/bin/python'
    selected = args.ehull_python or os.environ.get('EHULL_PYTHON')
    if args.reuse:
        candidates = [absolute(selected)] if selected else [engine_python, prefix.parent / 'gpu-env/bin/python',
                                                            prefix / 'qmof-ehull/pyenv/bin/python']
        ehull = next((p for p in candidates if p.is_file() and os.access(p, os.X_OK)), candidates[0])
        for rel in ('paths.env', 'mofsynth-env/bin/python', 'mofsimplify-env/bin/python',
                    'mofid-build/bin/sbu', 'xtb/bin/xtb', 'model/model_weights.json',
                    'qmof-ehull/UMA/checkpoints/uma-s-1p1.pt'):
            if not (prefix / rel).exists():
                errors.append(f'Existing prefix is incomplete: {prefix / rel}')
        if not ehull.is_file() or not os.access(ehull, os.X_OK):
            errors.append('Pass --ehull-python PATH to an existing torch/fairchem/ASE environment.')
    else:
        ehull = engine_python
        if selected:
            errors.append('--ehull-python is for --reuse; the full offline install uses its bundled GPU environment.')
        if not absent(prefix):
            errors.append(f'Refusing to overwrite any existing prefix: {prefix}. Use --reuse for an installed prefix.')
        for path in (ROOT / 'engines', ROOT / 'base-bundle'):
            if not absent(path):
                errors.append(f'Refusing to replace an existing extraction directory: {path}. Install into a fresh package directory.')
        for name, path in payloads.items():
            if not path.is_file():
                errors.append(f'Missing payload/{name}. For the lightweight package use ./install.sh --reuse --prefix EXISTING --ehull-python PATH.')
    disk = shutil.disk_usage(ROOT)
    return {'state': 'ready' if not errors else 'blocked', 'mode': 'reuse' if args.reuse else 'offline',
            'package_root': str(ROOT), 'prefix': str(prefix), 'ehull_python': str(ehull),
            'tools': tools, 'payloads': {name: {'path': str(path), 'exists': path.is_file(),
                        'bytes': path.stat().st_size if path.is_file() else None} for name, path in payloads.items()},
            'platform': {'system': platform.system(), 'machine': platform.machine(), 'libc': platform.libc_ver()},
            'free_bytes': disk.free, 'errors': errors,
            'will_modify_existing_prefix': False, 'will_change_host_tmp_mofid_link': False}


def extract(archive, destination):
    """Extract only ordinary files/directories and internal links into a new directory."""
    if not absent(destination):
        raise RuntimeError(f'Extraction target already exists: {destination}')
    destination.mkdir()
    count = 0
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise RuntimeError(f'Unsafe archive member: {member.name}')
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise RuntimeError(f'Archive member escapes its destination: {member.name}')
            if member.issym() or member.islnk():
                link = Path(member.linkname)
                if link.is_absolute():
                    raise RuntimeError(f'Absolute archive link: {member.name} -> {link}')
                link_target = (target.parent / link if member.issym() else destination / link).resolve()
                if not link_target.is_relative_to(destination.resolve()):
                    raise RuntimeError(f'Archive link escapes its destination: {member.name} -> {link}')
            elif not (member.isfile() or member.isdir()):
                raise RuntimeError(f'Unsupported archive member: {member.name}')
            source.extract(member, destination)
            count += 1
    return count


def python_launchers(directory, interpreter):
    """Shell/Python polyglot launchers support interpreter paths containing spaces."""
    count = 0
    for path in directory.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        with path.open('rb') as source:
            head = source.read(256)
        if not head.startswith(b'#!') or b'python' not in head.split(b'\n', 1)[0].lower():
            continue
        try:
            data = path.read_text()
        except (UnicodeError, OSError):
            continue
        body = data.split('\n', 1)[1] if '\n' in data else ''
        launcher = '#!/bin/sh\n' + "'''exec' " + shlex.quote(str(interpreter)) + ' "$0" "$@"\n' + "' '''\n"
        path.write_text(launcher + body)
        count += 1
    return count


def relocate_engines(prefix):
    engines = ROOT / 'engines'
    py311 = engines / 'python' / PY311
    py313 = engines / 'python' / PY313
    if not py311.is_file() or not py313.is_file():
        raise RuntimeError('engine-components payload lacks the pinned CPython 3.11/3.13 interpreters.')
    replacements = {
        OLD_ROOT + '/runtime/qmof-ehull/pyenv': str(engines / 'cpu-env'),
        OLD_ROOT + '/gpu-env': str(engines / 'gpu-env'),
        OLD_ROOT + '/runtime': str(prefix),
        OLD_ROOT + '/python': str(engines / 'python'),
    }
    changes = []
    for name in ('gpu-env', 'cpu-env'):
        env = engines / name
        config = env / 'pyvenv.cfg'
        text = config.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        lines = [line for line in text.splitlines() if not line.startswith(('home =', 'executable =', 'command ='))]
        config.write_text(f'home = {py311.parent}\nexecutable = {py311}\n' + '\n'.join(lines) + '\n')
        # All these entries belong to the newly extracted package, never a reused prefix.
        for name_python in ('python', 'python3', 'python3.11'):
            target = env / 'bin' / name_python
            if target.is_symlink():
                target.unlink()
            elif target.exists():
                raise RuntimeError(f'Refusing to replace an unexpected regular interpreter: {target}')
            target.symlink_to(os.path.relpath(py311, target.parent))
        sites = env / 'lib/python3.11/site-packages'
        for path in list((env / 'bin').iterdir()) + list(sites.glob('*.pth')):
            if not path.is_file() or path.is_symlink():
                continue
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            try:
                before = path.read_text()
            except UnicodeError:
                continue
            after = before
            for old, new in replacements.items():
                after = after.replace(old, new)
            if after != before:
                path.write_text(after)
                changes.append(str(path))
        (sites / 'qmof_thermo_payload.pth').write_text(str(prefix / 'qmof-ehull/src') + '\n')
        if name == 'gpu-env':
            (sites / 'cpu_env_dependencies.pth').write_text(str(engines / 'cpu-env/lib/python3.11/site-packages') + '\n')
        python_launchers(env / 'bin', env / 'bin/python')
    return {'py311': str(py311), 'py313': str(py313), 'relocated_text_files': changes}


class Steps:
    def __init__(self, log_dir):
        self.directory = log_dir
        self.records = []
        log_dir.mkdir(exist_ok=True)

    def run(self, name, command, timeout=900):
        start = time.monotonic()
        record = {'name': name, 'command': [str(x) for x in command], 'started_utc': utc()}
        self.records.append(record)
        env = dict(os.environ)
        for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH'):
            env.pop(key, None)
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1', PIP_NO_INDEX='1', UV_OFFLINE='1')
        with (self.directory / (name + '.log')).open('w') as output:
            proc = subprocess.Popen([str(x) for x in command], stdout=output, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except BaseException:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=2)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                raise
        record.update(returncode=code, wall_s=time.monotonic() - start, log=str(self.directory / (name + '.log')))
        write(self.directory / 'steps.json', self.records)
        if code:
            raise RuntimeError(f'{name} failed (exit {code}); inspect {record["log"]}')


def install(args, plan):
    prefix = Path(plan['prefix'])
    logs = ROOT / 'install-logs'
    steps = Steps(logs)
    receipt = {'state': 'running', 'started_utc': utc(), 'plan': plan, 'steps': steps.records}
    write(logs / 'receipt.json', receipt)
    started = time.monotonic()
    try:
        if not args.reuse:
            checksums_path = ROOT / 'payload/checksums.json'
            expected = json.loads(checksums_path.read_text()) if checksums_path.exists() else {}
            expected['base-bundle.tar.gz'] = BASE_SHA256
            hashes = {}
            for name, item in plan['payloads'].items():
                digest = sha(Path(item['path']))
                if name in expected and digest != expected[name]:
                    raise RuntimeError(f'Payload checksum mismatch: {name}')
                hashes[name] = digest
            receipt['payload_sha256'] = hashes
            receipt['engine_members'] = extract(ROOT / 'payload/engine-components.tar.gz', ROOT / 'engines')
            receipt['base_members'] = extract(ROOT / 'payload/base-bundle.tar.gz', ROOT / 'base-bundle')
            py313 = ROOT / 'engines/python' / PY313
            # The original installer removes PREFIX even without --force. Run
            # an audited copy that instead refuses every existing destination.
            original = ROOT / 'base-bundle' / BASE_NAME / 'install.sh'
            script = original.read_text()
            before = 'rm -rf "$PREFIX"\nmkdir -p "$PREFIX/.tmp"'
            after = '[ ! -e "$PREFIX" ] && [ ! -L "$PREFIX" ] || die "refusing an existing prefix"\nmkdir -p "$PREFIX/.tmp"'
            if script.count(before) != 1:
                raise RuntimeError('Upstream installer safety guard no longer matches the pinned source.')
            script = script.replace(before, after, 1)
            for variable in ('VENV', 'SC'):
                script = script.replace(f'"${variable}/bin/pip" install', f'"${variable}/bin/python" -m pip install')
            safe_installer = original.with_name('install-mofsynth-fast-safe.sh')
            safe_installer.write_text(script)
            receipt['original_installer_sha256'] = sha(original)
            receipt['safe_installer_sha256'] = sha(safe_installer)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            if not absent(prefix):
                raise RuntimeError(f'Prefix appeared during extraction; refusing to overwrite: {prefix}')
            steps.run('base-install', ['bash', safe_installer, '--prefix', prefix,
                                      '--py313', py313, '--no-tmp-link'], timeout=1200)
            receipt['relocation'] = relocate_engines(prefix)
            for env_name in ('mofsynth-env', 'mofsimplify-env', 'xtb'):
                interpreter = prefix / ('mofsimplify-env/bin/python' if env_name == 'mofsimplify-env' else 'mofsynth-env/bin/python')
                python_launchers(prefix / env_name / 'bin', interpreter)
            import_code = ('import json,sys,torch,ase,fairchem.core,qmof_thermo; '
                           'print(json.dumps({"executable":sys.executable,"prefix":sys.prefix,'
                           '"torch":torch.__version__,"cuda":torch.version.cuda,'
                           '"cuda_available":torch.cuda.is_available(),"ase":ase.__version__,'
                           '"qmof_thermo":qmof_thermo.__file__}))')
            for name in ('cpu-env', 'gpu-env'):
                steps.run(name + '-imports', [ROOT / 'engines' / name / 'bin/python', '-c', import_code], timeout=180)
            steps.run('mofsynth-imports', [prefix / 'mofsynth-env/bin/python', '-c',
                      'import rdkit,mofid,mofsynth_qm; print(rdkit.__version__,mofid.__file__,mofsynth_qm.__file__)'], timeout=120)
            steps.run('mofsimplify-imports', [prefix / 'mofsimplify-env/bin/python', '-c',
                      'import molSimplify,pymatgen,tensorflow,sklearn; print(molSimplify.__file__,tensorflow.__version__,sklearn.__version__)'], timeout=180)
            steps.run('original-plan', [prefix / 'mofsynth-env/bin/python', prefix / 'predict/predict_synthesizability.py',
                                       '--prefix', prefix, '--ehull-python', plan['ehull_python'], '--plan'], timeout=30)
        version_file = ROOT / 'VERSION'
        config = {'prefix': str(prefix), 'ehull_python': plan['ehull_python'],
                  'installed_version': version_file.read_text().strip() if version_file.exists() else '3.1.0',
                  'installation_mode': plan['mode'], 'installed_utc': utc(),
                  'bwrap_path': plan['tools']['bwrap'], 'package_root': str(ROOT)}
        write(ROOT / 'runtime.json', config)
        receipt.update(state='complete', ended_utc=utc(), wall_s=time.monotonic() - started, runtime=config)
        write(logs / 'receipt.json', receipt)
        print(json.dumps({'state': 'complete', 'runtime_config': str(ROOT / 'runtime.json'), **config}, indent=2))
        return 0
    except BaseException as exc:
        receipt.update(state='failed', ended_utc=utc(), wall_s=time.monotonic() - started, error=repr(exc))
        write(logs / 'receipt.json', receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Inspect tools/payloads/paths without installing or writing configuration.')
    parser.add_argument('--prefix', help='Fresh offline installation destination, or the existing prefix with --reuse.')
    parser.add_argument('--ehull-python', help='Existing Ehull venv entrypoint for --reuse; retain its bin/python path.')
    parser.add_argument('--reuse', action='store_true', default=True, help='Only write this package runtime.json; never modify the existing prefix.')
    args = parser.parse_args()
    plan = inspect(args)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0 if not plan['errors'] else 2
    if plan['errors']:
        for error in plan['errors']:
            print('ERROR: ' + error, file=sys.stderr)
        return 2
    with (ROOT / '.install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return install(args, plan)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
