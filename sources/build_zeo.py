#!/usr/bin/env python3
"""Rebuild the delivered Zeo++ source with conservative optimization flags."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--verification-python', type=Path)
    ap.add_argument('--verification-script', type=Path)
    ap.add_argument('--verification-input-root', type=Path)
    args = ap.parse_args()
    args.out.mkdir(exist_ok=False)
    receipt = {'state': 'running', 'host': socket.gethostname(),
               'start_unix': time.time(), 'source': str(args.source),
               'cpu_affinity': sorted(os.sched_getaffinity(0))}

    def write():
        (args.out/'build_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')

    try:
        inputs = {str(p.relative_to(args.source)): sha(p) for p in args.source.rglob('*')
                  if p.is_file() and p.suffix in ['.cc', '.h', '.hh', '.hpp', '.cpp', '.a', '.mk']}
        for name in ['Makefile', 'Makefile.dep', 'LICENSE', 'network']:
            inputs[name] = sha(args.source/name)
        (args.out/'source_sha256.json').write_text(json.dumps(inputs, indent=2)+'\n')
        source = args.out/'source'
        shutil.copytree(args.source, source, symlinks=True)
        for obj in source.glob('*.o'):
            obj.unlink()
        (source/'network').unlink()
        flags = ['-O2', '-g', '-fno-fast-math', '-ffp-contract=off',
                 '-fno-tree-vectorize', '-fno-strict-aliasing']
        command = ['make', '-j8', 'CFLAGS='+' '.join(flags), 'network']
        receipt['command'] = command
        receipt['flags'] = flags
        receipt['compiler'] = subprocess.check_output(['g++', '--version'], text=True)
        write()
        with (args.out/'build.log').open('w') as log:
            proc = subprocess.run(command, cwd=str(source), stdout=log, stderr=subprocess.STDOUT,
                                  timeout=900)
        receipt['build_returncode'] = proc.returncode
        if proc.returncode:
            raise RuntimeError('Compilation failed; see build.log')
        binary = source/'network'
        shutil.copy2(binary, args.out/'network')
        receipt['original_binary_sha256'] = sha(args.source/'network')
        receipt['candidate_binary_sha256'] = sha(binary)
        receipt['binary_bytes'] = binary.stat().st_size
        for label, executable in [('original', args.source/'network'), ('candidate', binary)]:
            for name, cmd in [('ldd', ['ldd', str(executable)]),
                              ('readelf_comment', ['readelf', '-p', '.comment', str(executable)]),
                              ('elf_versions', ['readelf', '--version-info', str(executable)])]:
                data = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
                (args.out/(label+'_'+name+'.txt')).write_text(data)
                if name=='elf_versions':
                    receipt[label+'_glibc_requirements'] = sorted(set(re.findall(r'GLIBC_[0-9.]+', data)),
                        key=lambda v: tuple(map(int, v.split('_')[1].split('.'))))
                    receipt[label+'_glibcxx_requirements'] = sorted(set(re.findall(r'GLIBCXX_[0-9.]+', data)),
                        key=lambda v: tuple(map(int, v.split('_')[1].split('.'))))
        receipt['state'] = 'complete'
    except BaseException as exc:
        receipt['state'] = 'failed'
        receipt['error'] = repr(exc)
        (args.out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        receipt['end_unix'] = time.time()
        receipt['wall_s'] = receipt['end_unix']-receipt['start_unix']
        write()
    if args.verification_script:
        command = [str(args.verification_python), str(args.verification_script),
            '--original', str(args.source/'network'), '--candidate', str(args.out/'network'),
            '--input-root', str(args.verification_input_root), '--out', str(args.out/'verification')]
        with (args.out/'verification.log').open('w') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        receipt['verification_returncode'] = proc.returncode
        receipt['verification_command'] = command
        write()
        if proc.returncode:
            raise RuntimeError('Verification failed; see verification.log')


if __name__ == '__main__':
    main()
