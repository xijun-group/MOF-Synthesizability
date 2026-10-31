#!/usr/bin/env python3
"""Install the external, pinned UMA checkpoint using the user's own HF access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def verify(path, spec):
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        raise ValueError('UMA checkpoint missing or incorrect size')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != spec['sha256']:
        raise ValueError('UMA checkpoint SHA-256 mismatch; expected the frozen uma-s-1p1 weights')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument('--prefix', type=Path)
    target.add_argument('--ehull-root', type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--checkpoint', type=Path, help='already obtained local checkpoint; no network')
    mode.add_argument('--check', action='store_true', help='verify installed checkpoint; no network')
    args = ap.parse_args()
    spec = json.loads((Path(__file__).resolve().parents[1] / 'specs/uma.json').read_text())
    root = args.ehull_root or args.prefix / 'qmof-ehull'
    if not (root / 'src/qmof_thermo/mlip.py').is_file():
        ap.error('install the QMOF-Ehull code first (install.sh)')
    dest = root / 'UMA/checkpoints/uma-s-1p1.pt'
    try:
        if args.check:
            verify(dest, spec)
            print('UMA checkpoint SHA-256 verified. Inference not executed.')
            return 0
        if args.checkpoint:
            source = args.checkpoint.resolve()
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                ap.error('install huggingface_hub, accept the conditions at https://huggingface.co/facebook/UMA, then run hf auth login')
            try:
                source = Path(hf_hub_download(repo_id=spec['repo_id'], filename=spec['filename'],
                                             revision=spec['revision'], token=True))
            except Exception:
                # Do not echo authentication headers, tokens, or signed URLs from SDK exceptions.
                print('UMA download failed. Check network access, accept the upstream access conditions, and run hf auth login with an authorized account.', flush=True)
                return 1
        verify(source, spec)
        if source != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix='.uma-', dir=dest.parent)
            os.close(fd)
            tmp = Path(name)
            try:
                shutil.copyfile(source, tmp)
                verify(tmp, spec)
                tmp.replace(dest)
            finally:
                tmp.unlink(missing_ok=True)
        print('UMA checkpoint installed and SHA-256 verified. Inference not executed.')
        return 0
    except (OSError, ValueError) as exc:
        print(f'UMA setup failed: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
