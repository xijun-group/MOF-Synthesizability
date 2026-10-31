"""Rebuild the public manifest and archive without local runtimes or weights."""
import argparse, hashlib, zipfile
from pathlib import Path
r=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
if a.output.resolve().is_relative_to(r):ap.error('output must be outside the checkout')
excluded={'.git','__pycache__','.cache','.huggingface','runtime','engines','base-bundle','runs','install-logs','dist'}
files=[]
for p in sorted(r.rglob('*')):
 rel=p.relative_to(r)
 if any(x in excluded for x in rel.parts) or p.name in {'runtime.json','.install.lock','MANIFEST.sha256'} or p.name.startswith('.env') or p.suffix in {'.pt','.pth','.safetensors','.pyc'} or (rel.parts[0]=='payload' and p.name.endswith('.tar.gz')):continue
 if p.is_symlink():raise ValueError(f'Unexpected symlink: {rel}')
 if p.is_file():
  if p.stat().st_size>=100*1024**2:raise ValueError(f'File exceeds Git limit: {rel}')
  files.append(p)
(r/'MANIFEST.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(r).as_posix()+'\n' for p in files))
files.append(r/'MANIFEST.sha256');a.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,'mofsynth-fast-3.1.1/'+p.relative_to(r).as_posix())
print(f'Packaged {len(files)} files: {a.output}')
