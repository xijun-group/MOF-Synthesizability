# MOFSynth Fast 3.1.1

GitHub distribution of the supplied 3.1.1 lightweight package, including the
2026-10-25 stereo-compatible RMSD selection revision. This package reuses an
existing portable runtime. **It does not include UMA weights, Python environments,
or a standalone runtime installer.** Original scientific code and historical
validation evidence are preserved.

## Requirements and installation

Linux x86_64, Python >=3.10, bash, tar, timeout, sed, grep, glibc >=2.34,
GLIBCXX >=3.4.29, and unprivileged user namespaces for bubblewrap are required.
The existing runtime must supply MOFid, xTB, Java, MOFSimplify, the frozen VPU
model and preprocessing, the QMOF reference hull, and a compatible torch/fairchem
interpreter. Do not substitute an arbitrary environment or a newly trained model.

After obtaining that runtime and accepting the access conditions at
https://huggingface.co/facebook/UMA, install UMA with your own authorized account:

```bash
python3 -m pip install huggingface_hub
hf auth login
python3 install/setup_uma.py --prefix /absolute/path/runtime
```

Alternatively use an already authorized local copy (no network needed):

```bash
python3 install/setup_uma.py --prefix /absolute/path/runtime \
    --checkpoint /absolute/path/uma-s-1p1.pt
```

The helper verifies the frozen checkpoint size and SHA-256, using the upstream
revision recorded in `specs/uma.json`. It does not accept upstream conditions
on your behalf. Keep authentication caches and tokens outside this repository.
The setup helper writes only the checkpoint location in the selected runtime;
the runtime-reuse installer below does not modify the runtime itself.

```bash
bash install.sh --plan --prefix /absolute/path/runtime \
    --ehull-python /absolute/path/gpu-env/bin/python
bash install.sh --prefix /absolute/path/runtime \
    --ehull-python /absolute/path/gpu-env/bin/python
bash predict.sh --plan
python3 verify.py --quick
bash predict.sh --cif examples/reference58.cif --label demo --gpu-id 0
```

Runtime reuse is the default in this public distribution; `--reuse` remains accepted.
An incomplete runtime is reported as an error. `verify.py --quick` checks paths and
interpreter imports; it does not establish successful GPU inference. Run
`python3 verify.py --gpu-id 0` to execute a fresh reference calculation.
Use fresh work/output directories when changing protocols or input structures.

## Scientific protocol

Default Ehull convergence requires all four conditions for five consecutive distinct
optimizer steps: atomic force <0.01 eV/A, maximum absolute principal Cauchy stress
<0.05 GPa at zero pressure, FrechetCellFilter generalized force <0.03, and E/N
range <=1e-5 eV/atom over six configurations spanning five updates. The step cap
is 10000; reaching it is not convergence. Old force-only caches are rejected.
`--baseline` is explicitly historical. Restart/perturbation diagnostics are not
mandatory branches and no minimum-over-repeats selection is introduced.

The parent-normalization and RMSD selection revisions in this ZIP remain unchanged.
Scores are uncalibrated operational scores, not synthesis probabilities; no default
0.5 cutoff is prescribed. Packaging does not retrain or recalibrate the model.
See `EHULL_COMBINED_3_1_1.txt`, `RMSD_SA_STEREO_V4.txt`, and `README.txt` for details.
The latter retains historical offline-package instructions, which do not apply to
this lightweight GitHub distribution. Historical validation is not a new run.

## Publishing

Upload this directory as the repository. Its existing native Linux binaries and
source archive are small enough for ordinary Git; retain `LICENSES/`, `sources/`
and the OpenBabel patch/source. UMA must not be added to commits or Release archives.
No new blanket license has been assigned to project-owned code. Existing third-party
notices apply; see `THIRD_PARTY_NOTICES.md`.

Rebuild the public archive outside the checkout:

```bash
python3 package_public.py --output /outside/checkout/mofsynth-fast-3.1.1-github.zip
sha256sum -c MANIFEST.sha256 --quiet
```

The packager excludes runtime configuration, environments, caches and external
weights even after local use. `PUBLIC_RELEASE.json` records source provenance.
