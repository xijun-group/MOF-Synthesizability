# MOF Synthesizability

Run MOFSynth Fast 3.1.1 on CIF files to calculate five descriptors and a synthesizability score.

## Requirements

- Linux x86_64 with Python 3.10 or later.
- An existing MOFSynth portable runtime containing MOFid, xTB, Java, MOFSimplify, the prediction model and QMOF reference hull.
- A compatible Python environment with PyTorch, fairchem and ASE for Ehull calculations.
- For GPU execution, an NVIDIA GPU with a compatible driver.
- glibc >=2.34, GLIBCXX >=3.4.29, bash, tar, timeout, sed and grep. The system must allow unprivileged user namespaces for bubblewrap.

This repository uses an existing runtime; it does not install those environments from scratch.

## 1. Download the code

```bash
git clone https://github.com/xijun-group/MOF-Synthesizability.git
cd MOF-Synthesizability
```

## 2. Set up UMA

Request access to [facebook/UMA](https://huggingface.co/facebook/UMA) using your Hugging Face account. Once access is granted:

```bash
python3 -m pip install huggingface_hub
hf auth login
python3 install/setup_uma.py --prefix /absolute/path/runtime
```

If you already have `uma-s-1p1.pt`, use the local file instead:

```bash
python3 install/setup_uma.py --prefix /absolute/path/runtime \
    --checkpoint /absolute/path/uma-s-1p1.pt
```

## 3. Configure the runtime

Replace the paths below with your existing runtime directory and Ehull Python interpreter:

```bash
bash install.sh --prefix /absolute/path/runtime \
    --ehull-python /absolute/path/gpu-env/bin/python
```

Check the configuration:

```bash
bash predict.sh --plan
python3 verify.py --quick
```

The configuration is saved locally, so you only need to do this once per installation.

## 4. Predict a single MOF

Run the included example:

```bash
bash predict.sh --cif examples/reference58.cif --label demo --gpu-id 0
```

Run your own CIF and choose an output directory:

```bash
bash predict.sh --cif /absolute/path/my_mof.cif \
    --label my_mof --work /absolute/path/results/my_mof --gpu-id 0
```

Without `--work`, results are written to `runs/<label>/prediction.json`.
The output includes solvent-removal stability, RMSD, organic-fragment HOMO, sascore, Ehull and the final score.

To run a complete installation check on the reference structure:

```bash
python3 verify.py --gpu-id 0
```

## 5. Predict multiple MOFs

```bash
python3 predict_batch.py --cifs /path/a.cif /path/b.cif \
    --out /absolute/path/batch-results --slots 2 --cpus-per-slot 6 --gpus 0
```

For a larger set, create `cifs.txt` with one absolute CIF path per line:

```bash
python3 predict_batch.py --cif-list cifs.txt \
    --out /absolute/path/batch-results --slots 4 --cpus-per-slot 6 --gpus 0,1
```

- `--slots`: number of concurrent MOF jobs.
- `--cpus-per-slot`: CPU cores assigned to each job; slots × cores must fit the available CPUs.
- `--gpus`: comma-separated GPU indices.
- CIF filenames must be unique within a batch.

Results are saved as `<output>/<label>/prediction.json`; the batch summary is `<output>/summary.csv`.
Use a fresh output directory when changing input structures or calculation settings.

## Command help

```bash
bash predict.sh --help
python3 predict_batch.py --help
```
