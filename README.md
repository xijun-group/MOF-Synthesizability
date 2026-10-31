# MOF Synthesizability

MOFSynth Fast 3.1.1 is a CIF-to-score workflow for assessing and ranking candidate metal–organic frameworks (MOFs). Starting from a crystal structure, it computes five inputs that describe the organic building blocks, their geometric deformation, framework stability after solvent removal, and energetic stability relative to competing phases. A pretrained positive–unlabeled learning model combines these inputs into a single synthesizability score.

The tool supports both single-structure prediction and batch screening. You provide CIF files; the pipeline performs structure decomposition, molecular calculations, framework relaxation and model inference, then saves the calculated properties and score.

## What properties does it calculate?

| Model input | How it is obtained | What it describes |
|---|---|---|
| **Solvent-removal stability** (`solvent_removal_stability`) | Zeo++ and molSimplify generate pore/geometric and chemical descriptors from the MOF structure. The pretrained MOFSimplify solvent-removal neural network converts them into a stability score. | The model-estimated tendency of the framework to retain its structure after solvent removal, a relevant consideration for obtaining an activated porous material. |
| **Fragment deformation RMSD** (`RMSD_[A]`, Å) | MOFid extracts organic-fragment geometries from a 2 × 2 × 2 supercell. A selected fragment is optimized in the gas phase with GFN2-xTB, then its optimized and in-framework geometries are aligned and compared. | How much the selected organic fragment changes geometry when released from the framework. It provides a geometric measure of deformation rather than a deformation energy. |
| **Organic-fragment HOMO** (`Linker_HOMO_[eV]`, eV) | Standardized neutral organic fragments are calculated with GFN2-xTB. The model uses the highest HOMO energy across the complete fragment set, requiring a converged value for every fragment. | An electronic descriptor of the organic components. The aggregation selects the fragment with the highest occupied orbital energy; it is not the HOMO of the periodic framework. |
| **Synthetic accessibility score** (`sascore`) | RDKit SA_Score is calculated for each standardized neutral organic fragment. The largest score across the complete fragment set is used. | A molecular synthetic-accessibility descriptor emphasizing the fragment with the highest estimated synthetic difficulty. It describes the organic components rather than an explicit MOF synthesis route. |
| **Energy above hull** (`Ehull`, eV/atom) | UMA-s-1p1 with the ODAC task head relaxes the framework and calculates its energy. The energy is evaluated against the fixed QMOF DFT reference hull using the pipeline's energy convention. | An energetic descriptor of the framework relative to competing reference phases, complementing the molecular and solvent-removal descriptors. |

For RMSD, the pipeline first looks for a fragment matching the parent identity with the highest SA score. If there is no exact match, it permits the implemented stereo-compatible match for a stereo-unspecified target; otherwise it falls back to the largest fragment selected by Open Babel. The output records which selection was used. The SA maximum and HOMO maximum are taken independently and can come from different organic fragments.

## How are the properties combined?

The five inputs are passed in the order shown above to the pretrained **VPU + MLP** model. VPU is the positive–unlabeled learning approach used to train the model; the MLP has one hidden layer with 128 units. At prediction time, the tool applies the saved training preprocessing and model weights to produce the final score.

This combines several aspects of a candidate MOF in one prediction: accessibility of its organic components, their electronic properties and geometric deformation, and stability of the assembled framework. No single descriptor is used as a pass/fail rule. Use the final score to prioritize candidates evaluated with the same pipeline for further investigation, rather than as a numerical experimental success rate.

```text
Input CIF
   ├─ MOFid → organic fragments → SA score and GFN2-xTB HOMO
   ├─ Fragment geometry → gas-phase optimization → RMSD
   ├─ Zeo++ + molSimplify → MOFSimplify solvent-removal score
   └─ UMA framework relaxation → QMOF reference hull → Ehull
                              ↓
                 Five-input pretrained VPU + MLP
                              ↓
                Synthesizability score and property results
```

The default Ehull relaxation checks atomic forces, cell stress, generalized forces and energy stability together before accepting a result. Individual fragment calculations and framework calculations are saved alongside the final prediction, so you can inspect the inputs behind a score.

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
