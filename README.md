# Siamese Network Few-Shot Learning

Repository: [github.com/mehdy06x/siamese-network](https://github.com/mehdy06x/siamese-network)

This project evaluates Siamese neural networks for few-shot classification of three signal classes:

- `normal`
- `irf`
- `orf`

It compares a normalized 1D signal representation with Gramian Angular Field image representations (GASF and GADF), including 512x512 image experiments.

## Project Layout

```text
data/                 Raw per-signal CSV files, grouped by class
datasets/raw/          Combined raw signal arrays
datasets/normalized/   Per-signal min-max normalized arrays
datasets/images/       GASF/GADF image datasets
datasets/images_512/   512x512 GASF/GADF image datasets
models/                Saved Siamese model checkpoints
results/               Splits, metrics, predictions, and figures
src/                   Reusable models, losses, FSL, image, GAF, and evaluation code
notebooks/             Data preparation, training, evaluation, and plotting notebooks
```

## Environment

Use the workspace virtual environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

The notebooks use Python packages including NumPy, pandas, PyTorch, scikit-learn, SciPy, Pillow, Matplotlib, and `pyts`.

Install missing packages in the active environment with:

```powershell
python -m pip install numpy pandas torch scikit-learn scipy pillow matplotlib pyts jupyter
```

The exact PyTorch package may need to match the available CPU or CUDA runtime.

## Reproduction Order

Run notebooks in this order from the `notebooks/` directory, or open them in VS Code and run cells from top to bottom.

1. `01_fsl_data_audit.ipynb` - inspect raw signal files and class sizes.
2. `02_fsl_split.ipynb` - create the fixed train/validation/test split.
3. `03_fsl_normalization.ipynb` - normalize each signal to `[0, 1]`.
4. `00_test_pyts_gaf.ipynb` - verify GASF/GADF generation with `pyts`.
5. `04_gaf_resolution_comparison.ipynb` - compare GAF resolutions.
6. `04_gaf-fsl.ipynb` - generate the standard GASF/GADF image datasets.
7. `05_fsl_conv1d.ipynb` - train/evaluate the 1D Siamese model.
8. `06_fsl_gasf.ipynb` - train/evaluate the GASF image model.
9. `07_fsl_gadf.ipynb` - train/evaluate the GADF image model.
10. `09_fsl_gasf_512.ipynb` - train the 512x512 GASF model.
11. `10_fsl_gadf_512.ipynb` - train the 512x512 GADF model.
12. `11_fsl_512_evaluation.ipynb` - evaluate both 512x512 models and export tables/figures.
13. `08_final_evaluation.ipynb` - load saved results and produce final comparisons.

Some notebooks contain exploratory or duplicated cells. Execute the main pipeline cells in order and avoid rerunning training cells unless retraining is intended.

## Data and Labels

There are 100 signals per class, for 300 total signals. Each original signal contains 100,000 samples. The fixed split is stratified:

- Train: 210 samples, 70 per class
- Validation: 45 samples, 15 per class
- Test: 45 samples, 15 per class

The split is stored in `results/fsl_split.npz`.

## Model Design

`src/models.py` defines shared-weight Siamese encoders:

- `SiameseNetwork`: Conv1D encoder for normalized signals.
- `Siamese2D`: Conv2D encoder for GASF/GADF images.
- Both produce 128-dimensional embeddings by default.

`src/losses.py` implements contrastive loss. Positive pairs are pulled together and negative pairs are pushed apart by a margin.

`src/fsl.py` creates N-way K-shot episodes, extracts embeddings, computes class prototypes, and classifies queries by Euclidean distance.

`src/image_fsl.py` provides image loading, batched image embeddings, image prototypes, and repeated episodic evaluation.

`src/evaluation.py` converts prototype distances to probabilities and calculates accuracy, macro F1, confusion matrices, and multiclass ROC-AUC.

## Existing Artifacts

Saved checkpoints include:

- `models/fsl_conv1d_best.pt`
- `models/fsl_gasf_best.pt`
- `models/fsl_gadf_best.pt`
- `models/fsl_gasf_512_best.pt`
- `models/fsl_gadf_512_best.pt`

Saved evaluation outputs include JSON summaries, CSV metric tables, NPZ prediction files, and figures under `results/`.

The saved comparison currently reports normalized 1D as the strongest representation, while GADF generally performs better than GASF among the image representations. The exact values are available in `results/fsl_512_final_metrics.csv`, `results/fsl_512_auc_by_class.csv`, and the JSON result files.

## Verification

A lightweight project smoke check should verify:

```powershell
python -c "import numpy, pandas, torch, sklearn, scipy, PIL, matplotlib; import src.models, src.losses, src.fsl, src.gaf, src.evaluation, src.image_fsl; print('Project imports OK')"
```

It is also useful to verify that `results/fsl_split.npz`, the five model checkpoints, prediction files, and final CSV tables exist before running plotting-only notebooks.

## Notes

- Project-root discovery should use the local `Path.cwd()` search pattern used in the notebooks; do not hard-code hosted paths such as `/content/siamese-network`.
- Notebook kernels must be configured to the project `.venv`.
- Running all training notebooks from scratch can be time-consuming and may overwrite checkpoints. The committed/generated artifacts allow the evaluation notebooks to run without retraining.
