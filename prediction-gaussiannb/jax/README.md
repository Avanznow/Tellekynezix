# GaussianNB for EEG drone commands - JAX implementation

This folder provides the JAX decision model (`GaussianNBJAX`), its training notebook and the artifacts the GUI's
**JAX** route loads. The live path, the GUI route, the calibration and the guided training session are shared with the
PyTorch folder (`../pytorch/improved_gnb_inference.py` and `../pytorch/subject_gnb_inference.py` with
`backend="jax"`) and are documented in `../pytorch/README.md`. This file covers what is specific to JAX: the model, the
notebook, why it runs on the CPU, and the checks.

## Contents

| File | Role |
|---|---|
| `algorithm_gnb_jax.py` | `GaussianNBJAX` (`fit`, `predict`, `predict_proba`, `predict_log_proba`; one `var_smoothing` parameter) and the `save_model` / `load_model` pickle helpers. |
| `instructions_log_gnb_jax.ipynb` | The training notebook (11 cells). Cell 0 pins JAX to the CPU; cell 1 loads the repo's `Data_clean` through the shared loader and splits the recording files 60 / 20 / 20 into training, validation and test; cells 2-5 run the original GaussianNB experiments (PSD features, quantile and power transforms, PCA) and an RBF-SVM baseline; cell 7 trains the two ported chains and writes their pickles, cell 8 the per-participant table and the protocol comparison; cell 10 saves the best classic model as `gaussiannb_jax_model.pkl`. Windows are 125 samples with a 50-sample stride. |
| `previnstructions_log_gnb_jax.ipynb` | The previous version of the notebook (September 2026), kept for reference. |
| `gaussiannb_jax_improved_global.pkl` | The improved global chain: 570 features, participant z-score, LDA(5), `GaussianNBJAX`. Loaded by `ImprovedGaussianNBChain(backend="jax")`. |
| `gaussiannb_jax_subject_specific.pkl` | The subject-specific chain: 435 features, the 17 training participants' own LDA + `GaussianNBJAX`, global fallback. Loaded by `SubjectSpecificGaussianNBChain(backend="jax")`. |
| `gaussiannb_jax_protocol_comparison.csv`, `gaussiannb_jax_subject_per_participant.csv` | The notebook's numbers the live path is checked against (below). |
| `gaussiannb_jax_model.pkl` | The best classic experiment (PSD features, quantile transform, PCA): `model`, `meta` (pipeline name, `label_names`, window geometry, bands, gaussianizer, scaler, transformer, PCA, `load_config`, the notebook's `results_df`) and `format`. Written by `save_model`, read with `load_model`. |
| `trained_classifier_gnb_jax.pkl` | The first JAX model (February 2026): raw statistics (`theta_`, `var_`, `class_prior_`), `index_to_label`, and metadata of its 55-feature engineering. Kept for `inference_gaussiannb.py`, see the known issue. |
| `inference_gaussiannb.py` | The original inference script (`python inference_gaussiannb.py <model.pkl> <data>`): `standardize`, `predict_on_data`. |
| `subject_gnb_jax.py` | The JAX backend of the live path: `ProjectedGaussianNBJAX` (same interface as the PyTorch and TensorFlow ones, `to_arrays` / `from_arrays` for the framework-free `model.npz`), a pickle resolver for the notebook's `algorithm_gnb_jax.GaussianNBJAX`, `load_jax_pickle`, `subject_models_from_pickle`, and the CPU pin `pin_jax_to_cpu`. |
| `README.md` | This file. |

Tests: `../tests/test_subject_gnb_jax.py` and the JAX case of `../tests/test_gui_gaussiannb_routes.py`.

## The model

For each class $c$ and feature $j$, `fit` stores

- the mean $\mu_{cj} = \frac{1}{n_c} \sum_{i \in c} X_{ij}$,
- the variance $\sigma_{cj}^2 = \frac{1}{n_c} \sum_{i \in c} X_{ij}^2 - \mu_{cj}^2 + \lambda$, with $\lambda$ the
  `var_smoothing` added for stability (the population variance; PyTorch's `GaussianNB` uses the unbiased one),
- the prior $P(c)$, the class frequency in the training data.

The log-posterior of a sample $x$ is

$$\log P(c \mid x) \propto \sum_j \left[ -\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2} - \frac{1}{2}\log(2\pi\sigma_{cj}^2) \right] + \log P(c),$$

and `predict` takes the argmax. Everything is a few `jax.numpy` operations, jitted once per input shape.

## Results on `Data_clean`

Six commands, chance 16.7 %. On the notebook's test files:

| Experiment | Test window accuracy | Test file accuracy (majority vote) |
|---|---|---|
| Best classic GaussianNB experiment (PSD features, quantile transform, PCA) | 23.4 % | 26.6 % |
| Improved global chain (`GaussianNBJAX`) | 52.5 % (2810 windows) | 63.3 % |
| Subject-specific chain (`GaussianNBJAX`) | 85.9 % | 90.4 % |

Over all windows (`gaussiannb_jax_protocol_comparison.csv`):

| Protocol | Improved global chain | Subject-specific chain |
|---|---|---|
| 5-fold by recording (seed 42), window / file | 49.1 % / 60.0 % | 83.9 % / 89.6 % |
| 5-fold by participant (held-out people), window | 18.0 % | 18.3 % |

`GaussianNBJAX` reproduces, to four decimals, the chains found on the PyTorch notebook
(`../pytorch/GaussianNB_EEG_Classification_Notebook.ipynb`, which holds the ablations, the confirmation partition
and the audit), so the implementation is not the bottleneck. No global model of any kind exceeds about 56 % on these
features, the notebook's RBF-SVM included. 17 of the 18 participants have an own model; `G03/person03`'s 16 test
recordings give 0.834286 on 175 windows (`gaussiannb_jax_subject_per_participant.csv`).

**Read the numbers with the caveat** in `../pytorch/README.md`, *What the numbers mean*: every command in `Data_clean`
is one contiguous block per participant, so the split by recording measures within-session separability and held-out
participants stay at chance for every chain. The notebook's closing section *What the subject-specific number means
here* says the same for this notebook's split.

## CPU, not GPU

Both the notebook (`JAX_PLATFORMS=cpu` in its first cell) and `subject_gnb_jax.py` (`JAX_PLATFORMS=cpu` before the
import, then `jax.config.update("jax_platforms", "cpu")`) force JAX onto the CPU on purpose. A GaussianNB with 6
classes and a few hundred features is tiny; what costs time is the XLA compile that every new input shape triggers,
plus a device copy per call. Measured on this machine (RTX 3070 Laptop, JAX 0.10.2):

| Operation | CPU | GPU |
|---|---|---|
| `fit` on 24615 x 34 | 0.52 s | 1.61 s |
| `fit` on 24615 x 435 | 0.16 s | 0.78 s |
| `fit` on 12089 x 5 | 0.36 s | 0.80 s |
| `predict` on a new shape | 36-71 ms | 131-537 ms |
| `predict` on a shape already compiled | about 1 ms | about 1 ms |
| simulated per-participant training workload | 33 s | 77 s |

JAX reads `JAX_PLATFORMS` when the `jax` package is imported, so the setting only works when it runs before anything
else imported JAX in the process. `jax.config.update("jax_platforms", "cpu")` still works after the import as long as
nothing has run; when a GPU backend was already initialised, both the notebook and the backend print a note and the
model runs there (for the notebook: restart the kernel and run the first cell first). Without the variable the default
backend on this machine is the GPU. The live path sees few shapes (5-window captures), so after the first call each
prediction is about 1 ms.

## Live use

In `GUI5.py` (Read Brain tab) select **GaussianNB** with the **JAX** framework (`GUI5.GNB_BACKENDS` maps `"JAX"` to
`backend="jax"`, `run_gaussiannb_jax`). The panel, the *Global model* / *My model* switch, calibration, the guided
interleaved training session, the 5-window context and the vote gate are the ones described in `../pytorch/README.md`;
what differs follows from this notebook's windows (`JAX_WINDOW_SIZE` / `JAX_WINDOW_STRIDE` in
`../pytorch/subject_gnb_inference.py`):

- windows of **125 samples with a 50-sample stride and no tail window**, so a 5-window live capture is 325 samples
  (2.6 s) instead of 384 (PyTorch) or 373 (TensorFlow);
- the global chain decides by argmax; there are no per-class thresholds;
- calibrations and user models live in `predictions_local/gnb_calibration_jax/<id>.npz` and
  `predictions_local/gnb_users_jax/<id>/` (`AVATAR_GNB_CALIBRATION_DIR_JAX` / `AVATAR_GNB_USERS_DIR_JAX` override
  them). Every chain ignores the user folders of the other frameworks (`meta.json` records `framework`, `window_size`
  and `window_stride`); the stored model arrays are framework-free, and the three backends give the same decisions
  when built from the same arrays.

Offline, add `--backend jax` to either CLI:

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/improved_gnb_inference.py --backend jax --file Data_clean/G03/person03/Left/OpenBCI-RAW-....txt --participant G03/person03
```

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/subject_gnb_inference.py --backend jax train-folder --participant G03/person03 --folder Data_clean/G03/person03
```

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/subject_gnb_inference.py --backend jax replay --participant G03/person03 --file Data_clean/G03/person03/Left/OpenBCI-RAW-2025-03-13_13-45-04.txt
```

## Checks

```bash
/home/prasndhdl/ai_env/bin/python -m pytest prediction-gaussiannb/tests/test_subject_gnb_jax.py -q
```

Skipped without JAX, `Data_clean` or the pickles. It checks that:

- the JAX window rule has no tail window, and the chain's geometry and artifacts are the notebook's;
- JAX runs on the CPU and there is one `GaussianNBJAX` class in the process (the pickles, the fitted user models and
  `algorithm_gnb_jax` share it);
- the notebook's split (cell 1: sorted usable recordings, `train_test_split(test_size=0.2, random_state=42)`
  stratified on the file label) is rebuilt and `G03/person03`'s 16 **test** recordings through the live path give
  **0.834286 on 175 windows**, the row of `gaussiannb_jax_subject_per_participant.csv`, with the notebook's window
  counts per kept segment;
- all 177 test recordings through the global chain give the notebook's **0.5253 on 2810 windows** (the chain cell's
  `test_win`, also stored in `gaussiannb_jax_model.pkl`'s `results_df`);
- the training flow with `backend="jax"` on that participant's 44 training and validation recordings reproduces the
  pickled subject model: the notebook scaler's statistics exactly, the projection up to sign, the same posteriors;
- the three backends agree on the decisions from the same `model.npz` arrays, and separately fitted models agree on at
  least 99 % of the test windows (PyTorch uses the unbiased variance, TensorFlow and JAX the population variance);
- users of the three backends are kept apart in one store, and the CLI with `--backend jax` reaches the JAX store;
- the global JAX chain replays with the notebook's geometry, decides by argmax, calibrates a new user into the JAX
  calibration folder only, and runs inside the quality gate on a 325-sample capture.

## Known issue

`inference_gaussiannb.py` expects the pickle to carry `state["index_to_label"]`, which only the February 2026
`trained_classifier_gnb_jax.pkl` has. The notebook's `save_model` writes `meta["label_names"]` instead, so the script
fails on `gaussiannb_jax_model.pkl`. Use the live path (`--backend jax`, above) for the current models, or
`algorithm_gnb_jax.load_model` directly:

```python
from algorithm_gnb_jax import load_model

model, meta = load_model("gaussiannb_jax_model.pkl")
print(meta["label_names"])
```
