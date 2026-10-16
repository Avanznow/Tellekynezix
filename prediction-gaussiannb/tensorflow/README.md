# GaussianNB for EEG drone commands - TensorFlow implementation

This folder provides the TensorFlow decision model (`GaussianNBTF`), its training notebook and the artifacts the GUI's
**TensorFlow** route loads. The live path, the GUI route, the calibration and the guided training session are shared
with the PyTorch folder (`../pytorch/improved_gnb_inference.py` and `../pytorch/subject_gnb_inference.py` with
`backend="tensorflow"`) and are documented in `../pytorch/README.md`. This file covers what is specific to TensorFlow.

## Contents

| File | Role |
|---|---|
| `gaussiannb_model.py` | `GaussianNBTF`: a Gaussian Naive Bayes whose class statistics and log-likelihoods are computed with TensorFlow (CPU is enough); `save_model` / `load_model` pickle helpers. It uses the population variance, where the PyTorch `GaussianNB` uses the unbiased one. |
| `gaussiannb_train.ipynb` | The training notebook (26 cells), run on the repo's `Data_clean` through the shared loader `../eeg_data.py`: the original grouped-window experiments with `GaussianNBTF` (PSD features with log-relative band powers, quantile and power transforms, PCA; cell 9), the two chains ported from the PyTorch notebook (cell 11 writes the pickles, cell 12 the per-participant table and the protocol comparison), the best original model saved as `gaussiannb_trained.pth` (cell 15), and further experiment variants with the final results table (cells 16-25). |
| `prevtensorgaussiannb_train.ipynb` | The previous version of the notebook (March 2026), kept for reference. |
| `gaussiannb_tf_improved_global.pkl` | The improved global chain: `ParticipantScaler`, LDA projection, `GaussianNBTF`, label names, window geometry, loader configuration. Loaded by `ImprovedGaussianNBChain(backend="tensorflow")`. |
| `gaussiannb_tf_subject_specific.pkl` | The subject-specific chain: the training participants' own projections and `GaussianNBTF` models, the global fallback, the scaler. Loaded by `SubjectSpecificGaussianNBChain(backend="tensorflow")`. |
| `gaussiannb_tf_protocol_comparison.csv`, `gaussiannb_tf_subject_per_participant.csv` | The notebook's numbers the live path is checked against (below). |
| `gaussiannb_trained.pth` | The best original experiment, a pickle despite the extension: `model`, `meta` (pipeline name, label names, feature columns, window geometry, bands, gaussianizer, scaler statistics, `load_config`) and `format`. |
| `subject_gnb_tf.py` | The TensorFlow backend of the live path: `ProjectedGaussianNBTF` (same interface as `subject_gnb.ProjectedTorchGaussianNB`), `load_tf_pickle`, `subject_models_from_pickle`, `tf_model_module`. The pickles were written with the class name `gaussiannb_model.GaussianNBTF`, which collides with the PyTorch module of the same name; the loader resolves it. |

Tests: `../tests/test_subject_gnb_tf.py` and the TensorFlow case of `../tests/test_gui_gaussiannb_routes.py`.

## What the notebook does

The loader is the shared one (`LoadConfig` defaults: 16-channel 125 Hz board, `EXG Channel 15` excluded, quality
filter, 1-40 Hz band-pass, 60 Hz notch). Windows are **125 samples with a 62-sample stride and no tail window**, cut
inside the loader's kept 2 s blocks. Recording files are split into training, validation and test sets (20 % of the
files for test, stratified on the file label, `random_state` fixed). The two ported chains are the ones found and
audited on the PyTorch notebook:

- **improved global chain**: 570 window features -> per-participant z-score -> LDA projection -> one `GaussianNBTF`;
- **subject-specific chain**: 435 features -> per-participant z-score -> own LDA + own `GaussianNBTF` per
  participant, global fallback.

## Results on `Data_clean`

Six commands, chance 16.7 %. On the notebook's test files:

| Experiment | Test window accuracy | Test file accuracy (majority vote) |
|---|---|---|
| Best original experiment (PSD log-relative band powers, quantile transform, PCA) | 22.5 % | 24.9 % |
| Improved global chain | 51.7 % | 65.0 % |
| Subject-specific chain | 85.9 % | 89.8 % |

Over all windows (`gaussiannb_tf_protocol_comparison.csv`):

| Protocol | Improved global chain | Subject-specific chain |
|---|---|---|
| 5-fold by recording (seed 42), window / file | 49.3 % / 60.8 % | 84.0 % / 89.1 % |
| 5-fold by participant (held-out people), window | 18.5 % | 18.4 % |

17 of the 18 participants have an own model (per-participant test accuracy in
`gaussiannb_tf_subject_per_participant.csv`); the remaining one has a single command in training and uses the global
fallback. `GaussianNBTF` reproduces the PyTorch chains to four decimals on the same partitions, so the implementation
is not what limits the accuracy.

**Read the numbers with the caveat** in `../pytorch/README.md`, *What the numbers mean*: every command in `Data_clean`
is one contiguous block per participant, so the split by recording measures within-session separability and held-out
participants stay at chance for every chain. The notebook's section *What the subject-specific number means here*
says the same for this notebook's split.

## Live use

In `GUI5.py` (Read Brain tab) select **GaussianNB** with the **TensorFlow** framework. The panel, the *Global model* /
*My model* switch, calibration, the guided training session, the 5-window context and the vote gate are the ones
described in `../pytorch/README.md`; what differs follows from this notebook's windows:

- a 5-window live capture is 373 samples (2.98 s) instead of 384;
- the global chain decides by argmax; there are no per-class thresholds;
- calibrations and user models live in `predictions_local/gnb_calibration_tf/` and `predictions_local/gnb_users_tf/`
  (`AVATAR_GNB_CALIBRATION_DIR_TF` / `AVATAR_GNB_USERS_DIR_TF` override them). Every chain ignores the user folders of
  the other frameworks (`meta.json` records `framework`, `window_size` and `window_stride`); the stored model arrays
  are framework-free, and the backends give the same decisions on the same arrays.

Offline, add `--backend tensorflow` to either CLI:

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/improved_gnb_inference.py --backend tensorflow --file Data_clean/G03/person03/Left/OpenBCI-RAW-....txt
```

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/subject_gnb_inference.py --backend tensorflow replay --participant G03/person03 --file Data_clean/G03/person03/Left/OpenBCI-RAW-....txt
```

## Tests

```bash
/home/prasndhdl/ai_env/bin/python -m pytest prediction-gaussiannb/tests/test_subject_gnb_tf.py -q
```

Skipped without TensorFlow, `Data_clean` or the pickles. It checks that the TensorFlow window rule has no tail window,
that the chain's geometry and artifacts are the notebook's, that the live path gives the notebook's per-participant
accuracy and the training flow reproduces the notebook's model, that the PyTorch and TensorFlow backends agree on the
same model arrays and on separately fitted models (at least 99 % of the test windows), that users of the two backends
are kept apart in one store, and that the global chain replays, calibrates and decides by argmax.
