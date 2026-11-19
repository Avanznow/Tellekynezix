# What each file in prediction-gaussiannb is

A map of the folder for readers of the starred paper: which files are the current pipeline, which are earlier
versions kept for reference, which are the project's original baseline files, and which are outputs. Nothing listed
as "previous version" or "original" is used by the current pipeline; they stay so that the numbers quoted from them
in the paper (the 26 % baseline, the reconciliation table) can be reproduced.

## Current pipeline (shared)

| File | Role |
|---|---|
| `eeg_data.py` | The verified loader: named OpenBCI exports, header-based channel selection, label normalisation, 2 s block screening, segments, 1–40 Hz band-pass and 60 Hz notch; `recordings_to_frame` for the notebooks |
| `eeg_quality.py`, `quality_policy.json` | The quality code the loader and the live path use (moved here from `signal_quality/` on 19 October 2026): readers, `folder_label`, the per-block rules, the deployed policy and the fail-closed prediction gate |
| `improved_gnb_features.py` | The 570 improved-chain features and the 435 subject-chain features; `ParticipantScaler` |
| `subject_gnb.py` | The subject-specific chain: one shrinkage-LDA projection and one GaussianNB per participant, with the global fallback |
| `fast_preprocessing.py` | Vectorised filtering helpers used by the loader and the features |

## Current pipeline (per framework)

| File | Role |
|---|---|
| `pytorch/gaussiannb_model.py` | The project's PyTorch GaussianNB module, modified for the chains (class priors, log-space likelihoods) |
| `pytorch/improved_gnb_inference.py` | Live path of the improved global chain: `ImprovedGaussianNBChain`, calibration store, replay |
| `pytorch/subject_gnb_inference.py` | Live path of the subject-specific chain: `SubjectSpecificGaussianNBChain`, the vote gate, `TrainingPlan` / `TrainingSession` (guided cue session), user stores |
| `tensorflow/subject_gnb_tf.py`, `jax/subject_gnb_jax.py` | The TensorFlow and JAX decision models behind the same live path (same stored arrays) |
| `pytorch/GaussianNB_EEG_Classification_Notebook.ipynb` | The main notebook: baseline, improved chain, subject chain, protocols A–C, ablations, learning curve, context sweep; writes the CSV artifacts in `pytorch/` |
| `tensorflow/gaussiannb_train.ipynb` | The current TensorFlow training notebook (both chains, run on the shared loader) |
| `jax/instructions_log_gnb_jax.ipynb` | The current JAX training notebook (both chains, CPU-pinned) |

## Defense material

| File | Role |
|---|---|
| `defense/make_defense_figures.py` | Regenerates every figure and table of the paper from the caches and artifacts (`--skip-*` flags for the slow parts) |
| `defense/defense_figures.ipynb` | The same functions cell by cell, labelled with the paper's figure and table numbers, plus the cells added for the revision (architecture diagrams, loader demonstration, block-screening table, reconciliation table) |
| `defense/reconcile_counts.py`, `defense/reconciliation_report.md` | Cross-check of the loader's counts against the signal-quality replay |
| `defense/defense_*.csv`, `defense_*.png`, `defense_summary.json`, `defense_tables.md` | Outputs of the script and the notebook |

## Previous versions, kept for reference (not used by the pipeline)

| File | What it is |
|---|---|
| `pytorch/gaussiannb_train.ipynb` | The original 7-cell PyTorch training notebook (March 2026; the project's first GaussianNB experiment, 26 % baseline). Re-run on 18 October 2026 with only the data path, loader and evaluation cells changed, for the reconciliation in Supplementary Appendix D; that re-run is uncommitted. |
| `tensorflow/prevtensorgaussiannb_train.ipynb` | The previous TensorFlow notebook (March 2026) |
| `jax/previnstructions_log_gnb_jax.ipynb` | The previous JAX notebook (September 2026) |

## Original project files (pre-existing baselines)

| File | Status |
|---|---|
| `tensorflow/gaussiannb_model.py` | The project's original TensorFlow decision model, unchanged since April 2026 |
| `jax/algorithm_gnb_jax.py` | The project's original JAX decision model, with small modifications for the chains |
| `jax/inference_gaussiannb.py` | The project's original JAX inference script (`python inference_gaussiannb.py <model.pkl> <data>`), modified to the shared arrays |
| `pytorch/*.pth`, `tensorflow/*.pth`, `jax/*.pkl` | Trained artifacts: the classic checkpoints of the original experiments and the pickles of the two chains |

## Tests

`tests/` holds the 46-case suite the paper cites (loader, features, both chains in the three frameworks, the live path on
a replayed capture, the GUI routes with stubs): `python -m pytest prediction-gaussiannb/tests -q` from the repository root.
