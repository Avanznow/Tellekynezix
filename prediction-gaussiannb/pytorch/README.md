# GaussianNB for EEG drone commands - PyTorch implementation

This folder holds the PyTorch Gaussian Naive Bayes that the Avatar GUI uses to turn a 16-channel OpenBCI EEG capture
into one of six drone commands: `backward`, `forward`, `landing`, `left`, `right`, `takeoff`. Three generations of the
model live here:

| Chain | What it is | Accuracy on `Data_clean` | Where |
|---|---|---|---|
| Classic `GaussianNB` module (March 2026) | the module as a plain classifier on the main notebook's hybrid window features | 26 % per window (the main notebook's baseline) | `gaussiannb_model.py`, `gaussiannb_train.ipynb` |
| Improved global chain (October 2026) | 570 window features, per-participant z-score, LDA projection, one `GaussianNB` for everyone | 49 % per window, 63 % per recording | notebook section *Improved GaussianNB pipeline*, `improved_gnb_inference.py` |
| Subject-specific chain (October 2026) | 435 window features, the user's own z-score, own LDA and own `GaussianNB`, global fallback | 84 % per window, 90 % per recording | notebook section *Subject-specific GaussianNB*, `subject_gnb_inference.py` |

Six commands, so chance is 16.7 %. Held-out participants stay at chance for every chain: read
[What the numbers mean](#what-the-numbers-mean) before quoting any of them.

The two live modules in this folder serve all three GUI frameworks. The TensorFlow and JAX folders contribute only
their decision model, training notebook and artifacts; see [Three frameworks, one live path](#three-frameworks-one-live-path).

## Contents

**Code**

| File | Role |
|---|---|
| `gaussiannb_model.py` | `GaussianNB`, a `torch.nn.Module` whose per-class means, variances and priors are registered buffers; `fit`, `forward` (class indices), `predict_proba`. The decision model of every PyTorch chain. |
| `improved_gnb_inference.py` | The live path of the improved global chain: `ImprovedGaussianNBChain`, the calibration store, a replay CLI. The `backend` argument (`torch`, `tensorflow` or `jax`) picks the decision model. |
| `subject_gnb_inference.py` | The live path and the training flow of the subject-specific chain: `SubjectSpecificGaussianNBChain`, `TrainingPlan` / `build_schedule`, `TrainingSession`, `BrainFlowStream`, the per-user store, a CLI. Same `backend` switch. |

**Notebooks**

| File | Role |
|---|---|
| `GaussianNB_EEG_Classification_Notebook.ipynb` | The main notebook, run on the repo's `Data_clean`: the verified loader and windows, the baseline, the improved chain (artifacts written by cell 13, discussion in *Improved GaussianNB pipeline: what changed and why*), the subject-specific chain (cells 16-21, artifacts written by cell 18), the defense experiments and figures, the best-effort ceiling (nested CV, repeated hold-outs), the benchmark against the project's other models, and a calibrated trial-level section (one decision per trial with per-person calibration; at chance on `Data_clean`). |
| `gaussiannb_train.ipynb` | The original 7-cell training notebook (March 2026), kept for reference. It reads `.txt` files with four header lines, takes the label from the file name and writes `gaussiannb_trained.pth`. The shared loader supersedes it, see [Data and the shared loader](#data-and-the-shared-loader). |

**Artifacts the live path loads**

| File | Written by | Content |
|---|---|---|
| `torch_gaussiannb_improved_features.pth` | main notebook, cell 13 | `GaussianNB` state (6 classes x 5 projected features), `label_encoder_classes`, per-class `class_thresholds`, `feature_names` (570), `load_config` (board, channels, filtering), `projection`, test metrics |
| `torch_gaussiannb_preprocessing.joblib` | main notebook, cell 13 | sklearn `Pipeline([("lda", LinearDiscriminantAnalysis(n_components=5, solver="eigen", shrinkage="auto"))])`, fitted on the training windows only |
| `torch_gaussiannb_participant_scaler.joblib` | main notebook, cell 13 | `improved_gnb_features.ParticipantScaler`: global feature mean / std and the 18 training participants' own statistics |
| `torch_gaussiannb_subject_models.joblib` | main notebook, cell 18 | `subject_gnb.SubjectSpecificGaussianNB`: the 17 `Data_clean` participants' own projection + `GaussianNB` and the global fallback; loaded read-only |
| `torch_gaussiannb_subject_scaler.joblib` | main notebook, cell 18 | the `ParticipantScaler` of the 435 subject features (global and per-participant statistics) |
| `gaussiannb_trained.pth` | `gaussiannb_train.ipynb` | the classic checkpoint: `model_state_dict`, `num_features`, `num_classes`, `label_encoder_classes`, train and test accuracy |

**Results written by the main notebook** (CSV and PNG)

| Group | Files |
|---|---|
| Improved chain vs baseline (cells 15, 19) | `gaussiannb_improved_vs_baseline_folds.csv`, `gaussiannb_improved_vs_baseline_summary.csv`, `gaussiannb_improved_vs_baseline.png` |
| Subject-specific chain (cells 18-20) | `gaussiannb_subject_specific_per_participant.csv`, `gaussiannb_subject_specific_cm.png`, `gaussiannb_subject_learning_curve.csv`, `gaussiannb_subject_context.csv`, `gaussiannb_subject_specific_curves.png` |
| Defense experiments (cells 23-32) | `gaussiannb_defense_comparison.csv`, `gaussiannb_per_class_metrics.csv`, `gaussiannb_normalised_cms.png`, `gaussiannb_pr_curves.png`, `gaussiannb_threshold_sweep.png`, `gaussiannb_calibration.png`, `gaussiannb_pca_feature_corr.png`, `gaussiannb_runtime_comparison.csv`, `gaussiannb_ablation_table.csv` |
| Ceiling and roadmap (cells 34-35) | `gaussiannb_nested_cv_folds.csv`, `gaussiannb_nested_cv_summary.csv`, `gaussiannb_repeated_holdout_runs.csv`, `gaussiannb_repeated_holdout_summary.csv`, `gaussiannb_ceiling_holdout_distribution.png`, `model_improvement_roadmap.csv` |
| Benchmark against the project's models (cell 36) | `final_model_benchmark_project_models.png`, `final_model_benchmark_project_models_runs.csv`, `final_model_benchmark_project_models_summary.csv` |
| Calibrated trial-level section (cell 38) | `gaussiannb_calibrated_trial_level.csv`, `gaussiannb_calibrated_trial_cm.png` |
| Legacy, from the March 2026 run and no longer written | `final_model_benchmark.png`, `final_model_benchmark_runs.csv`, `final_model_benchmark_summary.csv` |

**Shared modules one level up** (`prediction-gaussiannb/`)

| File | Role |
|---|---|
| `../eeg_data.py` | The verified loader shared by the three frameworks: `LoadConfig`, `load_recordings`, `recordings_to_frame`, `load_session`. Its `_filter` (mean removal, zero-phase 1-40 Hz band-pass, 60 Hz notch) is the filtering the live path applies. |
| `../improved_gnb_features.py` | `extract_features` (570 features) and `extract_subject_features` (435), `feature_names` / `subject_feature_names`, `ParticipantScaler`, `participant_of`, `bandpass_segment`, and the feature version strings the stores are tied to. |
| `../subject_gnb.py` | `SubjectSpecificGaussianNB` (per-participant LDA projection of at most 5 dimensions with Ledoit-Wolf shrinkage, then `GaussianNB`; global fallback; `add_participant(Z, y, id)` fits one user) and `ProjectedTorchGaussianNB`; `framework_backend` selects the PyTorch, TensorFlow or JAX decision model. |
| `../fast_preprocessing.py` | `ParallelPowerTransformer`, a multi-process drop-in for sklearn's `PowerTransformer`, used by the three notebooks' original experiments and by the JAX inference script. |

**Tests** live in `../tests/`, see [Tests](#tests).

## Data and the shared loader

`Data_clean/<group>/<person>/<command>/OpenBCI-RAW-*.txt` holds the recordings: named OpenBCI GUI exports, one command
per recording, about 10 s each. The main notebook's first cell reports what the loader keeps with the default
configuration:

| What | Count |
|---|---|
| Named OpenBCI recordings found | 1826 |
| On the 16-channel 125 Hz board (the default) | 1394 loaded, 432 skipped (the 8-channel 250 Hz recordings of `G01/person01-07`, or unreadable) |
| 2 s quality blocks kept | 3249 of 6703 (48.5 %) |
| Recordings with usable data | 883 of 1394 |
| Participants with usable data | 18 of 23 |

`eeg_data.LoadConfig` defaults: `board="16ch_125hz"`, `exclude_channels=("EXG Channel 15",)` (dead for six
participants), `quality_filter=True`, `bandpass_hz=(1.0, 40.0)`, `notch_hz=60.0`, `filter_order=4`. What the loader
fixes compared with the per-notebook loaders it replaced:

1. **Channels and sampling rate.** EEG channels are selected by header name through
   `signal_quality.acquisition.read_openbci`, which also reports the real sampling rate. The old loaders took 16
   columns after the sample index, so 8-channel recordings contributed accelerometer and digital-pin columns as EEG
   and were analysed as if sampled at 125 Hz. Headerless BrainFlow exports are skipped. One board configuration per run.
2. **Labels** come from the action folder, normalised with `signal_quality.salvage.folder_label`, so folders such as
   `TakOff`, `Take off`, `Foreward` and `backard` are labelled instead of dropped (the old path matcher kept 134 of
   303 takeoff recordings).
3. **Signal quality.** Every recording is assessed in non-overlapping 2 s blocks with the deployed `signal_quality`
   policy on the kept channels. The start-up block and every failing block are dropped; consecutive passing blocks
   with continuous packet counters form a segment.
4. **Filtering** is applied per segment (zero-phase Butterworth band-pass and mains notch), so artifacts in rejected
   blocks do not ring into kept data.

Cued sessions in the `record_session.py` format (`brainflow_raw.npy`, `events.csv`, `session.json`) are read too:
every trial becomes one labelled item from the cue to the end marker, through the same checks and filtering
(`../tests/test_eeg_data_sessions.py`). Windows must be cut within segments (`segment_id`), and data must be split by
recording or by participant, never by window.

## What the numbers mean

Every command in `Data_clean` was recorded as **one contiguous block per participant**. A split by recording therefore
tests how well the windows of one block are told from the windows of the neighbouring blocks of the same session, and
the command is confounded with recording time. The consequences, all shown in the notebook (sections *Improved
GaussianNB pipeline* and *Reading the subject-specific numbers honestly*):

- held-out participants are at chance for every chain (about 18 %);
- the 49 % and the 84 % are within-session separability. They are what a user with their own training or calibration
  data gets under this protocol, and the dataset cannot show cross-person command decoding;
- a calibration only adapts the feature scale, not the class model.

That is why the GUI's training session cues the commands **interleaved** (every round presents every command once in a
fresh random order, with rest gaps), and why the cue-grouped self-check stored with a user's model, not the
`Data_clean` number, is the accuracy to trust for a live user.

## The improved global chain

One `GaussianNB` for everyone, with features and normalisation that respect its independence assumption. From the
loader's kept segments: 128-sample windows with a 64-sample stride (plus a last window ending at the segment end),
`extract_features` (570 per window: log-Euclidean covariance of the window and of its first difference, Teager-Kaiser
energy, the notebook's hybrid band-power statistics on the raw and on the common-average-referenced channels),
`ParticipantScaler` (z-score with the participant's training statistics), a train-only LDA projection to 5 dimensions,
`GaussianNB`. Recording-grouped window accuracy went from 26 % to 49 % and per-recording accuracy from 27 % to 63 %,
selected on 5-fold recording-grouped cross-validation and confirmed on an independent partition.

**The live path** (`improved_gnb_inference.ImprovedGaussianNBChain`), on a raw capture `(channels, samples)` in uV
from the 16-channel 125 Hz board:

1. keep the trained channels `EXG Channel 0-14` (BrainFlow rows `EEG_1..EEG_16` map by position; channel 15 was
   excluded at training time);
2. the loader's filtering (`eeg_data._filter`) on the whole accepted capture as one segment;
3. 128-sample windows with a 64-sample stride (the notebook's cutting rule);
4. `extract_features` -> 570 features per window;
5. `ParticipantScaler.transform` with the current participant id: the user's own statistics after calibration,
   the training statistics for a training participant, else the global statistics;
6. the saved LDA projection -> 5 dims -> `GaussianNB` log-posteriors per window;
7. causal context: sum of the log-posteriors of the last 5 windows (384 samples, 3.07 s), normalised;
8. the notebook's threshold rule `argmax(probability / class_threshold)` -> one command.

Only captures accepted by the app's quality gate (`signal_quality.QualityGatedPredictor`) reach step 1;
`predict_capture(capture, participant, gate)` assesses first and returns the gate's decision dict. The gate also
refuses synthetic or unverified sources and stale captures.

**Calibration** stores one `predictions_local/gnb_calibration/<participant>.npz` per user (feature mean / std, number
of windows, feature version, source, time). The participant state is `calibrated`, `training participant` or
`global fallback`. Record the kind of EEG you will produce while controlling the drone, not only rest: the training
statistics of each participant cover all six commands, and a calibration on a single state shifts every feature
relative to that. Calibrations are tied to `FEATURE_VERSION = improved_logcov_dcov_tkeo_hybridlog_car_v1` and ignored
when the version differs.

## The subject-specific chain

The projection **and** the `GaussianNB` are fitted per participant on that participant's own labelled windows, because
the discriminative directions differ from person to person: 84 % window accuracy and 90 % per recording on the
notebook's split, against 49 % / 63 % for the global chain. The price is that every user needs their own labelled
recordings, so the GUI runs a guided training session (below). The notebook's learning curve, by the k earliest
recordings per command each user contributes (one `Data_clean` recording is about 10 s, about 15 kept windows; the GUI
counts one cue as one recording and shows this expectation):

| Recordings per command | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| Window accuracy | 54 % | 66 % | 69 % | 75 % | 81 % | 85 % |

**The live path** (`subject_gnb_inference.SubjectSpecificGaussianNBChain`), on a raw capture `(channels, samples)` in uV:

1. the trained channels `EXG Channel 0-14` (BrainFlow rows `EEG_1..16` map by position);
2. three aligned views of the same samples, each filtered as one segment: the loader's filtering
   (`eeg_data._filter`: mean removal, 1-40 Hz band-pass, 60 Hz notch), mean removal only (the loader with
   `bandpass_hz=None, notch_hz=None`), and `bandpass_segment` on the raw view (30-45 Hz + notch);
3. the same 128-sample / 64-stride windows cut from every view -> `extract_subject_features` -> 435 features
   (log-covariance of the filtered window, of its first difference and of the 30-45 Hz raw window, multitaper band powers);
4. z-score with the **user's own** statistics (the mean / std of their labelled windows);
5. the **user's** model: LDA projection -> `GaussianNB` log-posteriors per window;
6. causal context: sum of the log-posteriors of the last 5 windows (384 samples, 3.07 s) -> argmax = the decision;
7. **vote-agreement gate**: the per-window decisions of the last 5 windows must agree with the context decision in at
   least 4 of 5 cases (the notebook's gate: coverage 0.81, accuracy of accepted decisions 0.90 on `Data_clean`);
   otherwise the decision is `uncertain` and the quality gate withholds it with reason `uncertain_decision: k/5 ...`.

`predict_capture(capture, participant, gate)` runs inside the app's `QualityGatedPredictor` like the global chain. A
participant without an own model uses the notebook's per-participant model (the 17 `Data_clean` participants) or the
global fallback; the states are `own model`, `training participant`, `global fallback (own statistics)` and
`no model yet: global fallback`.

**The training flow** (`TrainingSession` with a `TrainingPlan`): a 3 s settle, then for every cue of the interleaved
schedule a rest gap and the cue. The EEG of each cue is read from one continuous BrainFlow session
(`BrainFlowStream`), cut into 2 s blocks exactly like the loader (the first block dropped for the reaction time after
the cue, failing blocks dropped, runs split on packet loss); every kept run gives the three views, windows and
features, labelled with the cue. At the end, or when stopped early with what was collected, the user's statistics and
model are refitted from **all** their stored windows (sessions accumulate) and saved. An own model needs at least 60
labelled windows and two commands (`MIN_WINDOWS`). `meta.json` also holds a cue-grouped cross-validation on the user's
own data (statistics and model fitted without the held-out cues): with interleaved cues this is the honest estimate of
the user's accuracy, independent of `Data_clean`.

**The per-user store**, `predictions_local/gnb_users/<id>/` (`AVATAR_GNB_USERS_DIR` overrides the folder):

| File | Content |
|---|---|
| `training_windows.npz` | every labelled window: 435 features, label, cue id, session id |
| `scaler.npz` | the user's feature mean / std |
| `model.npz` | projection, classes, `GaussianNB` priors / means / variances, as framework-free arrays |
| `meta.json` | sessions, windows and cues per command, the learning-curve expectation, the self-check, `framework`, `window_size`, `window_stride`, `feature_version` |
| `sessions/<session>/cues.npz`, `cues.json` | the raw EEG of every cue, so a user can be retrained if the features change |

Stored windows are tied to `SUBJECT_FEATURE_VERSION = subject_logcov_dcov_gammacov_multitaper_v1`. The first 2 s of
every cue are dropped (the loader's rule), so a 20 s cue yields at most 9 blocks (about 34 windows); a cue whose blocks
all fail the quality checks contributes nothing and is listed in the session summary.

## Using it in the GUI

`GUI5.py`, **Read Brain** tab: choose **GaussianNB** as the model, **PyTorch**, **TensorFlow** or **JAX** as the
framework, the **OpenBCI** source and **live** mode (the synthetic BrainFlow board never produces a command). The
GaussianNB panel beside *Read my mind...* shows the participant id (`AVATAR_PARTICIPANT_ID`, default `user`), its
state, and the **Global model** / **My model** radios (`setGaussianNBMode`; the start value is `AVATAR_GNB_MODE`,
`global` by default, `subject` for *My model*).

With **Global model** (the improved global chain):

- **Calibrate (30 s)** records `AVATAR_GNB_CALIBRATION_SECONDS` (default 30, use 30-60) seconds of unlabelled EEG
  from the headset in a worker thread, cuts it into 2 s quality blocks exactly like the loader, filters each run of
  passing blocks, extracts the features and stores the user's statistics. Sit still with eyes open and cycle through
  the imagined commands; at least 2 usable windows are needed (a 30 s recording gives about 55).
- **Calibrate from recording...** does the same from one of the user's named OpenBCI GUI `.txt` recordings.
- *Read my mind...* captures about 5 s, keeps the last 384 samples and runs the chain; the console log shows the
  command, the context length, the participant state and the six class probabilities.

With **My model** (the subject-specific chain):

- **Train my model (15 min)** (`startGaussianNBTraining`) runs the guided session for the participant id in the
  field: 6 rounds x 6 commands x 20 s cues with 5 s rest (`AVATAR_GNB_TRAIN_ROUNDS`, `AVATAR_GNB_TRAIN_CUE_SECONDS`,
  `AVATAR_GNB_TRAIN_REST_SECONDS`). An overlay shows the phase (*Rest - next command*, *Think the command now*), the
  command in large letters, the countdown, cue *k of 36*, the round and the quality-checked windows kept so far;
  **Stop and keep what was recorded** ends the session early. Think the cued command for the whole cue, for example
  the imagined movement you will use to fly, and sit still otherwise. The console log reports every session: cues,
  labelled windows, whether the model was fitted, the learning-curve expectation and the cue-grouped self-check.
- **Train from folder...** (`trainGaussianNBFromFolder`) trains or extends the model from a folder of the user's
  labelled OpenBCI GUI recordings laid out like `Data_clean` (`<folder>/<command>/OpenBCI-RAW-*.txt`; every recording
  is one cue).
- The status text reads `<id>: my model: own model (631 windows, 6 cues/command, expect ~81 %)` or
  `... no model yet: global fallback`; the line below gives the learning-curve expectation for the configured session.
- *Read my mind...* captures about 5 s, keeps the last 384 samples and runs the user's model; the log shows the
  decision, the vote count (`4/5 window votes for 'left'`), the participant state and the six probabilities. An
  uncertain decision appears as `Prediction withheld: uncertain_decision: 3/5 window votes agree with 'left', need 4`
  and is never queued to the drone.

**Environment variables**

| Variable | Default | Meaning |
|---|---|---|
| `AVATAR_PARTICIPANT_ID` | `user` | participant id shown in the panel |
| `AVATAR_GNB_MODE` | `global` | start mode: `global` or `subject` |
| `AVATAR_GNB_CALIBRATION_SECONDS` | `30` | length of *Calibrate (30 s)*, at least 10 |
| `AVATAR_GNB_CALIBRATION_DIR`, `AVATAR_GNB_CALIBRATION_DIR_TF`, `AVATAR_GNB_CALIBRATION_DIR_JAX` | `predictions_local/gnb_calibration/`, `gnb_calibration_tf/`, `gnb_calibration_jax/` | calibration store of each framework |
| `AVATAR_GNB_USERS_DIR`, `AVATAR_GNB_USERS_DIR_TF`, `AVATAR_GNB_USERS_DIR_JAX` | `predictions_local/gnb_users/`, `gnb_users_tf/`, `gnb_users_jax/` | user-model store of each framework |
| `AVATAR_GNB_TRAIN_ROUNDS` | `6` | rounds of the guided session |
| `AVATAR_GNB_TRAIN_CUE_SECONDS` | `20` | cue length in seconds |
| `AVATAR_GNB_TRAIN_REST_SECONDS` | `5` | rest between cues in seconds |

## Command line

Replay a recording through the improved global chain (per kept segment: command, confidence, per-window labels):

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/improved_gnb_inference.py --file Data_clean/G03/person03/Left/OpenBCI-RAW-....txt
```

Options: `--participant <id>` (default `<group>/<person>` from the path), `--calibrate-from <recording.txt>` to
calibrate the participant first, `--calibration-dir <folder>`, `--backend torch|tensorflow|jax`.

Train a user's model from a folder of labelled recordings, then replay one of their recordings through it:

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/subject_gnb_inference.py train-folder --participant G03/person03 --folder Data_clean/G03/person03
```

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/subject_gnb_inference.py replay --participant G03/person03 --file Data_clean/G03/person03/Left/OpenBCI-RAW-2025-03-13_13-45-04.txt
```

`status --participant <id>` prints what is stored, `forget --participant <id>` deletes it, `train-folder
--no-self-check` skips the cue-grouped cross-validation. `--users-dir <folder>` picks the store and `--backend` the
decision model; both go before the subcommand.

## Three frameworks, one live path

| Framework (GUI) | Decision model | Windows (size / stride / tail) | 5-window capture | Global decision | Calibrations, users |
|---|---|---|---|---|---|
| PyTorch | `GaussianNB` (`gaussiannb_model.py`) | 128 / 64 / yes | 384 samples (3.07 s) | per-class thresholds | `gnb_calibration/`, `gnb_users/` |
| TensorFlow | `GaussianNBTF` (`../tensorflow/gaussiannb_model.py`) | 125 / 62 / no | 373 samples (2.98 s) | argmax | `gnb_calibration_tf/`, `gnb_users_tf/` |
| JAX | `GaussianNBJAX` (`../jax/algorithm_gnb_jax.py`, CPU) | 125 / 50 / no | 325 samples (2.6 s) | argmax | `gnb_calibration_jax/`, `gnb_users_jax/` |

The TensorFlow and JAX routes run the same two chains through the same calibration, mode switch, training session,
5-window context and vote gate, with their own decision model: `ImprovedGaussianNBChain(backend=...)` and
`SubjectSpecificGaussianNBChain(backend=...)`, backend pieces in `../tensorflow/subject_gnb_tf.py` and
`../jax/subject_gnb_jax.py`, `GUI5.GNB_BACKENDS` mapping the framework name. Each route loads the artifacts of its own
training notebook and therefore inherits that notebook's window geometry. That is why calibrations and user models are
stored apart and every chain ignores the user folders of the other frameworks (`meta.json` records `framework`,
`window_size` and `window_stride`). The stored model arrays are framework-free: the three backends give the same
decisions on the same arrays, and separately fitted models agree on at least 99 % of the test windows (PyTorch uses the
unbiased variance, TensorFlow and JAX the population variance). JAX is pinned to the CPU because the model is tiny and
every new input shape costs an XLA compile on the GPU. Details: `../tensorflow/README.md` and `../jax/README.md`.

## Tests

All tests live in `../tests/` and run from the repository root:

```bash
/home/prasndhdl/ai_env/bin/python -m pytest prediction-gaussiannb/tests -q
```

The data-dependent tests skip when `Data_clean`, the notebook's feature caches (`predictions_local/feature_cache/`)
or the artifacts are missing. The GUI test runs with stubbed Qt and BrainFlow modules, so it needs neither PySide6 nor
a headset.

| Test | What it checks |
|---|---|
| `test_improved_gnb_inference.py` | the window cutting rule; a `Data_clean` recording replayed through the improved live path reproduces the notebook's windows, features and class probabilities; the live capture path equals the loader path on the same samples; the quality-gated capture gives the same command; channel mapping from BrainFlow and OpenBCI names; calibration registers a new user and unknown users fall back; the CLI |
| `test_subject_gnb_live.py` | the interleaved schedule; the learning-curve expectation; the vote gate; the three views and 435 features equal the notebook's cache window for window; the training flow on `G03/person03`'s 44 training recordings reproduces the notebook's participant model (statistics exactly, projection to 1e-15); the 16 test recordings through the live path give the row of `gaussiannb_subject_specific_per_participant.csv` (0.8857 on 210 windows); gated captures, persistence, fallback and `forget`; the guided session on a replayed stream; the CLI |
| `test_subject_gnb_tf.py` | the TensorFlow backend: no tail window, the notebook's geometry and artifacts, the per-participant accuracy and the training flow reproduced, agreement with PyTorch on the same model and on separately fitted models, separate user stores, the global chain replays, calibrates and decides by argmax |
| `test_subject_gnb_jax.py` | the JAX backend: the same checks plus the CPU pin with one model class in the process, and the whole notebook split through the global chain (0.5253 on 2810 windows) |
| `test_gui_gaussiannb_routes.py` | `GUI5.BrainwavesBackend`'s GaussianNB routes for the three frameworks: predictions with the global and training-participant models, calibration stored per framework, the guided training session with the selected framework |
| `test_eeg_data_sessions.py` | `eeg_data` reads cued sessions in the `record_session.py` format: one cue-locked item per trial, failing blocks dropped, unusable sessions skipped, irregular packet counters refused |

## Defense figures and tables

`../defense/make_defense_figures.py` recomputes, from the artifacts above and the notebook's feature caches, the material
for presenting the chains: confusion matrices and per-class metrics of both chains, the per-participant bar chart, the
vote-gate operating points, a confound-control table, the framework equivalence and latency tables, and the latency of
one live capture per chain and framework. Outputs and their descriptions are in `../defense/README.md`.

## The classic implementation (March 2026)

`gaussiannb_model.GaussianNB` is a `torch.nn.Module` that does not use gradient descent. `fit(X, y)` computes the mean
and variance of every feature within every class and the class priors `P(y) = count(class) / total`, and stores them
as registered buffers so they persist with `state_dict()`. `forward(X)` evaluates the Gaussian log-likelihood of every
class, adds the log prior and returns the class with the highest posterior; `predict_proba` returns the normalised
posteriors. Everything is computed in log space for numerical stability with plain tensor operations, so the model can
run on the GPU for large batches, and `torch.save` / `torch.load` persist it.

The original notebook `gaussiannb_train.ipynb` reads every `.txt` under `directory_path` with
`pd.read_csv(..., header=4)`, takes the label from the file name (`backward`, `forward`, `landing`, `left`, `right`,
`takeoff`; the misspelling `fowward` is mapped to `forward`), encodes it with a `LabelEncoder`, trains on the raw
columns and writes `gaussiannb_trained.pth`:

```python
import torch
from gaussiannb_model import GaussianNB

checkpoint = torch.load("gaussiannb_trained.pth")
model = GaussianNB(num_features=checkpoint["num_features"], num_classes=checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state_dict"])

X_new = torch.FloatTensor(your_new_data)       # (n_samples, n_features)
predictions = model(X_new)                     # class indices
probabilities = model.predict_proba(X_new)     # (n_samples, n_classes)
```

This notebook takes 16 columns by position and labels by file name, the two things the shared loader fixed, so it is
kept as the record of the first implementation, not as the way to retrain.

**Why Gaussian Naive Bayes.** Bayes' theorem gives `P(y | x) ∝ P(x | y) P(y)`. The naive assumption factorises the
likelihood over features, `P(x | y) = Π_i P(x_i | y)`, and each factor is a Gaussian `N(x_i; μ_iy, σ²_iy)`. Training
is one pass over the data (no iterations, no hyperparameters beyond variance smoothing), inference is a few vector
operations, and the posteriors are usable probabilities. The assumptions are also the limitation: EEG features are
correlated and not Gaussian, which is what the improved chains address with covariance features, per-participant
standardisation and a decorrelating projection in front of the model.

## Authors

- **Muhammad Arsalan** - PyTorch GaussianNB implementation (issue #505)
- **Prashant Dhaubhadel** - performance improvements: the verified loader, the improved global and subject-specific
  chains, the live path and the GUI integration

## References

- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- PyTorch documentation: https://pytorch.org/docs/
- Naive Bayes classifier: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- Gaussian distribution: https://en.wikipedia.org/wiki/Normal_distribution
- scikit-learn `GaussianNB`: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
