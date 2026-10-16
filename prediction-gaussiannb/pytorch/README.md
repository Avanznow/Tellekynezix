# GaussianNB Naive Bayes - PyTorch Implementation

## Overview

This folder contains a Gaussian Naive Bayes (GaussianNB) classifier implementation using PyTorch for brain wave (EEG) data classification to predict drone commands. This is a custom PyTorch implementation that uses the Gaussian Probability Density Function and Bayes' theorem for classification.

### Files

- **gaussiannb_model.py**: Contains the `GaussianNB` class - a custom `torch.nn.Module` implementation
- **gaussiannb_train.ipynb**: Jupyter notebook for training the model with EEG data
- **gaussiannb_trained.pth**: Trained model file (generated after training)
- **README.md**: This documentation file

### Improved chain (October 2026): files and live use

The notebook `GaussianNB_EEG_Classification_Notebook (Copy 1).ipynb` trains the improved chain on `Data_clean`
(window covariance features + per-participant standardisation + a train-only LDA projection in front of the
GaussianNB). Recording-grouped window accuracy goes from 26 % to 49 % (6 commands, chance 16.7 %) and per-recording
accuracy from 27 % to 63 %. The notebook writes three artifacts (cell 13) that the live path loads unchanged:

| File | Written by | Content |
|---|---|---|
| `torch_gaussiannb_improved_features.pth` | notebook cell 13 | `GaussianNB` state (6 classes x 5 projected features), `label_encoder_classes`, per-class `class_thresholds`, `feature_names` (570), `load_config` (board, channels, filtering), `projection`, test metrics |
| `torch_gaussiannb_preprocessing.joblib` | notebook cell 13 | sklearn `Pipeline([("lda", LinearDiscriminantAnalysis(n_components=5, solver="eigen", shrinkage="auto"))])`, fitted on the training windows only |
| `torch_gaussiannb_participant_scaler.joblib` | notebook cell 13 | `improved_gnb_features.ParticipantScaler`: global feature mean / std and the 18 training participants' own statistics |
| `../improved_gnb_features.py` | module | `extract_features(windows)` (570 features), `feature_names`, `ParticipantScaler`, `participant_of` |
| `../eeg_data.py` | module | the verified loader; its `_filter` (mean removal, zero-phase 1-40 Hz band-pass, 60 Hz notch) is the filtering the live path applies |
| `improved_gnb_inference.py` | module | **the live path**: `ImprovedGaussianNBChain` (below), a replay CLI |
| `../tests/test_improved_gnb_inference.py` | test | replays a `Data_clean` recording through the live path and checks windows, features and class probabilities against the notebook's feature cache and evaluation |
| `predictions_local/gnb_calibration/<participant>.npz` | the app | one file per calibrated user: feature mean / std, number of windows, feature version, source, time (`AVATAR_GNB_CALIBRATION_DIR` overrides the folder) |

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
`predict_capture(capture, participant, gate)` assesses first and returns the gate's decision dict.

**In the GUI** (`GUI5.py`, Read Brain tab): select **GaussianNB** with the **PyTorch** framework, OpenBCI, live
mode. Beside *Read my mind...* a panel shows the participant id (default `AVATAR_PARTICIPANT_ID` or `user`) and
its state (`calibrated`, `training participant` or `global fallback`):

- **Calibrate (30 s)** records `AVATAR_GNB_CALIBRATION_SECONDS` (default 30, use 30-60) seconds of unlabelled EEG
  from the headset in a worker thread, cuts it into 2 s quality blocks exactly like the loader (start-up block and
  failing blocks dropped), filters each run of passing blocks, extracts the features and stores the user's
  statistics. Sit still with eyes open; at least 2 usable windows are needed (a 30 s recording gives ~55).
  Record the kind of EEG you will produce while controlling the drone (for example cycle through the imagined
  commands), not only rest: the training statistics of each participant cover all six commands, and a
  calibration on a single state shifts every feature relative to that.
- **Calibrate from recording...** does the same from one of the user's named OpenBCI GUI `.txt` recordings.
- *Read my mind...* then captures ~5 s, keeps the last 384 samples, and runs the chain; the console log shows the
  command, the context length, the participant state and the six class probabilities.

Offline replay of a recording (per kept segment: command, confidence, per-window labels):

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/improved_gnb_inference.py --file Data_clean/G03/person03/Left/OpenBCI-RAW-....txt
```

add `--participant <id> --calibrate-from <recording.txt>` to calibrate first. Test:

```bash
/home/prasndhdl/ai_env/bin/python -m pytest prediction-gaussiannb/tests/test_improved_gnb_inference.py -q
```

**TensorFlow framework.** Selecting **GaussianNB** with the **TensorFlow** framework runs the same two chains with the
TensorFlow `GaussianNBTF` (`prediction-gaussiannb/tensorflow/gaussiannb_model.py`) as the decision model, through the same
calibration, mode switch, training session, 5-window context and vote gate (`ImprovedGaussianNBChain(backend="tensorflow")`,
`SubjectSpecificGaussianNBChain(backend="tensorflow")`, backend pieces in `prediction-gaussiannb/tensorflow/subject_gnb_tf.py`).
It loads the artifacts of `prediction-gaussiannb/tensorflow/gaussiannb_train.ipynb` (`gaussiannb_tf_improved_global.pkl`,
`gaussiannb_tf_subject_specific.pkl`) and therefore uses that notebook's windows: 125 samples with a 62-sample stride and no
tail window (the PyTorch route uses 128 / 64 with a tail window), so a 5-window context is 373 samples (2.98 s) and the global
chain decides by argmax (no per-class thresholds). Because the windows differ, calibrations and user models of the two
frameworks are stored apart (`predictions_local/gnb_calibration_tf/`, `predictions_local/gnb_users_tf/`;
`AVATAR_GNB_CALIBRATION_DIR_TF` / `AVATAR_GNB_USERS_DIR_TF` override them) and each chain ignores the other's user folders
(`meta.json` records `framework`, `window_size` and `window_stride`). The stored user model arrays are framework-free, and the
two backends give the same decisions on the same windows (`tests/test_subject_gnb_tf.py`). CLI: add `--backend tensorflow`
to `improved_gnb_inference.py` or `subject_gnb_inference.py`, e.g.

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/subject_gnb_inference.py --backend tensorflow replay --file Data_clean/G03/person03/Left/OpenBCI-RAW-....txt
```

Caveat (see the notebook's *Improved GaussianNB pipeline* section): `Data_clean` recorded each command as one
contiguous block per person, so held-out participants stay at chance for every chain. The improvement is
within-participant separability; the chain is meant for a user with their own training / calibration data, and a
calibration only adapts the feature scale, not the class model.

### Subject-specific chain (October 2026): the user's own model, trained in the GUI

The notebook's section *Subject-specific GaussianNB* fits the projection **and** the GaussianNB per participant on
that participant's own labelled windows: 84 % window accuracy and 90 % per recording on the notebook's split, against
49 % / 63 % for the global chain above (6 commands, chance 16.7 %). The price is that every user needs their own
labelled recordings, so the GUI gets a guided training session. Because `Data_clean` recorded every command as one
contiguous block per person (the 84 % partly measures *which block of the session* a window belongs to), the session
cues the commands **interleaved**: every round presents every command once in a fresh random order, with rest gaps.
That is also what makes a user's model mean the command rather than the state of the session.

| File | Content |
|---|---|
| `../subject_gnb.py` | `SubjectSpecificGaussianNB`: per-participant LDA projection (<= 5 dims, Ledoit-Wolf) + `GaussianNB`, global fallback; `add_participant(Z, y, id)` fits one user |
| `../improved_gnb_features.py` | `extract_subject_features(W_filt, W_raw, W_gamma)` (435 features), `bandpass_segment` (30-45 Hz + notch), `ParticipantScaler` |
| `torch_gaussiannb_subject_models.joblib`, `torch_gaussiannb_subject_scaler.joblib` | notebook cell 18: the 17 `Data_clean` participants' models + the global fallback, and their statistics; loaded read-only (training participants, fallback for users without a model) |
| `gaussiannb_subject_learning_curve.csv`, `gaussiannb_subject_specific_per_participant.csv`, `gaussiannb_subject_context.csv` | notebook cells 18-20: learning curve, per-participant accuracy, accuracy vs context length |
| `subject_gnb_inference.py` | **the live path and the training flow**: `SubjectSpecificGaussianNBChain`, `TrainingPlan` / `build_schedule`, `TrainingSession`, `BrainFlowStream`, a CLI |
| `../tests/test_subject_gnb_live.py` | pushes one `Data_clean` participant through the training flow and the live path and reproduces the notebook's per-participant model and accuracy (below) |
| `predictions_local/gnb_users/<id>/` | the app's per-user store (`AVATAR_GNB_USERS_DIR` overrides): `training_windows.npz` (every labelled window: 435 features, label, cue id, session id), `scaler.npz` (the user's feature mean / std), `model.npz` (projection, classes, GaussianNB priors / means / variances), `meta.json` (sessions, windows and cues per command, expectation, self-check), `sessions/<session>/cues.npz + cues.json` (the raw EEG of every cue) |

**The live path** (`SubjectSpecificGaussianNBChain`), on a raw capture `(channels, samples)` in uV:

1. the trained channels `EXG Channel 0-14` (BrainFlow rows `EEG_1..16` map by position);
2. three aligned views of the same samples, each filtered as one segment: the loader's filtering
   (`eeg_data._filter`: mean removal, 1-40 Hz band-pass, 60 Hz notch), mean removal only (the loader with
   `bandpass_hz=None, notch_hz=None`), and `bandpass_segment` on the raw view (30-45 Hz + notch);
3. the same 128-sample / 64-stride windows cut from every view -> `extract_subject_features` -> 435 features
   (log-covariance of the filtered window, of its first difference, of the 30-45 Hz raw window, multitaper band powers);
4. z-score with the **user's own** statistics (the mean / std of their labelled windows);
5. the **user's** model: LDA projection -> `GaussianNB` log-posteriors per window;
6. causal context: sum of the log-posteriors of the last 5 windows (384 samples, 3.07 s) -> argmax = the decision;
7. **vote-agreement gate**: the per-window decisions of the last 5 windows must agree with the context decision in at
   least 4 of 5 cases (the notebook's gate: coverage 0.81, accuracy of accepted decisions 0.90 on `Data_clean`);
   otherwise the decision is `uncertain` and the quality gate withholds it with reason `uncertain_decision: k/5 ...`.

`predict_capture(capture, participant, gate)` runs inside the app's `QualityGatedPredictor` like the global chain:
only accepted raw captures reach step 1, synthetic sources never produce a command. A participant without an own model
uses the notebook's per-participant model (the 17 `Data_clean` participants) or the global fallback.

**The training flow** (`TrainingSession` with a `TrainingPlan`): a 3 s settle, then for every cue of the interleaved
schedule a rest gap and the cue. The EEG of each cue is read from one continuous BrainFlow session
(`BrainFlowStream`), cut into 2 s blocks exactly like the loader (the first block dropped - reaction time after the
cue - failing blocks dropped, runs split on packet loss), every kept run gives the three views, windows and features,
and the windows are labelled with the cue. At the end (or when stopped early, with what was collected) the user's
statistics and model are refitted from **all** their stored windows (sessions accumulate) and saved; an own model
needs at least 60 labelled windows and two commands. `meta.json` also holds a cue-grouped cross-validation on the
user's own data (statistics and model fitted without the held-out cues): with interleaved cues this is the honest
estimate of the user's accuracy, independent of the `Data_clean` numbers.

Learning curve from the notebook (window accuracy of the subject-specific chain when each user contributes the k
earliest recordings per command; one `Data_clean` recording is ~10 s, about 15 kept windows; the GUI counts one cue
as one recording and shows this expectation):

| Recordings per command | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| Window accuracy | 54 % | 66 % | 69 % | 75 % | 81 % | 85 % |

**In the GUI** (`GUI5.py`, Read Brain tab, GaussianNB + PyTorch, OpenBCI, live mode): beside *Read my mind...* the
GaussianNB panel has **Global model** / **My model** radios (`setGaussianNBMode`, start value `AVATAR_GNB_MODE`,
default `global`). With *My model*:

- **Train my model (15 min)** (`startGaussianNBTraining`) runs the guided session for the participant id in the
  field: 6 rounds x 6 commands x 20 s cues with 5 s rest (`AVATAR_GNB_TRAIN_ROUNDS`, `AVATAR_GNB_TRAIN_CUE_SECONDS`,
  `AVATAR_GNB_TRAIN_REST_SECONDS`). An overlay shows the phase (*Rest - next command*, *Think the command now*), the
  command in large letters, the countdown, cue *k of 36*, the round, and the quality-checked windows kept so far;
  **Stop and keep what was recorded** ends the session early. Think the cued command for the whole cue (for example
  the imagined movement you will use to fly), sit still otherwise. The console log reports every session: cues,
  labelled windows, whether the model was fitted, the learning-curve expectation and the cue-grouped self-check.
- **Train from folder...** (`trainGaussianNBFromFolder`) trains or extends the model from a folder of the user's
  labelled OpenBCI GUI recordings laid out like `Data_clean` (`<folder>/<command>/OpenBCI-RAW-*.txt`; every recording
  is one cue).
- The status text reads `<id>: my model: own model (631 windows, 6 cues/command, expect ~81 %)` or
  `... no model yet: global fallback`; the line below gives the learning-curve expectation for the configured session.
- *Read my mind...* captures ~5 s, keeps the last 384 samples and runs the user's model; the log shows the decision,
  the vote count (`4/5 window votes for 'left'`), the participant state and the six probabilities. An uncertain
  decision appears as `Prediction withheld: uncertain_decision: 3/5 window votes agree with 'left', need 4` and is
  never queued to the drone.

Offline, with `/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/subject_gnb_inference.py`:

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/subject_gnb_inference.py train-folder --participant G03/person03 --folder Data_clean/G03/person03
```

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/pytorch/subject_gnb_inference.py replay --participant G03/person03 --file Data_clean/G03/person03/Left/OpenBCI-RAW-2025-03-13_13-45-04.txt
```

`status --participant <id>` prints what is stored, `forget --participant <id>` deletes it, `--users-dir` picks the store.

**Test** (`../tests/test_subject_gnb_live.py`, skipped without `Data_clean`, the notebook's subject feature cache and
artifacts):

```bash
/home/prasndhdl/ai_env/bin/python -m pytest prediction-gaussiannb/tests/test_subject_gnb_live.py -q
```

It reproduces the notebook's split (cell 7), pushes `G03/person03`'s 44 training recordings through the training flow
(one cue per recording) and checks that the three views and 435 features equal the notebook's feature cache window
for window (max abs diff 0.0), that the user's statistics and model equal the notebook's saved entries for that
participant (statistics exactly, projection to 1e-15), and that the 16 test recordings through the live path give the
per-participant accuracy of `gaussiannb_subject_specific_per_participant.csv` (0.8857 on 210 windows, the same
per-window decisions as the notebook's artifacts). It also covers the interleaved schedule, the vote gate on gated
384-sample captures (accepted and `uncertain_decision`), persistence and fallback, the guided session runner on a
replayed stream (windows per cue, labels, raw storage, accumulation, stop early, nothing usable) and the CLI.

Caveats: the 84 % is `Data_clean`'s number and partly reflects session state (see the notebook's *Reading the
subject-specific numbers honestly*); for a live user the cue-grouped self-check in `meta.json` / the log is the number
to trust, and it needs the interleaved session, not one block per command. The first 2 s of every cue are dropped
(the loader's rule), so a 20 s cue yields at most 9 blocks (about 34 windows); a cue whose blocks all fail the quality
checks contributes nothing and is listed in the session summary. Stored windows are tied to the feature version
`subject_logcov_dcov_gammacov_multitaper_v1`; the raw EEG of every cue is kept so a user can be re-trained if the
features change. The GUI could not be launched on this machine (no PySide6 / BrainFlow); the backend route was
exercised with stubbed Qt / BrainFlow modules and replayed `Data_clean` EEG, and the QML is compile-unchecked.

## How It Works

### Gaussian Naive Bayes Algorithm

GaussianNB is a probabilistic classifier based on Bayes' theorem with the assumption that features follow a Gaussian (normal) distribution and are independent.

**Key Steps:**

1. **Training (fit method)**:
   - Calculate mean (μ) and variance (σ²) for each feature within each class
   - Calculate class priors P(y) = count(class) / total_samples
   - Store these as non-trainable buffers (no backpropagation needed)

2. **Prediction (forward method)**:
   - For each class, calculate the Gaussian Probability Density Function:
```
     P(x|y) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
```
   - Apply Bayes' theorem:
```
     P(y|x) ∝ P(x|y) * P(y)
```
   - Use log probabilities for numerical stability
   - Return class with highest posterior probability

### PyTorch Implementation Details

- **Custom nn.Module**: Does not use traditional gradient descent or backpropagation
- **Buffers**: Stores μ, σ², and priors as registered buffers (persist with model state)
- **Tensor Operations**: Uses PyTorch operations (torch.mean, torch.var, torch.log) for GPU acceleration
- **Numerical Stability**: Uses log-probabilities to avoid numerical underflow

## Key Features

- Pure PyTorch implementation with no scikit-learn dependencies for inference
- GPU-compatible (can move model to CUDA for faster prediction on large batches)
- Stores learned parameters (means, variances, priors) in model state
- Can save/load trained models using `torch.save()` and `torch.load()`
- Includes `predict_proba()` method for probability estimates

## Usage

### Training

1. **Update data path** in `gaussiannb_train.ipynb`:
```python
   directory_path = "/path/to/your/brainwave_readings/"
```

2. **Run the notebook** to:
   - Load brain wave .txt files
   - Extract labels from filenames (backward, forward, landing, left, right, takeoff)
   - Train the GaussianNB model
   - Evaluate accuracy on test set
   - Save trained model

3. **Brain wave data format**:
   - `.txt` files with CSV format (skip first 4 header lines)
   - Filenames must contain command labels
   - Default: 32 EEG feature columns

### Inference
```python
import torch
from gaussiannb_model import GaussianNB

# Load trained model
checkpoint = torch.load('gaussiannb_trained.pth')
model = GaussianNB(
    num_features=checkpoint['num_features'],
    num_classes=checkpoint['num_classes']
)
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
X_new = torch.FloatTensor(your_new_data)  # Shape: (n_samples, n_features)
predictions = model(X_new)  # Returns class indices

# Get probabilities
probabilities = model.predict_proba(X_new)  # Shape: (n_samples, n_classes)
```

## Mathematical Background

### Bayes' Theorem
```
P(y|X) = [P(X|y) * P(y)] / P(X)
```

Where:
- `P(y|X)`: Posterior probability of class y given features X
- `P(X|y)`: Likelihood of features X given class y (Gaussian PDF)
- `P(y)`: Prior probability of class y
- `P(X)`: Evidence (constant for all classes, can be ignored for classification)

### Naive Assumption

Features are conditionally independent given the class:
```
P(X|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)
```

### Gaussian PDF

For continuous features:
```
P(xᵢ|y) = (1/√(2πσᵢ²)) * exp(-(xᵢ - μᵢ)²/(2σᵢ²))
```

## Performance Notes

- **Advantages**:
  - Fast training (just calculate statistics, no iterative optimization)
  - Fast inference
  - Works well with small datasets
  - Probabilistic predictions
  - No hyperparameters to tune

- **Limitations**:
  - Assumes features are independent (rarely true for EEG data)
  - Assumes Gaussian distribution for features
  - May underperform compared to deep learning on complex patterns

## Integration with Avatar GUI

This model can be integrated into GUI5.py alongside Random Forest and Deep Learning models. See the Avatar wiki page "Inserting a new model" for integration instructions.

## Author

**Muhammad Arsalan** - PyTorch GaussianNB Implementation for Issue #505

## References

- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- PyTorch Documentation: https://pytorch.org/docs/
- Naive Bayes Classifier: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- Gaussian Distribution: https://en.wikipedia.org/wiki/Normal_distribution
- Scikit-learn GaussianNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html