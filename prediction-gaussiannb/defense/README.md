# Defense material for the GaussianNB chains

Figures and tables for presenting the GaussianNB work: the improvement from the original model to the improved global
chain and the subject-specific chain, the controls that say what those numbers mean, and the comparison of the PyTorch,
TensorFlow and JAX implementations. Everything here is produced by one script from the saved artifacts and the
notebook's feature caches, on the notebook's own split, so nothing has to be re-run in a notebook:

```bash
/home/prasndhdl/ai_env/bin/python prediction-gaussiannb/defense/make_defense_figures.py
```

About 20 minutes on this machine. The slow parts each have a flag: `--skip-controls` (the confound controls),
`--skip-latency` (the latency tables; `--only-latency` redoes just those), `--skip-permutation` (the 100-permutation
null, about 9 minutes; `--permutations N` changes the count), `--skip-ablation` (the subject chain's ablation, about
1 minute), `--skip-learning-curve` (about 40 s), `--skip-heldout` (the held-out-participant matrix, about 30 s) and
`--skip-dataset-load` (do not reload `Data_clean` for the dataset table, 5 s). With every flag the script runs in
about 2 minutes. It needs `Data_clean` (one recording is read for the latency capture, all of it for the dataset
table), the notebook's caches in `predictions_local/feature_cache/` (`improved_windows_*.npz`, `subject_windows_*.npz`
and `hybrid_windows_*.npz`, 12089 windows of 883 recordings from 18 participants), the PyTorch artifacts in
`../pytorch/` and, for the framework parts, the TensorFlow and JAX pickles. The split is the notebook's cell 7, one
fold of a stratified group split by recording with seed 42: 9670 training and 2419 test windows (707 and 176
recordings).

These complement what the main notebook already writes in `../pytorch/`: the three-protocol bar chart
`gaussiannb_improved_vs_baseline.png`, the ablation table, the learning curve and context CSVs, the subject-specific
confusion matrix, and the benchmark against the project's other models `final_model_benchmark_project_models.png`.
The notebook's own normalised confusion matrices, precision-recall curves, threshold sweep and calibration plots
(cells 25-30) are computed on three old experiments at about 20 % accuracy, not on the deployed chains; the files
below replace them for the deployed chains.

## Outputs

### Accuracy of the two deployed chains

| File | What it shows |
|---|---|
| `defense_confusion_matrices.png` | Both chains on the 2419 held-out windows, row-normalised, count and share per true command: improved global chain 0.499, subject-specific chain 0.846. `defense_improved_global_confusion_matrix.png` is the left panel alone. |
| `defense_progression_confusion_matrices.png` | Three panels on the same 2419 windows: the original chain rebuilt from the hybrid cache with sklearn, fitted on the training windows only (150 features, Yeo-Johnson power transform, PCA to 62 dims, one GaussianNB) at 0.260, the improved global chain at 0.499, the subject-specific chain at 0.846. |
| `defense_per_recording_confusion_matrices.png` | One decision per held-out recording (argmax of the sum of its windows' log-posteriors), 176 recordings: improved global chain 0.625 (110 right), subject-specific chain 0.903 (159 right). |
| `defense_gated_confusion_matrix.png`, `defense_gated_confusion_matrix.csv` | The 5-window context decision at the deployed 4-of-5 vote gate as 6 true x 7 predicted matrices, the last column being the withheld ("uncertain") decisions, on the 1725 held-out positions with a full context. Subject-specific chain: 1453 accepted (coverage 0.84) at 0.942, 272 withheld; per command the coverage runs from 0.75 (forward, accepted at 0.82) to 0.92 (right, at 0.99). Improved global chain: 735 accepted (0.43) at 0.754. The CSV has coverage, accuracy of accepted decisions and accuracy counting withheld decisions as wrong, per command. |
| `defense_participant_confusion_matrices.png` | Small multiples: one row-normalised 6 x 6 per participant with an own model (17), ordered by accuracy from G01/person08 at 0.99 to G04/person02 at 0.53; the global fallback G03/person01 (one command in training, 30 test windows, 0.00) is left out and noted in the caption. |
| `defense_heldout_participant_confusion_matrix.png` | The subject-specific chain under the notebook's protocol C (5-fold by participant, held-out people only have the global fallback), pooled over the folds: 0.188 on 12089 windows (folds 0.189, 0.181, 0.220, 0.194, 0.161; the notebook's mean is 0.189), chance 0.167. Every row spreads over every column. |
| `defense_per_class_metrics.csv` | Precision, recall, F1 and support per command for the improved chain with the deployed threshold rule (accuracy 0.499, macro-F1 0.497), the same chain with argmax (0.495), and the subject-specific chain (0.846, macro-F1 0.843). |
| `defense_deployed_pr_roc.png`, `defense_deployed_pr_roc.csv` | One-vs-rest precision-recall and ROC curves of the posteriors per command, both deployed chains on the 2419 windows: macro average precision 0.52 against 0.92, macro ROC AUC 0.81 against 0.98; forward is the weakest command in both chains (AP 0.38 and 0.89). |
| `defense_per_participant_accuracy.png` | Window accuracy of the subject-specific chain per participant, from `../pytorch/gaussiannb_subject_specific_per_participant.csv`: 17 own models between 0.53 and 0.99, one global fallback (`G03/person01`, one command in training) at 0.00 on 15 windows; 0.846 over all test windows. |
| `defense_statistics.csv` | Wilson 95 % intervals: window accuracy 0.499 [0.479, 0.519] against 0.846 [0.831, 0.860], per-recording 0.625 [0.552, 0.693] against 0.903 [0.851, 0.939], gated 0.754 [0.721, 0.784] against 0.942 [0.929, 0.953]. Cohen's kappa 0.398 against 0.815. McNemar's exact test between the chains on the 2419 windows: 939 windows only the subject chain gets right against 99 only the improved chain gets right (p below 1e-100). Paired per-fold differences over the notebook's protocol A (5 folds by recording): improved minus original +0.237 (0.202 to 0.261), subject minus improved +0.354 (0.323 to 0.394), Wilcoxon signed-rank p = 0.0625 for both, the smallest value five folds can give. |
| `defense_itr.csv` | Information transfer rate (Wolpaw, N = 6). Single 1.02 s windows: 0.42 bits per decision, 24.6 bits/min (improved) against 1.61 bits, 94.1 bits/min (subject). The 5-window context at one decision per 3.07 s: 0.71 bits, 13.9 bits/min against 1.87 bits, 36.5 bits/min. The gated decision (accuracy of accepted decisions, bits/min scaled by coverage): 1.21 bits, 10.1 bits/min against 2.13 bits, 35.1 bits/min. |

### Why it works, and what it costs

| File | What it shows |
|---|---|
| `defense_lda_scatter.png` | The first two discriminant axes of G03/person03's own LDA on their 210 held-out windows, coloured by command (six clusters, window accuracy 0.89 for this participant), next to the same windows through the global chain's LDA (the clusters overlap, 0.51). Both from the stored projections. |
| `defense_decision_timeline.png` | One held-out recording per command of G03/person03 concatenated in time (85 windows, 43 s): stacked per-window posteriors of the subject-specific chain, the 5-window context decision with the gate's verdict (filled = accepted, hollow = withheld), the true command as a band above. 61 decisions with a full context, 46 accepted, all 46 right; the withheld ones sit where the posteriors flip (the forward recording). |
| `defense_confidence.png` | For right against wrong 5-window decisions, per chain: the margin 1 - max posterior on a log axis, and the number of the last 5 window votes that agree with the decision. The summed GaussianNB log-posteriors saturate (99 % of the subject chain's right and 89 % of its wrong decisions have a max posterior above 0.99), so the posterior does not separate right from wrong; the vote count does: 74 % of the subject chain's right decisions are unanimous against 21 % of the wrong ones (improved chain: 28 % against 10 %). This is why the gate uses votes. |
| `defense_permutation.png`, `defense_permutation.csv` | The subject-specific chain refitted per participant on 100 permutations of the training recordings' labels within participant (the confound control's scheme): null accuracy 0.162 +- 0.018 (max 0.209) against the observed 0.856 on the same code path (2389 windows of the 17 participants whose fit is possible), empirical p = 0.0099 (no permutation reaches it), also in `defense_summary.json`. |
| `defense_time_distance.png`, `defense_time_distance.csv` | Window accuracy of the subject-specific chain binned by the time between the test window and the nearest training window of the same participant (file timestamp plus window offset). Under the notebook split 2254 of the 2419 test windows lie within 30 s of a training window of their own command block, and the accuracy is flat across the range: 0.86 (< 2 s, n = 307), 0.87 (2-10 s), 0.89 (10-15 s), 0.84 (15-20 s), 0.81 (20-30 s), 0.87 (30-60 s, n = 135); the 30 windows more than a minute away belong to the fallback participant (0.00). The longer-range decay is in the confound table's temporal 60/40 split (0.677). |
| `defense_learning_curve_spread.png`, `defense_learning_curve_spread.csv` | The notebook's learning curve (each participant's training windows restricted to the k = 1, 2, 3, 4, 6, 8 earliest recordings per command, test windows unchanged) with one grey line per participant with an own model and the window-weighted mean in orange: 0.54, 0.66, 0.69, 0.75, 0.81, 0.85, and 0.85 with all recordings, identical to `../pytorch/gaussiannb_subject_learning_curve.csv`. The spread is large: at k = 1 participants range from 0.26 to 0.80, with everything from 0.53 (G04/person02) to 0.99. |
| `defense_subject_ablation.csv` | The subject-specific chain on the notebook split with each feature group removed in turn: without the 1-40 Hz log-covariance 0.812 (-0.034), without the first-difference log-covariance 0.831 (-0.015), without the 30-45 Hz log-covariance 0.832 (-0.014), without the multitaper band powers 0.834 (-0.012); and with 1 to 5 LDA dimensions: 0.549, 0.670, 0.763, 0.822, 0.846. Every group helps a little, no group is essential, and the five discriminant dimensions are all used. |
| `defense_channel_contribution.png`, `defense_channel_contribution.csv` | The absolute LDA loadings of the 17 own models summed over the five axes, each cross-channel covariance feature split equally between its two channels, normalised per participant and averaged, stacked by feature group: channels 7 and 12 carry the most (0.078 and 0.077 of the total against an equal share of 0.067), channel 10 the least (0.060), with large differences between participants (error bars). No scalp map is drawn: the channel numbers are board indices of the Cyton + Daisy, not verified electrode positions. |
| `defense_model_complexity.csv` | Parameters per stage and artifact sizes. Improved global chain: 24 582 parameters, of which 21 660 are the 18 participants' z-score statistics, 2850 the 570 x 5 projection and 72 the GaussianNB with its thresholds; 5.3 MB on disk, of which 5.1 MB is the pickled sklearn LDA object (the projection itself is 23 KB). Subject-specific chain: 3111 parameters per user (870 z-score, 2175 projection, 66 GaussianNB), 455 KB for all 18 models and the scaler; one user's model (631 windows) fits in about 300 ms. The project's RandomForest artifacts are 8.4 MB (PyTorch) and 800 KB (JAX), the JAX MLP 213 KB. |
| `defense_dataset.png`, `defense_dataset.csv` | Windows per participant and command as stacked bars (training solid, held-out test hatched), 912 windows for G03/person06 down to 64 for G03/person01. The table: 3648 files under `Data_clean`, 1826 named OpenBCI recordings, 1394 of the 16-channel board loaded, 432 skipped (422 of the 8-channel board), 3249 of 6703 two-second blocks kept (48.5 %), 883 recordings and 18 of 23 participants with usable data, 12089 windows (3 to 23 per recording, median 15), windows per command, 9670 training and 2419 test windows. |

### Controls, frameworks, latency

| File | What it shows |
|---|---|
| `defense_vote_gate.csv`, `defense_vote_gate.png` | Coverage against accuracy of accepted decisions when k of the last 5 window votes must agree with the 5-window context decision, for both chains, on the 1725 held-out positions with a full context. Deployed 4 of 5: subject-specific chain coverage 0.84, accuracy 0.94; improved global chain coverage 0.43, accuracy 0.75. |
| `defense_confound_controls.csv` | Six controls recomputed on the cache (table below). |
| `defense_framework_equivalence.csv` | PyTorch, TensorFlow and JAX on the same 2389 test windows of the 17 participants with own models: identical decisions from the same stored arrays (max difference in class probability 4e-6), 99.87 % agreement when each backend fits its own model on the same training windows (accuracy 0.8564 against 0.8560). |
| `defense_framework_notebooks.csv` | The three notebooks' own protocol numbers (5-fold by recording and by participant) next to each framework's window geometry, read from the notebooks' CSVs and the live chains. |
| `defense_decision_model_latency.csv` | The decision models alone, on random data of 5 projected features: first and second fit, first and steady-state prediction on one 5-window batch, PyTorch on CPU and CUDA, TensorFlow on CPU, JAX on CPU and GPU. |
| `defense_latency_per_capture.csv`, `defense_latency_per_capture.png` | One live capture through each chain and framework on the CPU, median of 50 after warm-up, split into signal-to-features and features-to-decision. |
| `defense_tables.md` | Every table above as markdown, ready to paste. `defense_summary.json` holds the key numbers, including the permutation p-value. |

## Confound controls

Every command in `Data_clean` is one contiguous block of recordings per participant, so a split by recording lets the
chain use the session's drift. The controls make that visible with the same windows and the same chain; the last column
is the figure the notebook's section *Reading the subject-specific numbers honestly* quotes from the original audit.

| Control | Here | Reference | Notebook |
|---|---|---|---|
| Held-out participants, 5-fold by participant (improved / subject-specific) | 0.184 / 0.189 | chance 0.167 | 0.18 |
| Acquisition time only, no EEG: label of the nearest training window in time | 0.885 | chance 0.167 | 0.97 |
| Labels permuted across recordings within participant, 3 seeds | 0.159 | chance 0.167 | 0.15 |
| Train on the first 60 % of every command block, test on the last 40 % | 0.677 | 0.846 on the notebook split | 0.66 |
| First half against second half of the same command block, LDA + GaussianNB, leave-one-recording-out | 0.757 | chance 0.5 | 0.73 |
| Share of the subject chain's errors landing on the block adjacent in time | 0.694 | 0.339 if errors ignored time | 0.79 |

The time-only control uses the recording's file timestamp plus the window offset, a coarser clock than the audit had,
which is why it reads 0.89 rather than 0.97; the conclusion is the same. `defense_permutation.png` is the 100-permutation
version of the third row, `defense_heldout_participant_confusion_matrix.png` the confusion matrix behind the first.

## Reading notes

- The vote-gate numbers differ from the 0.81 coverage and 0.90 accuracy quoted in the README because they are computed
  here on the notebook split at positions with a full 5-window context, as the GUI decides, and the earlier figure came
  from the 5-fold protocol. Both chains use argmax votes in the gate figures and in the gated confusion matrix; the
  deployed improved chain applies per-class thresholds and has no gate of its own, which changes its window accuracy
  by less than half a point (0.499 against 0.495).
- The per-recording decision, the 5-window context and the vote gate are computed per recording, as the GUI computes
  them per capture; the first four windows of every recording have no full context and are not decisions.
- The information transfer rate counts one decision per 1.02 s window or per 3.07 s capture. Live, the context slides
  by one 0.51 s stride, but successive context decisions share four of their five windows and are not independent, so
  the per-capture rate is the honest one. For the gated decision the bits per decision use the accuracy of the accepted
  decisions and the bits per minute are scaled by the coverage.
- The posterior of a 5-window decision is the softmax of five summed GaussianNB log-posteriors and sits at 1.000 for
  most decisions, right or wrong; `defense_confidence.png` therefore plots 1 - max posterior on a log axis. The count
  of agreeing votes is the usable confidence signal, which is what the gate uses.
- The permutation test's observed accuracy (0.856) is the subject chain refitted per participant on the same code
  path as the permutations, i.e. over the 2389 windows of the 17 participants whose fit is possible; the deployed
  artifacts give 0.846 over all 2419 windows because the fallback participant's 30 windows are all wrong.
- The time-distance figure has almost no range: with a split by recording, every test window of a participant has a
  training window of the same command block within a minute, which is exactly the point of the confound table. The
  decay with distance only appears when whole later recordings are held out (the 60/40 split).
- The learning curve orders recordings by file path, as the notebook does; that order equals the timestamp order in
  all 108 participant-command blocks, so "k earliest recordings" is exact.
- The held-out-participant matrix uses the notebook's protocol C exactly: a held-out person is z-scored with the
  global statistics and decided by the global fallback. The deployed chain would first calibrate that person's own
  statistics from an unlabelled recording, which the notebook's protocol C does not do.
- Framework accuracy bars would measure the notebooks, not the implementations: the three notebooks differ in window
  geometry and split (`defense_framework_notebooks.csv`), while the implementations agree on identical windows
  (`defense_framework_equivalence.csv`). Present the equivalence table and the latency tables, not accuracy per framework.
- In the decision-model table the first fit and first prediction include compilation or warm-up: JAX compiles for about
  0.3-0.5 s on the CPU and 0.5-1 s on the GPU, CUDA warms up for about 50 ms, TensorFlow for about 40 ms. The second fit
  takes 0.4-3.3 ms everywhere, and a steady-state prediction for one capture takes 0.05 ms (PyTorch CPU), 0.1 ms (CUDA),
  0.2 ms (JAX CPU), 0.45 ms (JAX GPU) and 0.94 ms (TensorFlow CPU). The GPU is slower than the CPU for a model this
  small, which is why the JAX route is pinned to the CPU. No GPU is visible to this TensorFlow build.
- A live capture costs 3.5-5.2 ms end to end, of which 3.1-4.1 ms is filtering, windowing and features and 0.2-1.2 ms
  is the decision; the quality gate that runs before the chain is not included. The capture is 384 samples for PyTorch,
  373 for TensorFlow and 325 for JAX, following each notebook's windows.
- The chart colours are the first six slots of a colour-blind-validated categorical palette: blue for the improved
  global chain, orange for the subject-specific chain, aqua for the fallback, and the six slots in command order
  (backward, forward, landing, left, right, takeoff) wherever the commands are coloured, with marker shapes as the
  second encoding in the scatter. The confusion matrices use one sequential blue ramp.
