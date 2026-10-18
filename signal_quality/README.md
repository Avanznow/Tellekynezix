# Avatar EEG signal quality and prediction withholding

The new pipeline assesses **original physical EEG**, records explainable reasons, and withholds inference and prediction-derived commands when the capture is unreliable. It works independently of EEGNet. Start with `Avatar_Signal_Quality.ipynb`: it includes all core source modules, executed real-data plots and a real rejected-window demonstration, without connecting to hardware.

## What is implemented

| File | Responsibility |
| --- | --- |
| `assessment.py` | Raw amplitude, flatness, rail saturation, discontinuity and spectral features; configurable rules |
| `acquisition.py` | Physical-channel adapters for verified Cyton/Daisy exports and BrainFlow descriptors |
| `gate.py` | Accepted/withheld decisions, same-capture inference, recovery and expiry |
| `replay.py` | Real recording replay, provenance, features and initially empty independent-review fields |
| `train.py` | Optional independently labelled quality classifier with subject-grouped evaluation |
| `report.py` | Electrode-level dataset report (CSV + HTML) and per-recording preflight with likely causes and fixes |
| `dashboard.py`, `dashboard_template.html` | Interactive per-recording dashboard, blind review import and reviewer agreement |
| `mains.py`, `mains_template.html` | Mains-frequency determination from the recordings and line-noise rejection impact |
| `salvage.py` | Export of windows recovered by excluding dead electrodes, with masks, montage options and a loader |
| `sensitivity.py`, `sensitivity_template.html` | Threshold sensitivity sweeps, margins, verdict stability and review-based error curves |
| `review_set.py`, `review_set_template.html` | Stratified blind review sets for multi-reviewer validation of the rules |
| `gui_view.py`, `explain.py` | GUI quality panel view models (withholding indicator, channel grid) and shared plain-language reasons |
| `default_config.json`, `config.py` | Deployed policy (verified 60 Hz mains; line ratios recorded, not rejected), loaded by the GUI gate via `load_default_config()` |
| `tests/test_quality.py` | Fault, provenance, gating, training and actual GUI/client method tests |

Assessment happens **before** filtering, resampling or standardization. Those transformations can hide acquisition faults. Features include channel peak-to-peak amplitude, standard deviation, largest sample jump, flat-step fraction, rail fraction, relative band powers, 50/60 Hz indicators and spectral entropy. Channel feature means/maxima/minima provide 39 window features across 8- and 16-channel recordings. Spectral ratios are indicators, not definitive eye/muscle artifact diagnoses.

A quality score describes how many channels passed the configured rules; it is **not** model confidence or a calibrated usability probability. The optional learned score is also not claimed calibrated. Passing the gate cannot establish the participant's intention or fix missing task labels/cues.

## Run

From the project root, use an environment containing `requirements.txt`; notebook/tests additionally require `requirements-dev.txt`.

```bash
python -m signal_quality.replay \
  --raw-root /home/prasndhdl/Downloads/Data_clean \
  --out signal_quality/data/new_real_replay \
  --config signal_quality/default_config.json
python -m pytest signal_quality/tests -q
```

Outputs must be new directories outside the raw source tree. Original files are never rewritten. The supported named TXT formats declare exactly 8-channel Cyton at 250 Hz or 16-channel Cyton Daisy at 125 Hz; all physical channels, counters and timestamps are selected by verified header names. Headerless and unknown schemas are reported as unassessed, rather than inventing a mapping. Cyton/Daisy counter steps are respectively 1/2 modulo 256. Arrival timestamps can repeat within a batched transfer; reversed timestamps, an entirely stationary clock or excessive gaps reject the capture. Arrival timestamps are not treated as a uniformly sampled device clock.

## Real-data result

`data/real_replay_v3` is the current replay, made with the deployed `default_config.json` (policy `avatar-raw-quality-v1-955cc6c1f48fa7c8`: 60 Hz mains verified by `mains.py`, `max_line_ratio` 1.0). Its decisions, reasons, raw hashes and bad channels are identical to `real_replay_v2` (policy `…c3cdcab9bb16f145`, mains unset) for all 8,786 windows; only the policy ID and the mains warning differ. `real_replay_v2` is kept because the notebook verifies against it; `real_replay_v1` is the earlier development replay.

Current outputs from `real_replay_v3`: `quality_report_v2`, `review_sets/set_v2`, `sensitivity_v3`, `salvaged_v2`, `mains_v2`, `dashboard_examples`. Superseded (old policy, same results): `quality_report_v1`, `review_sets/set_v1`, `sensitivity_v1`, `mains_v1`. `sensitivity_v2` is a run on `real_replay_v2` with `reviews/pd.jsonl`.

- 3,648 files inventoried; 1,816 supported named recordings assessed.
- 1,821 unsupported exports and 11 invalid/unsupported named files reported separately.
- 8,786 complete, nonoverlapping two-second windows: 3,381 pass, 5,405 fail (**61.52% rejected by this policy**).
- Overlapping reasons: saturation 3,812; flat channels 3,514; excessive amplitude 2,738; sample jumps 2,204; packet discontinuities 2,031; acquisition gaps 6.
- 168,060 trailing samples reported but not scored as complete offline windows. Live assessment checks every supplied sample.
- Feature assessment latency on this CPU: median 0.83 ms, p95 1.08 ms. This excludes acquisition, model inference, networking and command execution.

There are **zero independently reviewed quality labels**. The rejection rate is not artifact-detection accuracy, and a rule flag alone does not establish its physical cause. Large differences between participants warrant electrode/wiring/gain and protocol review. No new task labels or cue times were invented. Source-file, raw-window, configuration and implementation hashes are saved for reproducibility.

## Electrode-level report and recording preflight

`report.py` turns rule flags into **what to fix**. It re-assesses every replayed window from its source file with the replay's own policy, refuses to continue if any decision differs from the published replay or a source changed, and adds which electrodes failed and why.

```bash
# Dataset report: participants.csv, recordings.csv, channels.csv, findings.json, report.html
python -m signal_quality.report dataset \
  --replay signal_quality/data/real_replay_v3 --out signal_quality/data/new_quality_report
# Check new recordings during a collection session (exit code 1 if any is not good)
python -m signal_quality.report preflight path/to/OpenBCI-RAW-*.txt
```

Findings name likely causes and checks: a persistently flat/railed electrode (`electrode_contact`), most electrodes railing together (`reference_or_bias`), the same dead electrode across participants (`systematic_electrode`), packet loss after start-up (`wireless_dropouts`), widespread amplitude/jump flags after start-up (`movement_or_muscle`), a failing first window (`startup_transient`) and mixed board configurations (`mixed_hardware`). Thresholds are in `FindingPolicy` and are recorded with every output. Findings are heuristics over rule flags, not verified hardware diagnoses.

Results (`data/quality_report_v2` on the v3 replay; identical to `quality_report_v1`; 8,786 windows, 1,800 recordings, 30 participants):

- **Start-up transient:** the first 2 s window fails in 99.9% of recordings versus 51.6% of later windows (4.5% for mostly-good participants). Discard the first seconds of every recording.
- **Single dead electrodes condemn whole participants:** 14 participants are unusable, mostly because one to three electrodes are flat or railed for the entire session. Excluding only each participant's persistently dead electrodes would raise usable windows from **38.5% to an estimated 62.2%**; several "unusable" participants would return to 76–80%, like the good ones. Using that data requires a reduced-montage model or channel interpolation; the gate still never drops channels silently.
- **Systematic hardware fault:** EXG Channel 15 is dead for 6 G03/G04 participants and EXG Channel 4 for 3 G01 participants. Inspect those leads/pins before the next collection.
- **Wireless dropouts** after start-up for G03/person01 (74%), G04/person02 (45%) and G04/person01 (24%).
- **Mixed hardware:** G01/person01–07 used the 8-channel 250 Hz board; everyone else the 16-channel 125 Hz board.

## Interactive quality dashboard and blind review

`dashboard.py` writes one self-contained HTML file per recording (no server, hardware or BrainFlow; opens offline in any browser). Examples for a good recording, a dead-electrode recording and a packet-loss recording are in `data/dashboard_examples/`.

```bash
python -m signal_quality.dashboard build path/to/OpenBCI-RAW-*.txt --out-dir dashboards
```

| Feature | What the user sees |
| --- | --- |
| Channel quality grid | Every physical channel as pass / fail / unassessed with its reason and raw measurements |
| Raw signal viewer | The window's waveforms; failing channels tinted, rail occupancy labelled, sample jumps circled, packet discontinuities and gaps as dashed lines |
| Recording replay | Window timeline (pass / fail / unassessed tail); play at 1–8× or step with ←/→ and watch decisions change |
| Withholding indicator | The real `QualityGatedPredictor` replayed in order (including its recovery rule): "Prediction allowed" or "Prediction withheld" with plain-language reasons |
| Session report | Passing/rejected windows, rejection rate, most frequent problems, affected channels, dead electrodes, findings; JSON and CSV export |
| Review controls | Blind review (rule decisions, markers, gate and report hidden), shuffled order, A/U keys, notes; revealing a rule decision is recorded per window; review record export |

Reviews become training/evaluation data only through validation against the source:

```bash
python -m signal_quality.dashboard import review_<recording>_<reviewer>.json --out reviews/<reviewer>.jsonl
python -m signal_quality.dashboard agreement reviews/*.jsonl
```

`import` re-reads the recording, checks the source hash, policy and every window's raw hash, and writes annotations accepted by `signal_quality.train` (the review record is the hashed evidence file). `agreement` reports, per reviewer, agreement and Cohen's kappa with the rules, **false acceptance** (rules passed a window the reviewer rejected), **false rejection** (rules rejected a window the reviewer accepted) and the number of blind reviews, plus reviewer-vs-reviewer agreement on shared windows. Per-window assessment latency is shown in the session report.

The dashboard's rejection rate describes the configured checks, not detection accuracy; only independent reviews can measure that. Reviews are saved in the browser's local storage between sessions when it is available; export the review record to keep them.

## Validating the rules: stratified blind review sets

`review_set.py` turns reviewer validation into one study that two people can each finish in about an hour.

```bash
python -m signal_quality.review_set build --replay signal_quality/data/real_replay_v3 --out signal_quality/data/review_sets/set_v2 --size 300 --seed 42
python -m signal_quality.review_set import review_<set id>_<reviewer>.json --set signal_quality/data/review_sets/set_v2 --out reviews/<reviewer>.jsonl
python -m signal_quality.dashboard agreement reviews/*.jsonl
```

- **Sampling:** half rule-accepted and half rule-rejected windows (so false acceptance and false rejection can both be estimated); rejections split evenly across start-up, dead electrode, continuity and artifact types; each stratum spread across participants (leftover windows assigned at random).
- **Weights:** every window carries `sampling_weight` (windows in its stratum×participant cell ÷ windows sampled from it, scaled to the stratum total), so `agreement` also reports estimates for the whole replay, not just the sample. The manifest records any participants a stratum could not cover.
- **Blinding:** `review_set.html` contains only the raw signal, channel names and sampling rate — no rule decisions, reasons, participants or file names. Send reviewers the page only; keep `manifest.json`, which holds the rule decisions and provenance.
- **Shared definition:** the rubric is shown to reviewers and hashed into the set ID, so all reviews refer to the same definition.
- **Per window:** binary label, optional *unsure* flag and note, and time spent (review latency). Reviews survive page reloads in the browser's local storage; export the record to submit it.
- **Import** checks the set ID, every item, the label and timestamp, the source file hash and each window's raw hash, then writes annotations accepted by `signal_quality.train`.
- **Agreement** reports, per reviewer: agreement and Cohen's kappa with the rules, false acceptance and false rejection with 95% Wilson intervals, sampling-weighted estimates for the replay, the share of rule rejections the reviewer accepted **per rule reason** (which rules over-reject), the reviewer-acceptable share per stratum, unsure count and median seconds per window; plus reviewer-vs-reviewer agreement and kappa on shared windows.

`data/review_sets/set_v2` (set `rs-783b8cb434530570`, deployed policy): 300 windows from all 30 participants and 264 recordings; every participant present in each stratum is sampled. It contains exactly the same windows in the same order as the superseded `set_v1` (`rs-e99e2335c9a95582`, old policy); send reviewers `set_v2`. No set reviews have been collected yet, so no detection accuracy is claimed.

## Threshold sensitivity

`sensitivity.py` shows how the rule decisions depend on each configured limit, without re-reading raw files: every amplitude/electrode rule is an "any channel" rule, so each replayed window's channel extremes decide it under any threshold. The baseline is checked to reproduce every replay decision and reason exactly, and tests confirm the model matches re-running the assessor with other thresholds.

```bash
python -m signal_quality.sensitivity --replay signal_quality/data/real_replay_v3 --out signal_quality/data/sensitivity_vN
# with reviews, adds false acceptance / false rejection for every setting:
python -m signal_quality.sensitivity --replay signal_quality/data/real_replay_v3 --out signal_quality/data/sensitivity_vN --reviews reviews/*.jsonl
```

Outputs: `report.html` (small-multiple sweep charts, peak-to-peak × jump heatmap, margins, participant stability, data tables), `sweeps.csv`, `grid.csv`, `margins.csv`, `participants.csv`, `sensitivity.json`. Plausible ranges: peak-to-peak 500–2,000 µV, jump 250–1,000 µV, rail occupancy 0.5–5%, flat steps 90–100%.

Results (`data/sensitivity_v3` on the v3 replay, including `reviews/pd.jsonl`; the four original rules give the same figures as `sensitivity_v1`). With mains set, the line-noise limit is also modelled and swept (0.2–1.0: rejection 91.5% down to the current 61.5%):

- **The headline rate is robust:** across plausible single-threshold changes the rejection rate stays between **60.4% and 65.5%** (currently 61.5%; 51.6% after excluding start-up windows).
- **Electrode-fault rules are insensitive to their limits:** all 3,812 rail-flagged windows are at least 2× past the limit (median 100×, i.e. fully railed), and 2,847 of 3,514 flat-flagged windows are completely flat; no rail or flat setting from 0.1% to 100% changes the rejection rate. Only 6 rejections depend on the rail rule alone and none on the flat rule alone.
- **Amplitude and jump limits carry the judgement:** 268 and 311 flagged windows sit within 1.5× of those limits; stricter limits reject more (peak-to-peak 500 µV → 65.5%), looser ones accept few extra windows (jump ≥750 µV adds only 7).
- **Participant verdicts:** 24 of 30 are unchanged across all plausible ranges; G01/person03, G01/person09, G01/person10, G03/person07, G04/person01 and G04/person04 change with stricter or looser amplitude/jump limits — these are the participants whose usability claims are threshold-dependent.

Line-noise rejection is not modelled (it is off while mains is unverified). Thresholds chosen from the review curves are fitted to those reviews and must be confirmed on held-out reviews.

## Salvage export: recovering windows lost to dead electrodes

`salvage.py` turns the report's salvage estimate into a dataset. For each participant it removes the persistently dead electrodes (flat or railed in ≥80% of windows, the report's rule), **re-assesses every window with the same rules on the remaining channels**, drops start-up windows, and exports the passing windows. The export fails unless the recovered count per participant equals the quality report's estimate.

```bash
python -m signal_quality.salvage --replay signal_quality/data/real_replay_v3 --out signal_quality/data/salvaged_v2
```

```python
from signal_quality.salvage import load_salvaged
data = load_salvaged("signal_quality/data/salvaged_v2", "16ch_125hz",
                     [f"EXG Channel {i}" for i in range(15)])   # or "common" / "masked"
data["X"], data["labels"], data["participants"], data["channels"]
```

- **Samples are raw µV as recorded** (float32): no filtering, re-referencing or normalisation, so each pipeline applies its own preprocessing. Excluded channels are NaN with a `mask.npy`; never zero-filled or interpolated (electrode positions are not verified).
- **Boards are kept apart** (`16ch_125hz/`, `8ch_250hz/`) because window shapes differ. Each has `X.npy` [windows, channels, samples], `mask.npy`, and `windows.jsonl` with provenance (source and raw hashes, excluded channels, `recovered` flag, action folder and normalised folder label).
- **Participants needing more than 25% of channels removed are skipped** (`--max-excluded-fraction`). `--keep-startup` keeps first windows that pass.
- **`montage_options`** in `summary.json` gives, for every montage size, the best channel set and the windows/participants it retains; `load_salvaged(..., [channels])` loads exactly that montage.

Results (`data/salvaged_v2`, ≈85 MB; regenerated under the deployed policy with identical content to the removed `salvaged_v1`):

- Usable windows rise from **38.5% to 59.6%** of the replay: **5,235 windows exported, 1,855 of them recovered** (re-assessment matched the report estimate for every participant; the 62.2% estimate falls to 59.6% because G01/person05, with 3 of 8 channels dead, is excluded).
- 16-channel board (4,009 windows, 23 participants): all 16 channels → 2,501 windows / 14 participants; **dropping only Ch 15 → 3,062 / 18**; the common 9-channel montage → 4,009 / 23.
- 8-channel board (1,226 windows, 6 participants): 8 channels → 878 / 4; drop Ch 4 → 1,056 / 5; common 6 channels → 1,226 / 6.

The folder labels are exported as recorded and remain **confounded with recording order** (each class was recorded as one block). More usable windows do not make those labels trustworthy; evaluate any model on held-out participants.

## Mains frequency check

`mains.py` determines the power-line frequency from the recordings: for every replayed recording it computes the median spectrum over working channels (0.25 Hz resolution) and measures how far the 50 Hz and 60 Hz peaks stand above neighbouring frequencies. On the 250 Hz board the 100/120 Hz harmonics give independent evidence. It also computes, exactly, what switching line-noise rejection on would cost. It never changes the active configuration.

```bash
python -m signal_quality.mains --replay signal_quality/data/real_replay_v3 --out signal_quality/data/mains_v2
```

Outputs: `report.html` (median spectra per board, peaks by group and board, rejection impact, every recording), `mains.json`, `recordings.csv`, `suggested_config.json`.

Results (`data/mains_v1` on the old policy; `data/mains_v2` confirms the deployed setting):

- **The mains frequency is 60 Hz.** 1,543 of 1,800 recordings show a clear 60 Hz peak (median 13.7 dB above neighbouring frequencies) and 1 shows a 50 Hz peak (median 50 Hz height 1.5 dB). Every collection group and both boards agree; on the 250 Hz board the 120 Hz harmonic is visible and 100 Hz is not.
- **Line-ratio rejection should stay off for now.** At the configured `max_line_ratio` 0.35 it would reject 2,232 of the 3,381 currently usable windows (rejection 86.9%): in raw, unfiltered windows the median share of power at 60 Hz is 58%. A notch filter removes that; strong mains power alone does not make a window unusable.
- **Adopted configuration:** `default_config.json` now sets `mains_hz: 60` with `max_line_ratio: 1.0`, recording the verified frequency (line ratios become meaningful and the "mains unverified" warning disappears) without rejecting anything, since a line ratio cannot exceed 1. All outputs were regenerated from `real_replay_v3`; models whose saved quality policy differs (e.g. Riemannian / band-power artifacts) are withheld by the GUI until retrained under it.
- **Preprocessing:** apply a 60 Hz notch (120 Hz too for 250 Hz recordings), e.g. `notch_freq=60` in `prediction_eegnet`'s preprocessing.
- **Hardware:** the pickup is strong in most recordings; check the bias/reference contact, run the acquisition laptop on battery and keep electrode cables away from power cables.

## Thresholds and configuration

Defaults require at least two seconds, reject any channel with peak-to-peak amplitude above 1,000 µV, a sample jump above 500 µV, essentially flat signals, or more than 1% rail-level samples. The default rail is 187,499.99 µV and assumes the relevant OpenBCI conversion/gain; verify it against the acquisition hardware. Continuity faults reject the whole capture. No channel is silently removed, because doing so changes the classifier's input contract.

These are conservative **engineering starting points**, not validated universal EEG cutoffs. Freeze or tune the policy using an independent review/training partition and evaluate on held-out participants. Unknown mains is represented by `null`; 50/60 Hz ratios remain informative but do not cause rejection. The deployed `default_config.json` sets the verified `mains_hz: 60` with `max_line_ratio: 1.0`, so line ratios are recorded but never reject. Set verified `mains_hz` to 50 or 60 to enable the configured line-power rejection; frequencies unresolvable at the sample rate withhold rather than report a false pass. The quality module assesses line noise; task preprocessing still owns notch/bandpass filtering.

Configure a gate explicitly in Python, including for GUI deployment by supplying this assessor when constructing the backend gate:

```python
import json
from pathlib import Path
from signal_quality import QualityConfig, QualityAssessor, QualityGatedPredictor

config = QualityConfig(**json.loads(Path("signal_quality/default_config.json").read_text()))
gate = QualityGatedPredictor(QualityAssessor(config))
result = gate.predict(
    raw, predict=existing_model_callback, sfreq=fs,
    channels=physical_channel_names, units="uV",
    counters=packet_numbers, timestamps=arrival_times,
    counter_step=1, timestamp_kind="arrival", source_kind="real",
    stream_id="one-continuous-acquisition-session",
    captured_monotonic=capture_end_monotonic,
)
# result['status'] is 'accepted' or 'withheld'; withheld labels are None.
```

The callback gets a read-only copy of the exact assessed `[channels, samples]` capture. Perform classifier-specific filtering/features inside that callback. The acquisition provider must supply a trusted monotonic capture-end time; it defaults to invocation time only for synchronous use. Accepted decisions expire ten seconds after capture. A longer recovery requirement can be configured for sequential captures using a stable session ID; repeated capture times cannot advance recovery. Live adapters currently acquire separate sessions with unique IDs and the default one-window recovery. Continuity is checked **within** each supplied capture; there is no claim of cross-capture packet-boundary monitoring.

## Avatar integration

`GUI5.py` and `ReadBrain.qml` display quality status and reasons. The Read Brain tab has a quality panel driven by `BrainwavesBackend.qualityViewChanged` (built by `gui_view.py`, testable without Qt):

- **Withholding indicator:** after every *Read my mind*, "Prediction allowed" or "Prediction withheld" with the gate's reasons in plain language (hover for all of them); green/amber border. Changing model or framework resets it, so a stale decision is never shown.
- **Channel grid:** one numbered tile per physical channel, green pass / red fail / grey unassessed; hover for the reason and measurements. Decisions made before any measurement (e.g. an unsupported headset) show no invented tiles.
- **Quality dashboard...** opens a recording's interactive dashboard (above) in the default browser; files are written to the system temp folder under `avatar_quality_dashboards`.
- **Check recording...** runs the preflight on a recording file and shows it in the same panel: verdict, the estimate without dead electrodes, findings with fixes (also written to the console log), and channels judged after the start-up window (amber = flagged in under half of those windows).

The panel fits the default 1200×800 window; it was rendered off-screen with PySide6 against a stub backend and introduces no QML warnings beyond those already present in `ReadBrain.qml`. The existing RF controller assesses the real OpenBCI capture before posting that same capture to the service. The PyTorch CNN route assesses its original physical EEG before its existing StandardScaler transformation. Quality assessment is independent of the classifier architecture.

A withheld result clears the previous prediction. Synthetic input, missing continuity metadata, model errors, unsupported labels and expired input never substitute random commands. Each queued prediction carries its accepted result: the worker checks freshness and that it still matches the current decision immediately before dispatch. A new assessment or source/model/framework change invalidates older approvals. This does not cancel a hardware command already executing. Manual controls remain separate from prediction approval.

Neurosity lacks a verified raw timestamp/rate/channel contract in the current adapter and is withheld. GaussianNB lacks a verified deployment feature adapter and is withheld; its former remote-RF substitution was removed. Unimplemented TensorFlow CNN/GNB routes are withheld. Both RF framework selections retain the existing remote TensorFlow RF route. The legacy RF server's feature/width contract still needs validation against a live physical capture; the gate does not certify that classifier.

Tests do not initialize headset/drone hardware or send network requests. Live integration has not been verified. In the existing Avatar virtual environment, BrainFlow 5.14 currently fails import because `pkg_resources` is absent; repair its SDK/environment compatibility before live verification. The offline quality pipeline and notebook do not import BrainFlow.

## Independent review, learned classification and research

Select windows under a documented sampling protocol, blind reviewers to rule decisions where possible, and use an explicit quality rubric based on raw waveforms, spectra and continuity. Two reviewers and adjudication support a defensible label definition. Save their decisions in a separate review record; quality labels do not require fabricated task cues.

Copy reviewed rows to a separate JSONL file and populate:

- `reviewed_quality_label`: integer 0 (unacceptable) or 1 (acceptable).
- `label_verified`: true; `reviewed_by`: reviewer identity.
- `reviewed_at_utc`: timezone-aware ISO timestamp.
- `annotation_evidence`: `{ "path": "review_record.json", "sha256": "actual SHA256", "locator": "window-specific decision locator" }`.

Keep source/window provenance unchanged. The evidence file must be separate from the source EEG and training annotation file. Review evidence is a provenance requirement, not an automated guarantee of reviewer competence or label reliability. Do not use `rule_usable` as ground truth.

```bash
python -m signal_quality.train \
  --annotations /absolute/path/reviewed_quality.jsonl \
  --out signal_quality/data/reviewed_training_v1 \
  --config signal_quality/default_config.json
```

The fixed protocol requires both labels and at least five participants, with both labels present in every partition. Three outer subject-grouped folds reserve test participants; an inner grouped split selects logistic regression versus random forest and decision threshold by validation macro F1. Scaling is fit only on training data. Recording hashes are checked for partition overlap. Original features are recomputed; unverified labels, altered sources/evidence, incorrect participant identities and overlapping review windows are rejected.

Each fold saves its split identities, selected model, validation candidates, held-out predictions and metrics: balanced accuracy, macro F1, bad-window average precision, confusion matrix, bad-window acceptance, good-window rejection and retained coverage. No production quality model has been trained on the current unlabelled real data. Never choose a deployment fold by its test score.

After independently selecting a deployment artifact, load trusted local artifacts only:

```python
from signal_quality.train import QualityModel
model = QualityModel.load("selected_quality_model.joblib")
assessor = QualityAssessor(config, quality_model=model)
# Exact configuration-policy match is required; hard rules cannot be overridden.
```

A paper could evaluate quality-aware abstention across subjects, with reviewer agreement, subject-level confidence intervals and an independent final test set. Downstream risk-versus-coverage needs trustworthy task labels and a frozen existing classifier; it cannot currently establish six-intent accuracy from these recordings. These additional research measurements are not claimed by the current rule replay.
