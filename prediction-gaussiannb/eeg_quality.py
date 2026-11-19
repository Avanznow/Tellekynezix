"""Quality handling of the GaussianNB pipeline, moved here from the signal_quality package (19 October 2026).

Everything the loader, the live path and the GUI need to turn raw OpenBCI/BrainFlow samples into assessed
windows and fail-closed decisions, in one module next to eeg_data.py:

  read_openbci, brainflow_capture, file_sha256   physical acquisition adapters (channel names from the header,
                                                 true sampling rate, hardware counters, arrival timestamps)
  folder_label                                   action-folder name -> command, or None
  QualityConfig, QualityAssessor                 the fixed per-block rules (flat, saturated, excessive amplitude or
                                                 jump, line noise, packet and timestamp continuity)
  load_default_config                            the deployed policy, quality_policy.json (same values as
                                                 signal_quality/default_config.json: verified 60 Hz mains)
  withheld_result, may_execute, QualityGatedPredictor   the fail-closed prediction gate used by every GUI route

The code is the signal_quality code verbatim (acquisition.py, salvage.folder_label, assessment.py, config.py and
gate.py), so policy ids, reports and decisions are unchanged. signal_quality keeps its offline tools (replay,
report, dashboard, salvage, mains, sensitivity, train) and its own copies of the reader and the assessor.
"""

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from scipy.signal import welch


# ----------------------------------------------------------------------------- acquisition (signal_quality/acquisition.py)

def file_sha256(path):
    """Hex SHA-256 of a recording, read in 1 MiB blocks so a long session never has to fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def brainflow_capture(data, descriptor, *, real, counter_step):
    """Pack one BrainFlow board buffer (data is [num_rows, samples]) into the keyword dict QualityGatedPredictor.predict() takes.
    descriptor is the board description; its EEG, counter and timestamp row indices are checked against the buffer
    so a descriptor that does not match the data fails here rather than inside the gate. real=False marks a synthetic board as such."""
    data = np.asarray(data)
    eeg = descriptor.get("eeg_channels")
    rows = descriptor.get("num_rows")
    if (data.ndim != 2 or data.shape[0] != rows or not isinstance(eeg, list) or not eeg
            or len(set(eeg)) != len(eeg) or any(not isinstance(i, int) or not 0 <= i < rows for i in eeg)):
        raise ValueError("Unverified BrainFlow physical channel mapping")
    counter = descriptor.get("package_num_channel")
    timestamp = descriptor.get("timestamp_channel")
    if (not isinstance(counter, int) or not isinstance(timestamp, int)
            or counter in eeg or timestamp in eeg or counter == timestamp
            or not 0 <= counter < rows or not 0 <= timestamp < rows):
        raise ValueError("Missing/overlapping BrainFlow timing/counter channels")
    return dict(raw=data[eeg].copy(), sfreq=descriptor["sampling_rate"],
                channels=[f"EEG_{i}" for i in eeg], units="uV",
                counters=data[counter].copy(), timestamps=data[timestamp].copy(),
                counter_step=counter_step, timestamp_kind="arrival",
                source_kind="real" if real else "synthetic")


def read_openbci(path):
    """Select named physical electrodes; headerless/ambiguous exports fail."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        # The OpenBCI GUI writes four %-prefixed header lines, then the column-name row, then one sample per row.
        preamble = [stream.readline().strip() for _ in range(4)]
        names = [name.strip() for name in next(csv.reader([stream.readline()]), [])]
    count = re.fullmatch(r"%Number of channels = (\d+)", preamble[1])
    rate = re.fullmatch(r"%Sample Rate = ([0-9.]+) Hz", preamble[2])
    if preamble[0] != "%OpenBCI Raw EXG Data" or count is None or rate is None:
        raise ValueError("Requires a named OpenBCI physical acquisition header")
    c, fs = int(count[1]), float(rate[1])
    expected = { (16, 125.0): "%Board = OpenBCI_GUI$BoardCytonSerialDaisy",
                 (8, 250.0): "%Board = OpenBCI_GUI$BoardCytonSerial" }
    if expected.get((c, fs)) != preamble[3]:
        raise ValueError("Unsupported/unreviewed board/channel/rate combination")
    channels = [f"EXG Channel {i}" for i in range(c)]
    required = ["Sample Index", *channels, "Timestamp"]
    if any(names.count(name) != 1 for name in required):
        raise ValueError("Ambiguous/missing EEG or timing columns")
    values = np.loadtxt(path, delimiter=",", skiprows=5, usecols=[names.index(name) for name in required], ndmin=2)
    if not len(values):
        raise ValueError("Empty recording")
    # Columns are Sample Index, the c EXG channels, Timestamp; transposed to the [channels, samples] assess() wants.
    # With the Daisy attached each exported 16-channel row combines two packets, so Sample Index advances by 2.
    return dict(raw=values[:, 1:1+c].T, sfreq=fs, channels=channels, units="uV",
                counters=values[:, 0], timestamps=values[:, -1], counter_step=2 if c == 16 else 1,
                timestamp_kind="arrival", source_kind="real")


# ----------------------------------------------------------------------------- labels (signal_quality/salvage.py)

# Folder names as typed during the recording sessions, misspellings included, mapped to their class.
ALIASES = {"backward": "backward", "backard": "backward", "forward": "forward", "foreward": "forward",
           "fowward": "forward", "landing": "landing", "land": "landing", "left": "left", "right": "right",
           "takeoff": "takeoff", "takoff": "takeoff"}


def folder_label(name):
    """Normalise an action-folder name ('B ackward', 'OpenBCISession_Left_3') to a class, or None."""
    compact = re.sub(r"[\s_-]+", "", name.lower())
    if compact.startswith("openbcisession"):
        compact = re.sub(r"\d+$", "", compact[len("openbcisession"):])
    return ALIASES.get(compact)


# ----------------------------------------------------------------------------- assessment (signal_quality/assessment.py)

VERSION = "avatar-raw-quality-v1"
PER_CHANNEL_FEATURES = ("ptp_uv", "std_uv", "max_jump_uv", "flat_fraction", "rail_fraction",
                        "delta_ratio", "theta_ratio", "alpha_ratio", "beta_ratio",
                        "high_ratio", "line50_ratio", "line60_ratio", "spectral_entropy")
# The 13 per-channel features summarised over channels (mean/max/min): the 39 inputs of the learned quality model.
FEATURE_NAMES = tuple(f"{name}_{stat}" for name in PER_CHANNEL_FEATURES for stat in ("mean", "max", "min"))


@dataclass(frozen=True)
class QualityConfig:
    """Fixed per-block acceptance rules; amplitudes are uV of the raw signal, times are seconds.
    Values are validated once at construction, so an invalid policy file fails at load time, not mid-session."""

    min_seconds: float = 2.0
    min_ptp_uv: float = 1e-6
    flat_step_uv: float = 1e-6
    max_flat_fraction: float = .98
    max_ptp_uv: float = 1000.0
    max_jump_uv: float = 500.0
    # ADS1299 full scale at gain 24 is +-4.5 V / 24 = +-187.5 mV; a sample that reaches it is a railed ADC.
    rail_uv: float = 187499.99
    max_rail_fraction: float = .01
    # None means the mains frequency is unverified: line ratios are then reported with a warning, never enforced.
    # The deployed policy verifies 60 Hz but sets max_line_ratio to 1.0, so there too it is recorded, not enforced.
    mains_hz: float | None = None
    max_line_ratio: float = .35
    max_arrival_gap_seconds: float = .5
    max_prediction_age_seconds: float = 10.0
    recovery_good_windows: int = 1

    def __post_init__(self):
        values = asdict(self)
        for name, value in values.items():
            # mains_hz may be None and recovery_good_windows is a count; both get their own checks below.
            if name in ("mains_hz", "recovery_good_windows"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("max_flat_fraction", "max_rail_fraction", "max_line_ratio"):
            if not 0 < values[name] <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if self.mains_hz is not None and (isinstance(self.mains_hz, bool) or self.mains_hz not in (50, 60)):
            raise ValueError("mains_hz must be verified 50/60, or None")
        if isinstance(self.recovery_good_windows, bool) or not isinstance(self.recovery_good_windows, int) or self.recovery_good_windows < 1:
            raise ValueError("recovery_good_windows must be a positive integer")

    @property
    def policy_id(self):
        """VERSION plus 16 hex digits of the config's SHA-256, so every report names the exact rules it used."""
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        return f"{VERSION}-{digest}"


class QualityAssessor:
    """Applies a QualityConfig to one raw EEG block; the report it returns feeds the loader, the gate and the GUI.
    An optional learned model (signal_quality.train) adds a data-driven rejection on top of the fixed rules."""

    def __init__(self, config=None, *, quality_model=None):
        self.config = config if isinstance(config, QualityConfig) else QualityConfig(**(config or {}))
        self.quality_model = quality_model
        # A learned model is only meaningful under the policy it was trained with, so a mismatch is an error.
        if quality_model is not None and quality_model.policy_id != self.config.policy_id:
            raise ValueError("Learned quality model policy differs from assessment config")

    def assess(self, raw, *, sfreq, channels, units, counters=None, timestamps=None,
               counter_step=None, counter_modulus=256, timestamp_kind="arrival", signal_stage="raw"):
        """Assess [channels,samples]; physical channel mapping is caller-supplied.

        Arrival timestamps are not treated as uniformly spaced sample clocks.
        Hardware counters catch missing packets; zero-phase filtering never runs
        here. All rows supplied to a downstream predictor are checked.
        """
        cfg = self.config
        raw = np.asarray(raw)
        if (raw.ndim != 2 or not raw.size or not np.issubdtype(raw.dtype, np.number)
                or np.iscomplexobj(raw)):
            raise ValueError("Require nonempty real raw EEG [channels,samples]")
        if (not isinstance(channels, (list, tuple)) or len(channels) != len(raw)
                or any(not isinstance(name, str) or not name.strip() for name in channels)
                or len(set(channels)) != len(channels)):
            raise ValueError("Require explicit unique physical channel names in input order")
        if isinstance(sfreq, bool) or not isinstance(sfreq, (int, float)) or not np.isfinite(sfreq) or sfreq <= 2:
            raise ValueError("Require a finite acquisition sampling rate above 2 Hz")
        if units != "uV" or signal_stage != "raw":
            raise ValueError("Quality assessment requires original raw physical EEG in uV")
        if timestamp_kind not in ("arrival", "sample"):
            raise ValueError("timestamp_kind must be arrival or sample")
        reasons, warnings = [], []
        channel_reasons = [[] for _ in channels]
        features = None
        numeric = raw.astype(np.float64, copy=False)
        # A block must hold at least min_seconds of samples (2 s = 250 samples at 125 Hz); anything shorter is rejected.
        if numeric.shape[1] < int(np.ceil(cfg.min_seconds * sfreq)):
            reasons.append("insufficient_samples")
        if not np.isfinite(numeric).all():
            reasons.append("nonfinite_eeg")
            for i in range(len(channels)):
                if not np.isfinite(numeric[i]).all():
                    channel_reasons[i].append("nonfinite_eeg")
        # No electrode reaches 1e12 uV; such values are rejected and the feature block below is skipped for them.
        numerically_safe = np.isfinite(numeric).all() and np.max(np.abs(numeric)) < 1e12
        if np.isfinite(numeric).all() and not numerically_safe:
            reasons.append("out_of_range_eeg")
            for i in range(len(channels)):
                if (np.abs(numeric[i]) >= 1e12).any():
                    channel_reasons[i].append("out_of_range_eeg")
        if counters is not None:
            # Counters wrap at counter_modulus (256 on Cyton); any step other than counter_step (lost or repeated packet) is a discontinuity.
            values = np.asarray(counters)
            if (values.shape != (raw.shape[1],) or not np.issubdtype(values.dtype, np.number)
                    or np.iscomplexobj(values) or not np.isfinite(values).all()
                    or (values != np.round(values)).any() or (values < 0).any()
                    or isinstance(counter_modulus, bool) or not isinstance(counter_modulus, int)
                    or counter_modulus <= 1 or (values >= counter_modulus).any()
                    or isinstance(counter_step, bool) or not isinstance(counter_step, int)
                    or not 0 < counter_step < counter_modulus):
                raise ValueError("Invalid hardware counter schema/values")
            if (np.mod(np.diff(values), counter_modulus) != counter_step).any():
                reasons.append("packet_discontinuity")
        else:
            warnings.append("hardware_counter_unavailable")
        if timestamps is not None:
            values = np.asarray(timestamps)
            if (values.shape != (raw.shape[1],) or not np.issubdtype(values.dtype, np.number)
                    or np.iscomplexobj(values) or not np.isfinite(values).all()):
                raise ValueError("Invalid timestamp schema/values")
            differences = np.diff(values)
            # Arrival stamps are host-clock times and several samples can land in one serial read, so equal values
            # are fine there; only a true sample clock has to strictly advance and keep the nominal 1/sfreq spacing.
            if (differences < 0).any() or (timestamp_kind == "sample" and (differences == 0).any()):
                reasons.append("nonmonotonic_time")
            if not (differences > 0).any():
                reasons.append("nonadvancing_time")
            if (differences > cfg.max_arrival_gap_seconds).any():
                reasons.append("acquisition_gap")
            if timestamp_kind == "sample" and not np.allclose(differences, 1 / sfreq, rtol=.2, atol=1e-6):
                reasons.append("sample_clock_rate_mismatch")
        else:
            warnings.append("timestamps_unavailable")
        if cfg.mains_hz is None:
            warnings.append("mains_frequency_unverified_line_ratios_informational")
        elif cfg.mains_hz + 1 >= sfreq / 2:
            # The +-1 Hz line band has to sit below Nyquist, or the line ratio measures nothing.
            reasons.append("mains_band_not_resolvable")
        if numerically_safe and raw.shape[1] >= 2:
            jumps = np.abs(np.diff(numeric, axis=1))
            ptp = np.ptp(numeric, axis=1)
            std = numeric.std(axis=1)
            flat = (jumps <= cfg.flat_step_uv).mean(axis=1)
            rail = (np.abs(numeric) >= cfg.rail_uv).mean(axis=1)
            # Welch with 2 s segments (shorter only if the block is) gives 0.5 Hz bins, so each 2 Hz line band (49-51, 59-61) covers four bins.
            f, psd = welch(numeric, fs=sfreq, window="hann", nperseg=min(raw.shape[1], int(round(2 * sfreq))),
                           detrend="constant", axis=-1, scaling="density")
            # Power below 1 Hz (slow drift) is left out of the denominator of every ratio.
            usable_band = (f >= 1) & (f <= sfreq / 2)
            total = psd[:, usable_band].sum(axis=1)
            def ratio(low, high):
                band = (f >= low) & (f < high)
                return np.divide(psd[:, band].sum(axis=1), total, out=np.zeros(len(raw)), where=total > 0)
            p = np.divide(psd[:, usable_band], total[:, None], out=np.zeros_like(psd[:, usable_band]), where=total[:, None] > 0)
            # Entropy is normalised by log(bins) so 1 is white noise and 0 a single spike; 1e-300 keeps log finite.
            entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1) / np.log(max(p.shape[1], 2))
            # Column order must match PER_CHANNEL_FEATURES.
            matrix = np.column_stack((ptp, std, jumps.max(axis=1), flat, rail,
                         ratio(1, 4), ratio(4, 8), ratio(8, 13), ratio(13, 30), ratio(30, 40),
                         ratio(49, 51), ratio(59, 61), entropy))
            features = {name: matrix[:, i].tolist() for i, name in enumerate(PER_CHANNEL_FEATURES)}
            # A dead lead shows either as no swing at all (ptp) or as nearly every step below flat_step_uv (flat).
            for i in range(len(channels)):
                for name, bad in (("flat_channel", ptp[i] <= cfg.min_ptp_uv or flat[i] >= cfg.max_flat_fraction),
                                  ("adc_saturation", rail[i] >= cfg.max_rail_fraction),
                                  ("excessive_amplitude", ptp[i] > cfg.max_ptp_uv),
                                  ("excessive_sample_jump", jumps[i].max() > cfg.max_jump_uv)):
                    if bad:
                        channel_reasons[i].append(name)
                if cfg.mains_hz is not None:
                    line = features[f"line{int(cfg.mains_hz)}_ratio"][i]
                    if line > cfg.max_line_ratio:
                        channel_reasons[i].append("excessive_line_noise")
            for channel in channel_reasons:
                reasons.extend(channel)
        reasons = sorted(set(reasons))
        window_features = ({f"{name}_{stat}": float(function(values))
                            for name, values in features.items()
                            for stat, function in (("mean", np.mean), ("max", np.max), ("min", np.min))}
                           if features is not None else None)
        learned_probability = None
        if self.quality_model is not None and window_features is not None:
            learned_probability = self.quality_model.predict_usable(window_features)
            if learned_probability < self.quality_model.threshold:
                reasons.append("learned_quality_rejection")
        report = dict(usable=not reasons, status="acceptable" if not reasons else "unacceptable",
                      reasons=reasons, bad_channels=[name for name, flags in zip(channels, channel_reasons) if flags],
                      channel_reasons=dict(zip(channels, channel_reasons)), channel_features=features,
                      window_features=window_features, warnings=warnings,
                      quality_score=float(1 - sum(bool(flags) for flags in channel_reasons) / len(channels)) if not reasons else 0.0,
                      score_meaning="heuristic channel-pass fraction; not a calibrated probability",
                      learned_usable_probability=learned_probability,
                      quality_policy_id=cfg.policy_id, config=asdict(cfg), sfreq=float(sfreq), units=units,
                      channels=list(channels), samples=raw.shape[1], duration_seconds=raw.shape[1] / sfreq,
                      # Little-endian float64 bytes, so a block's fingerprint is the same on every platform.
                      raw_sha256=hashlib.sha256(np.ascontiguousarray(numeric, dtype="<f8").tobytes()).hexdigest(),
                      continuity_checked=counters is not None and timestamps is not None,
                      assessment_basis="fixed acquisition rules" if self.quality_model is None else "fixed rules plus independently trained quality model")
        return report


# ----------------------------------------------------------------------------- deployed policy (signal_quality/config.py)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("quality_policy.json")


def load_default_config(path=DEFAULT_CONFIG_PATH):
    """QualityConfig from quality_policy.json: verified 60 Hz mains, line ratios recorded but not rejected."""
    return QualityConfig(**json.loads(Path(path).read_text()))


# ----------------------------------------------------------------------------- fail-closed gate (signal_quality/gate.py)

COMMANDS = {"backward", "forward", "land", "left", "right", "takeoff", "up", "down"}


def withheld_result(reason, quality=None):
    """Fail-closed result: no label, command_allowed=False, plus the quality report when assessment got that far."""
    return dict(status="withheld", command_allowed=False, prediction_label=None,
                withholding_reason=str(reason), quality=quality, prediction_count=0)


def may_execute(result, *, now=None):
    """Only a fresh accepted decision may enter automatic/manual prediction execution."""
    now = time.monotonic() if now is None else now
    if not isinstance(result, dict):
        return False
    quality = result.get("quality")
    expires = result.get("expires_monotonic")
    issued = result.get("issued_monotonic")
    # bool is excluded explicitly because it is a subclass of int, so True would otherwise pass as a timestamp.
    return bool(result.get("status") == "accepted" and result.get("command_allowed") is True
                and result.get("prediction_label") in COMMANDS
                and isinstance(quality, dict) and quality.get("usable") is True
                and quality.get("continuity_checked") is True
                and isinstance(expires, (int, float)) and not isinstance(expires, bool) and np.isfinite(expires)
                and isinstance(issued, (int, float)) and not isinstance(issued, bool) and np.isfinite(issued)
                and issued <= now < expires)


class QualityGatedPredictor:
    """Wraps a model callback so a command can only leave once its block passed assess() and is still fresh.
    clock is injectable so tests and offline replay can drive the freshness rules; last_result keeps the latest decision."""

    def __init__(self, assessor=None, *, clock=time.monotonic):
        self.assessor = assessor or QualityAssessor()
        self.clock = clock
        # stream_id -> dict(count, captured): consecutive good blocks per stream, reset by any withhold on it.
        self._good = {}
        self.last_result = withheld_result("no_assessed_input")

    def withhold(self, reason, quality=None, *, stream_id=None):
        """Record a withheld result and restart the recovery count for stream_id (for every stream when None)."""
        if stream_id is None:
            self._good.clear()
        else:
            self._good.pop(stream_id, None)
        self.last_result = withheld_result(reason, quality)
        return self.last_result

    def predict(self, raw, predict, *, sfreq, channels, units, counters=None, timestamps=None,
                counter_step=None, counter_modulus=256, timestamp_kind="arrival",
                source_kind="real", stream_id="capture", captured_monotonic=None):
        """The callback receives a read-only copy of the assessed [C,T] samples.

        Any model-specific transformation happens inside this callback AFTER
        acceptance. Synthetic sources, unknown continuity, quality failure,
        inference errors and stale inputs produce no fallback command.
        """
        self.last_result = withheld_result("assessment_pending")
        if not isinstance(stream_id, str) or not stream_id.strip():
            return self.withhold("invalid_stream_id")
        if source_kind != "real":
            return self.withhold("synthetic_or_unverified_source", stream_id=stream_id)
        now = self.clock()
        captured = now if captured_monotonic is None else captured_monotonic
        # A block older than max_prediction_age_seconds, or stamped in the future, is stale however clean it looks.
        if (isinstance(captured, bool) or not isinstance(captured, (int, float))
                or not np.isfinite(captured) or not 0 <= now - captured < self.assessor.config.max_prediction_age_seconds):
            return self.withhold("stale_or_invalid_capture_time", stream_id=stream_id)
        try:
            # The callback later gets this same read-only copy: what the model sees is exactly what was assessed.
            samples = np.asarray(raw, dtype=np.float64).copy()
            samples.flags.writeable = False
            report = self.assessor.assess(samples, sfreq=sfreq, channels=channels, units=units,
                     counters=counters, timestamps=timestamps, counter_step=counter_step,
                     counter_modulus=counter_modulus, timestamp_kind=timestamp_kind)
        except (ValueError, TypeError) as error:
            return self.withhold(f"invalid_acquisition_contract: {error}", stream_id=stream_id)
        # Missing counters or timestamps are only a warning in the report; at the gate they are a refusal.
        if not report["usable"] or not report["continuity_checked"]:
            return self.withhold("; ".join(report["reasons"]) or "continuity_unverified", report, stream_id=stream_id)
        last = self._good.get(stream_id)
        # Reusing the same capture must not advance a recovery counter.
        if last is not None and captured <= last["captured"]:
            return self.withhold("repeated_or_reversed_capture", report, stream_id=stream_id)
        count = 1 if last is None else last["count"] + 1
        self._good[stream_id] = dict(count=count, captured=captured)
        # After a withhold, recovery_good_windows clean blocks in a row are needed before a command is issued.
        if count < self.assessor.config.recovery_good_windows:
            self.last_result = withheld_result("awaiting_quality_recovery", report)
            return self.last_result
        try:
            prediction = predict(samples)
            label = prediction.get("prediction_label") if isinstance(prediction, dict) else prediction
            # Recording folders say "landing"; the drone command set says "land".
            label = "land" if label == "landing" else label
            if not isinstance(label, str) or label not in COMMANDS:
                return self.withhold("missing_or_unsupported_prediction", report, stream_id=stream_id)
        except Exception as error:
            return self.withhold(f"inference_failed: {type(error).__name__}: {error}", report, stream_id=stream_id)
        result = dict(status="accepted", command_allowed=True, prediction_label=label,
                      prediction_count=1, quality=report, issued_monotonic=captured,
                      expires_monotonic=captured + self.assessor.config.max_prediction_age_seconds,
                      confidence=prediction.get("confidence") if isinstance(prediction, dict) else None)
        # Inference may itself have taken long enough for the block to expire, so re-check with a fresh clock read.
        if not may_execute(result, now=self.clock()):
            return self.withhold("input_expired_during_inference", report, stream_id=stream_id)
        self.last_result = result
        return result
