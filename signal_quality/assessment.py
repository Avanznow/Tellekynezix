"""Inspect original physical EEG before filtering or normalization.

Rules flag acquisition faults and gross contamination. Scores are descriptive,
not calibrated probabilities or proof that a task can be decoded. Spectral
features are indicators rather than physiological artifact diagnoses.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy.signal import welch


VERSION = "avatar-raw-quality-v1"
PER_CHANNEL_FEATURES = ("ptp_uv", "std_uv", "max_jump_uv", "flat_fraction", "rail_fraction",
                        "delta_ratio", "theta_ratio", "alpha_ratio", "beta_ratio",
                        "high_ratio", "line50_ratio", "line60_ratio", "spectral_entropy")
FEATURE_NAMES = tuple(f"{name}_{stat}" for name in PER_CHANNEL_FEATURES for stat in ("mean", "max", "min"))


@dataclass(frozen=True)
class QualityConfig:
    min_seconds: float = 2.0
    min_ptp_uv: float = 1e-6
    flat_step_uv: float = 1e-6
    max_flat_fraction: float = .98
    max_ptp_uv: float = 1000.0
    max_jump_uv: float = 500.0
    rail_uv: float = 187499.99
    max_rail_fraction: float = .01
    mains_hz: float | None = None
    max_line_ratio: float = .35
    max_arrival_gap_seconds: float = .5
    max_prediction_age_seconds: float = 10.0
    recovery_good_windows: int = 1

    def __post_init__(self):
        values = asdict(self)
        for name, value in values.items():
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
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        return f"{VERSION}-{digest}"


class QualityAssessor:
    def __init__(self, config=None, *, quality_model=None):
        self.config = config if isinstance(config, QualityConfig) else QualityConfig(**(config or {}))
        self.quality_model = quality_model
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
        if numeric.shape[1] < int(np.ceil(cfg.min_seconds * sfreq)):
            reasons.append("insufficient_samples")
        if not np.isfinite(numeric).all():
            reasons.append("nonfinite_eeg")
            for i in range(len(channels)):
                if not np.isfinite(numeric[i]).all():
                    channel_reasons[i].append("nonfinite_eeg")
        numerically_safe = np.isfinite(numeric).all() and np.max(np.abs(numeric)) < 1e12
        if np.isfinite(numeric).all() and not numerically_safe:
            reasons.append("out_of_range_eeg")
            for i in range(len(channels)):
                if (np.abs(numeric[i]) >= 1e12).any():
                    channel_reasons[i].append("out_of_range_eeg")
        if counters is not None:
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
            reasons.append("mains_band_not_resolvable")
        if numerically_safe and raw.shape[1] >= 2:
            jumps = np.abs(np.diff(numeric, axis=1))
            ptp = np.ptp(numeric, axis=1)
            std = numeric.std(axis=1)
            flat = (jumps <= cfg.flat_step_uv).mean(axis=1)
            rail = (np.abs(numeric) >= cfg.rail_uv).mean(axis=1)
            f, psd = welch(numeric, fs=sfreq, window="hann", nperseg=min(raw.shape[1], int(round(2 * sfreq))),
                           detrend="constant", axis=-1, scaling="density")
            usable_band = (f >= 1) & (f <= sfreq / 2)
            total = psd[:, usable_band].sum(axis=1)
            def ratio(low, high):
                band = (f >= low) & (f < high)
                return np.divide(psd[:, band].sum(axis=1), total, out=np.zeros(len(raw)), where=total > 0)
            p = np.divide(psd[:, usable_band], total[:, None], out=np.zeros_like(psd[:, usable_band]), where=total[:, None] > 0)
            entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1) / np.log(max(p.shape[1], 2))
            matrix = np.column_stack((ptp, std, jumps.max(axis=1), flat, rail,
                         ratio(1, 4), ratio(4, 8), ratio(8, 13), ratio(13, 30), ratio(30, 40),
                         ratio(49, 51), ratio(59, 61), entropy))
            features = {name: matrix[:, i].tolist() for i, name in enumerate(PER_CHANNEL_FEATURES)}
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
                      raw_sha256=hashlib.sha256(np.ascontiguousarray(numeric, dtype="<f8").tobytes()).hexdigest(),
                      continuity_checked=counters is not None and timestamps is not None,
                      assessment_basis="fixed acquisition rules" if self.quality_model is None else "fixed rules plus independently trained quality model")
        return report
