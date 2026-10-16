"""Fail-closed prediction wrapper: assess exactly the raw input used by inference."""

import time

import numpy as np

from .assessment import QualityAssessor


COMMANDS = {"backward", "forward", "land", "left", "right", "takeoff", "up", "down"}


def withheld_result(reason, quality=None):
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
    return bool(result.get("status") == "accepted" and result.get("command_allowed") is True
                and result.get("prediction_label") in COMMANDS
                and isinstance(quality, dict) and quality.get("usable") is True
                and quality.get("continuity_checked") is True
                and isinstance(expires, (int, float)) and not isinstance(expires, bool) and np.isfinite(expires)
                and isinstance(issued, (int, float)) and not isinstance(issued, bool) and np.isfinite(issued)
                and issued <= now < expires)


class QualityGatedPredictor:
    def __init__(self, assessor=None, *, clock=time.monotonic):
        self.assessor = assessor or QualityAssessor()
        self.clock = clock
        self._good = {}
        self.last_result = withheld_result("no_assessed_input")

    def withhold(self, reason, quality=None, *, stream_id=None):
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
        if (isinstance(captured, bool) or not isinstance(captured, (int, float))
                or not np.isfinite(captured) or not 0 <= now - captured < self.assessor.config.max_prediction_age_seconds):
            return self.withhold("stale_or_invalid_capture_time", stream_id=stream_id)
        try:
            samples = np.asarray(raw, dtype=np.float64).copy()
            samples.flags.writeable = False
            report = self.assessor.assess(samples, sfreq=sfreq, channels=channels, units=units,
                     counters=counters, timestamps=timestamps, counter_step=counter_step,
                     counter_modulus=counter_modulus, timestamp_kind=timestamp_kind)
        except (ValueError, TypeError) as error:
            return self.withhold(f"invalid_acquisition_contract: {error}", stream_id=stream_id)
        if not report["usable"] or not report["continuity_checked"]:
            return self.withhold("; ".join(report["reasons"]) or "continuity_unverified", report, stream_id=stream_id)
        last = self._good.get(stream_id)
        # Reusing the same capture must not advance a recovery counter.
        if last is not None and captured <= last["captured"]:
            return self.withhold("repeated_or_reversed_capture", report, stream_id=stream_id)
        count = 1 if last is None else last["count"] + 1
        self._good[stream_id] = dict(count=count, captured=captured)
        if count < self.assessor.config.recovery_good_windows:
            self.last_result = withheld_result("awaiting_quality_recovery", report)
            return self.last_result
        try:
            prediction = predict(samples)
            label = prediction.get("prediction_label") if isinstance(prediction, dict) else prediction
            label = "land" if label == "landing" else label
            if not isinstance(label, str) or label not in COMMANDS:
                return self.withhold("missing_or_unsupported_prediction", report, stream_id=stream_id)
        except Exception as error:
            return self.withhold(f"inference_failed: {type(error).__name__}: {error}", report, stream_id=stream_id)
        result = dict(status="accepted", command_allowed=True, prediction_label=label,
                      prediction_count=1, quality=report, issued_monotonic=captured,
                      expires_monotonic=captured + self.assessor.config.max_prediction_age_seconds,
                      confidence=prediction.get("confidence") if isinstance(prediction, dict) else None)
        if not may_execute(result, now=self.clock()):
            return self.withhold("input_expired_during_inference", report, stream_id=stream_id)
        self.last_result = result
        return result
