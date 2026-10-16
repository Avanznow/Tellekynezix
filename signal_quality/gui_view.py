"""Turn gate results and recording preflights into what the Avatar GUI displays.

Pure Python (no Qt), so the GUI's quality panel is testable without a display:
a withholding indicator (state, headline, plain-language reasons) and a channel
grid (one tile per physical channel: pass / fail / warn / unassessed + detail).
"""

from .assessment import QualityConfig
from .explain import explain

WARN_FRACTION = 0.5  # recording preflight: a channel flagged in < 50% of windows is a warning


def _short(name):
    return name.replace("EXG Channel ", "Ch ")


def _reasons(text, config):
    """Split a gate withholding reason ("a; b") into explained sentences, keeping order."""
    seen, out = set(), []
    for part in str(text or "").split(";"):
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            out.append(explain(part, config))
    return out


def quality_view(result, config=None):
    """View model for one gate decision from ``QualityGatedPredictor`` (or ``withheld_result``)."""
    config = config or QualityConfig()
    result = result if isinstance(result, dict) else {}
    quality = result.get("quality") if isinstance(result.get("quality"), dict) else None
    channels = []
    if quality is not None:
        features = quality.get("channel_features")
        for i, name in enumerate(quality.get("channels") or []):
            flags = (quality.get("channel_reasons") or {}).get(name, [])
            if features is None:
                status, detail = "unassessed", "not assessed: the capture could not be measured"
            elif flags:
                status = "fail"
                detail = "; ".join(explain(flag, config) for flag in flags)
            else:
                status = "pass"
                detail = (f"peak-to-peak {features['ptp_uv'][i]:.0f} µV, "
                          f"largest jump {features['max_jump_uv'][i]:.0f} µV")
            channels.append(dict(name=name, label=_short(name), status=status, detail=detail))
    failing = sum(c["status"] == "fail" for c in channels)
    if result.get("status") == "accepted":
        state, headline = "accepted", "Prediction allowed: EEG passed quality checks"
        reasons = []
    elif result.get("withholding_reason") in ("assessment_pending",):
        state, headline, reasons = "pending", "Assessing EEG input...", []
    elif result.get("withholding_reason") in (None, "no_assessed_input"):
        state, headline, reasons = "idle", "EEG quality: no assessed input yet", []
    else:
        state = "withheld"
        headline = "Prediction withheld: unreliable EEG input"
        reasons = _reasons(result.get("withholding_reason"), config)
    summary = (f"{len(channels) - failing}/{len(channels)} channels pass" if channels
               else "no channel measurements for this decision")
    return dict(state=state, headline=headline, reasons=reasons, channels=channels, summary=summary,
                source="live capture")


def preflight_view(result, config=None):
    """View model for ``signal_quality.report.preflight`` on a whole recording file."""
    config = config or QualityConfig()
    if "error" in result:
        return dict(state="withheld", headline="Recording could not be checked", reasons=[result["error"]],
                    channels=[], summary="", source=result.get("path", "recording"))
    channels = []
    for channel in result["channels"]:
        # Judge electrodes after the start-up window, which fails on every channel in most recordings.
        later = channel.get("flagged_fraction_after_startup")
        share = channel["flagged_fraction"] if later is None else later
        flags = channel["reasons"]
        status = "pass" if share == 0 else "fail" if share >= WARN_FRACTION else "warn"
        detail = (f"flagged in {100 * share:.0f}% of windows" + ("" if later is None else " after start-up")
                  + (": " + "; ".join(f"{explain(r, config).split(':')[0]} ({n})"
                                       for r, n in sorted(flags.items(), key=lambda kv: -kv[1])) if flags else ""))
        channels.append(dict(name=channel["channel"], label=_short(channel["channel"]), status=status, detail=detail))
    reasons = [f"{f['evidence']} → {f['action']}" for f in result["findings"]]
    if result.get("first_window_failed"):
        reasons.append("The first 2 s failed (a start-up transient is typical): discard the first seconds.")
    good = result["verdict"] == "good"
    headline = (f"Recording {result['verdict']}: {result['usable_windows']}/{result['windows']} windows pass "
                f"({100 * result['usable_fraction']:.0f}%)")
    if result.get("dead_electrodes"):
        reasons.insert(0, f"Usable without dead electrodes ≈ {100 * result['usable_fraction_if_dead_excluded']:.0f}% "
                          f"({', '.join(_short(c) for c in result['dead_electrodes'])})")
    return dict(state="accepted" if good else "withheld", headline=headline, reasons=reasons, channels=channels,
                summary=f"{sum(c['status'] == 'pass' for c in channels)}/{len(channels)} channels clean after start-up",
                source=result["path"])
