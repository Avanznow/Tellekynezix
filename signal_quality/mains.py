"""Determine the mains (power-line) frequency from the recordings themselves.

Line interference is a narrow spectral peak. For every replayed recording, the median
power spectrum over its working channels (0.25 Hz resolution) is computed, and the
height of the 50 Hz and 60 Hz peaks above their neighbouring frequencies is measured
in dB. On the 250 Hz board the second harmonics (100 / 120 Hz) give independent
evidence; on the 125 Hz board they would alias, so they are not used there.

The impact of switching line-noise rejection on is computed exactly from the replay's
per-window worst-channel line ratios. Nothing changes the active configuration:
enabling mains_hz changes the quality policy, so replays and downstream outputs must
be regenerated afterwards.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np
from scipy.signal import welch

from .acquisition import file_sha256, read_openbci
from .assessment import QualityConfig

TEMPLATE = Path(__file__).with_name("mains_template.html")
CLEAR_DB = 6.0        # a peak this far above its neighbours counts as clear line interference
MARGIN_DB = 3.0       # ...and must beat the other candidate by this much
AGREEMENT = 0.8       # share of clear recordings that must agree for a recommendation
COVERAGE = 0.5        # share of all recordings that must show a clear peak
LINE_RATIOS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 0.9, 1.0]   # 1.0 = record mains, never reject on line ratio


def peak_db(freqs, psd, f0):
    """Peak height at f0 (max within ±0.5 Hz) over the median of 1.5–4 Hz flanks, in dB."""
    nyquist = freqs[-1]
    if f0 + 0.5 >= nyquist:
        return None
    peak = psd[(freqs >= f0 - 0.5) & (freqs <= f0 + 0.5)].max()
    flank = (np.abs(freqs - f0) >= 1.5) & (np.abs(freqs - f0) <= 4) & (freqs < nyquist - 0.5)
    background = np.median(psd[flank])
    return float(10 * np.log10(peak / background)) if background > 0 and peak > 0 else None


def recording_spectrum(capture, rail_uv):
    """Median PSD across channels that are neither flat nor touching the ADC rail."""
    raw, fs = capture["raw"], capture["sfreq"]
    good = [i for i in range(len(raw)) if np.ptp(raw[i]) > 1e-3 and not (np.abs(raw[i]) >= rail_uv).any()]
    nperseg = int(4 * fs)
    if not good or raw.shape[1] < nperseg:
        return None, None, len(good)
    freqs, psd = welch(raw[good], fs=fs, nperseg=nperseg, detrend="constant", axis=-1)
    return freqs, np.median(psd, axis=0), len(good)


def classify(p50, p60):
    def clear(a, b):
        return a is not None and a >= CLEAR_DB and (b is None or a - b >= MARGIN_DB)
    if clear(p60, p50):
        return "60"
    if clear(p50, p60):
        return "50"
    return "none"


def analyse(replay_dir):
    replay_dir = Path(replay_dir).resolve()
    summary = json.loads((replay_dir / "summary.json").read_text())
    config = QualityConfig(**summary["config"])
    rows = [json.loads(line) for line in (replay_dir / "windows.jsonl").read_text().splitlines() if line.strip()]
    sources = {}
    for row in rows:
        sources.setdefault(row["source_path"], row)

    recordings, spectra = [], defaultdict(list)
    for source, row in sorted(sources.items()):
        if file_sha256(source) != row["source_sha256"]:
            raise ValueError(f"Source changed since the replay: {source}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            capture = read_openbci(source)
        freqs, psd, good = recording_spectrum(capture, config.rail_uv)
        board = f"{len(capture['channels'])}ch_{capture['sfreq']:g}hz"
        entry = dict(source_path=source, participant_id=row["participant_id"],
                     group=row["participant_id"].split("/")[0], board=board, sfreq=capture["sfreq"],
                     seconds=round(capture["raw"].shape[1] / capture["sfreq"], 1), channels_used=good)
        if freqs is None:
            entry.update(peak_50_db=None, peak_60_db=None, peak_100_db=None, peak_120_db=None, verdict="too_short_or_no_channels")
        else:
            entry.update(peak_50_db=peak_db(freqs, psd, 50), peak_60_db=peak_db(freqs, psd, 60),
                         peak_100_db=peak_db(freqs, psd, 100), peak_120_db=peak_db(freqs, psd, 120))
            entry["verdict"] = classify(entry["peak_50_db"], entry["peak_60_db"])
            band = (freqs >= 30) & (freqs <= freqs[-1] - 0.5)
            spectra[board].append((freqs[band], psd[band] / np.median(psd[band])))
        recordings.append(entry)

    def tally(subset):
        votes = Counter(r["verdict"] for r in subset)
        clear = votes["50"] + votes["60"]
        med = lambda key: (float(np.median([r[key] for r in subset if r[key] is not None]))  # noqa: E731
                           if any(r[key] is not None for r in subset) else None)
        return dict(recordings=len(subset), clear_60=votes["60"], clear_50=votes["50"], no_clear_peak=votes["none"],
                    unassessed=votes["too_short_or_no_channels"],
                    share_of_clear_agreeing_60=votes["60"] / clear if clear else None,
                    median_peak_50_db=med("peak_50_db"), median_peak_60_db=med("peak_60_db"),
                    median_peak_100_db=med("peak_100_db"), median_peak_120_db=med("peak_120_db"))

    overall = tally(recordings)
    groups = {g: tally([r for r in recordings if r["group"] == g]) for g in sorted({r["group"] for r in recordings})}
    boards = {b: tally([r for r in recordings if r["board"] == b]) for b in sorted({r["board"] for r in recordings})}
    assessed = overall["recordings"] - overall["unassessed"]
    clear = overall["clear_50"] + overall["clear_60"]
    recommended = None
    for f, votes in (("60", overall["clear_60"]), ("50", overall["clear_50"])):
        if clear and votes / clear >= AGREEMENT and clear / max(1, assessed) >= COVERAGE:
            recommended = int(f)
    harmonic = boards.get("8ch_250hz")
    harmonic_support = None
    if harmonic and harmonic["median_peak_120_db"] is not None and harmonic["median_peak_100_db"] is not None:
        harmonic_support = harmonic["median_peak_120_db"] - harmonic["median_peak_100_db"]

    # Mean normalised spectrum per board (for the figure): median over recordings on a common grid.
    figure = {}
    for board, items in spectra.items():
        grid = items[0][0]
        stack = np.array([np.interp(grid, f, p) for f, p in items])
        figure[board] = dict(freqs=[round(float(x), 3) for x in grid],
                             median_relative_power=[round(float(x), 5) for x in np.median(stack, axis=0)],
                             recordings=len(items))

    impact = None
    if recommended is not None:
        impact = line_rule_impact(rows, config, recommended)
    result = dict(
        replay=str(replay_dir), replay_summary_sha256=file_sha256(replay_dir / "summary.json"),
        quality_policy_id=config.policy_id, configured_mains_hz=config.mains_hz,
        decision_rule=dict(clear_peak_db=CLEAR_DB, margin_db=MARGIN_DB, agreement=AGREEMENT, coverage=COVERAGE),
        recommended_mains_hz=recommended, overall=overall, groups=groups, boards=boards,
        harmonic_120_minus_100_db_on_250hz_board=harmonic_support, impact=impact, spectra=figure,
        recordings=recordings,
        interpretation=("A clear peak is >= 6 dB above neighbouring frequencies and >= 3 dB above the other "
                        "candidate. The recommendation needs >= 80% of clear recordings to agree and >= 50% of "
                        "recordings to show a clear peak. On the 125 Hz board 60 Hz lies near Nyquist (62.5 Hz) "
                        "and may be attenuated by on-board filtering, which can only weaken a 60 Hz peak."))
    result["recommendations"] = recommendations(result)
    return result


def line_rule_impact(rows, config, mains_hz):
    """Exact extra rejections if line-noise rejection were enabled (window max over channels)."""
    key = f"line{mains_hz}_ratio_max"
    table = []
    usable = np.array([r["rule_usable"] for r in rows])
    ratio = np.array([r["window_features"][key] if r["window_features"] else np.nan for r in rows])
    people = np.array([r["participant_id"] for r in rows])
    for threshold in sorted(set(LINE_RATIOS) | {config.max_line_ratio}):
        flagged = np.nan_to_num(ratio, nan=0.0) > threshold
        newly = usable & flagged
        table.append(dict(max_line_ratio=threshold, is_configured=threshold == config.max_line_ratio,
                          windows_flagged=int(flagged.sum()), currently_usable_newly_rejected=int(newly.sum()),
                          rejection_rate=float(1 - (usable & ~flagged).mean()),
                          participants_affected=int(len(set(people[newly])))))
    usable_ratio = ratio[usable & np.isfinite(ratio)]
    configured = next(row for row in table if row["is_configured"])
    # A line ratio cannot exceed 1, so max_line_ratio=1.0 records the verified mains frequency
    # (line ratios become meaningful, the "unverified mains" warning disappears) without rejecting.
    return dict(mains_hz=mains_hz, sweep=table,
                usable_window_line_ratio_percentiles={str(q): float(np.percentile(usable_ratio, q))
                                                      for q in (50, 90, 99, 100)} if usable_ratio.size else None,
                enabling_at_configured_ratio=dict(max_line_ratio=configured["max_line_ratio"],
                                                  newly_rejected=configured["currently_usable_newly_rejected"],
                                                  of_currently_usable=int(usable.sum()),
                                                  rejection_rate=configured["rejection_rate"]),
                suggested_config={**asdict(replace(config, mains_hz=mains_hz, max_line_ratio=1.0)),
                                  "_note": ("Records the verified mains frequency without line-ratio rejection. "
                                            "Adopting it changes the policy ID: re-run the replay, review set, "
                                            "sensitivity and salvage afterwards. Enable rejection (lower "
                                            "max_line_ratio) only if independent reviews support it.")})


def recommendations(result):
    """Plain-language actions that follow from the evidence."""
    f = result["recommended_mains_hz"]
    if f is None:
        return ["The recordings do not show a consistent mains peak; keep mains_hz unset until acquisition "
                "notes confirm the local mains frequency."]
    o, impact = result["overall"], result["impact"]
    enabled = impact["enabling_at_configured_ratio"]
    out = [f"Set mains_hz to {f}: {o['clear_' + str(f)]} of {o['recordings']} recordings show a clear {f} Hz peak "
           f"(median {o['median_peak_' + str(f) + '_db']:.1f} dB above neighbouring frequencies) and "
           f"{o['clear_' + str(110 - f)]} {'shows' if o['clear_' + str(110 - f)] == 1 else 'show'} a {110 - f} Hz peak.",
           f"Apply a {f} Hz notch filter in model preprocessing (and {2 * f} Hz for 250 Hz recordings); e.g. "
           f"prediction_eegnet's preprocessing accepts notch_freq={f}.",
           f"Do not enable line-ratio rejection at max_line_ratio={enabled['max_line_ratio']:g}: it would reject "
           f"{enabled['newly_rejected']} of the {enabled['of_currently_usable']} currently usable windows "
           f"(rejection {100 * enabled['rejection_rate']:.1f}%). Raw, unfiltered windows routinely carry most of "
           f"their power at {f} Hz, which a notch removes; record mains with max_line_ratio=1.0 (suggested_config.json) "
           "and decide on rejection only with independent reviews.",
           f"The {f} Hz pickup is strong in most recordings: check the bias/reference electrode contact, run the "
           "acquisition laptop on battery, and route electrode cables away from power cables."]
    return out


def write_outputs(result, out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mains-", dir=out.parent) as temporary:
        stage = Path(temporary) / "out"
        stage.mkdir()
        (stage / "mains.json").write_text(json.dumps(result, indent=1))
        fields = ["participant_id", "group", "board", "seconds", "channels_used", "peak_50_db", "peak_60_db",
                  "peak_100_db", "peak_120_db", "verdict", "source_path"]
        import csv
        with (stage / "recordings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result["recordings"])
        if result["impact"]:
            (stage / "suggested_config.json").write_text(json.dumps(result["impact"]["suggested_config"], indent=2))
        payload = json.dumps({k: v for k, v in result.items() if k != "recordings"} |
                             dict(recordings=[{k: r[k] for k in fields if k != "source_path"} for r in result["recordings"]]),
                             allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
        (stage / "report.html").write_text(TEMPLATE.read_text(encoding="utf-8").replace("/*__PAYLOAD__*/null", payload),
                                           encoding="utf-8")
        stage.rename(out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True, help="New output directory")
    args = parser.parse_args(argv)
    result = analyse(args.replay)
    out = write_outputs(result, args.out)
    o = result["overall"]
    print(f"Recordings: {o['recordings']}  clear 60 Hz: {o['clear_60']}  clear 50 Hz: {o['clear_50']}  "
          f"no clear peak: {o['no_clear_peak']}  unassessed: {o['unassessed']}")
    print(f"Median peak height: 60 Hz {o['median_peak_60_db']:.1f} dB, 50 Hz {o['median_peak_50_db']:.1f} dB")
    for g, t in result["groups"].items():
        print(f"  {g}: 60 Hz clear in {t['clear_60']}/{t['recordings']}, 50 Hz in {t['clear_50']}/{t['recordings']} "
              f"(median {t['median_peak_60_db']:.1f} vs {t['median_peak_50_db']:.1f} dB)")
    if result["harmonic_120_minus_100_db_on_250hz_board"] is not None:
        print(f"  250 Hz board harmonics: 120 Hz exceeds 100 Hz by {result['harmonic_120_minus_100_db_on_250hz_board']:.1f} dB")
    if result["recommended_mains_hz"]:
        print(f"Recommended mains_hz: {result['recommended_mains_hz']} (configured: {result['configured_mains_hz']})")
        print("If line-ratio rejection were enabled:")
        for row in result["impact"]["sweep"]:
            print(f"  max_line_ratio {row['max_line_ratio']:<5g}{'*' if row['is_configured'] else ' '} would newly reject "
                  f"{row['currently_usable_newly_rejected']} currently usable windows "
                  f"({row['participants_affected']} participants); rejection {100 * row['rejection_rate']:.1f}%")
    for i, text in enumerate(result["recommendations"], 1):
        print(f"{i}. {text}")
    print(f"Report: {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
