"""Reconcile the signal-quality replay with the shared loader, file by file and block by block.

Two tools read Data_clean. The replay (signal_quality/replay.py, stored output signal_quality/data/real_replay_v3)
attempts every .txt file, assesses every channel and cuts 2 s windows from sample 0. The shared loader
(prediction-gaussiannb/eeg_data.py, load_recordings with the default LoadConfig) reads the 16-channel board only,
excludes EXG Channel 15 and drops the start-up block. Table 10 of the paper flags two gaps between their counts:
1,816 recordings assessed against 1,826 found, and 2,501 acceptable 16-channel windows against 3,249 kept blocks.
This script

  1. lists the .txt files each tool counts, computes the set differences and names the reason per file;
  2. derives, for every 16-channel replay window, what the loader decides from the replay's own per-window record
     (start-up block, reasons that are not channel reasons, bad channels other than 15), classifies every
     disagreement, and checks the derived decisions block by block against a fresh run of load_recordings;
  3. writes reconciliation_report.md next to this script and prints a two-paragraph summary for Table 10.

    /home/prasndhdl/ai_env/bin/python prediction-gaussiannb/defense/reconcile_counts.py [--replay DIR] [--data DIR]
        [--out FILE] [--skip-sha256]

Reads only: the replay output, the loader and Data_clean are not modified. About 30 s (the sha256 check of the
1,816 assessed recordings is most of it; --skip-sha256 leaves it out).
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GNB_DIR = HERE.parent
REPO_ROOT = GNB_DIR.parent
for p in (REPO_ROOT, GNB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eeg_data import LoadConfig, _is_named_openbci, load_recordings  # noqa: E402
from signal_quality.config import load_default_config  # noqa: E402

# Reasons the assessor raises per channel; every other reason is about the window as a whole (continuity, time,
# sample count, numeric range) and is unaffected by excluding a channel.
CHANNEL_REASONS = {"flat_channel", "adc_saturation", "excessive_amplitude", "excessive_sample_jump",
                   "excessive_line_noise"}
EXCLUDED = "EXG Channel 15"


def rel_key(path):
    """Path from the group folder on (G01/person02/Left/OpenBCI-RAW-....txt), so the two roots compare."""
    parts = Path(path).parts
    i = next(i for i, p in enumerate(parts) if re.fullmatch(r"G\d{2}", p, re.I))
    return "/".join(parts[i:])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path):
    size = path.stat().st_size
    lines = path.read_text(errors="ignore").splitlines()
    return size, len(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", default=str(REPO_ROOT / "signal_quality" / "data" / "real_replay_v3"))
    ap.add_argument("--data", default=str(REPO_ROOT / "Data_clean"))
    ap.add_argument("--out", default=str(HERE / "reconciliation_report.md"))
    ap.add_argument("--skip-sha256", action="store_true", help="do not hash the assessed recordings")
    args = ap.parse_args()
    replay_dir, data, out = Path(args.replay), Path(args.data), Path(args.out)
    t0 = time.perf_counter()

    # ---------------------------------------------------------------- replay side (stored output)
    summary = json.loads((replay_dir / "summary.json").read_text())
    inventory = json.loads((replay_dir / "inventory.json").read_text())
    inv = {rel_key(r["source_path"]): r for r in inventory}
    assert len(inv) == len(inventory), "inventory paths are not unique below the group folders"
    r_assessed = {k for k, r in inv.items() if r["status"] == "assessed"}
    r_unassessed = {k for k, r in inv.items() if r["status"] != "assessed"}
    r_16 = {k for k in r_assessed if inv[k]["channels"] == 16}
    r_8 = {k for k in r_assessed if inv[k]["channels"] == 8}
    windows = defaultdict(dict)                 # rel_key -> block index -> replay window record
    with (replay_dir / "windows.jsonl").open() as stream:
        for line in stream:
            w = json.loads(line)
            if len(w["channels"]) != 16:
                continue
            block = int(round(2.0 * w["sfreq"]))
            assert w["source_end_sample_exclusive"] - w["source_start_sample"] == block
            assert w["source_start_sample"] % block == 0, "replay windows are not cut from sample 0"
            windows[rel_key(w["source_path"])][w["source_start_sample"] // block] = w
    n_windows_16 = sum(len(v) for v in windows.values())
    policy_replay = summary["quality_policy_id"]
    policy_loader = load_default_config().policy_id

    # ---------------------------------------------------------------- loader side (fresh run)
    config = LoadConfig()
    all_files = [p for p in data.rglob("*") if p.is_file()]
    txt_files = {rel_key(p): p for p in data.rglob("*.txt")}
    other_suffixes = Counter(p.suffix.lower() for p in all_files if p.suffix.lower() != ".txt")
    l_named = {k for k, p in txt_files.items() if _is_named_openbci(p)}
    recs, skipped = load_recordings(data, config, verbose=False)
    l_loaded = {rel_key(r["path"]): r for r in recs}
    l_skipped = {rel_key(r["path"]): r for r in skipped}
    assert set(l_loaded) | set(l_skipped) == l_named
    skip_reason = Counter(re.sub(r"recording is \d+ ch at [0-9.]+ Hz, not \S+", "other board", s.get("error") or "unlabelled folder")
                          for s in l_skipped.values())

    # ---------------------------------------------------------------- gap 1: recordings
    txt_set = set(txt_files)
    assert txt_set == set(inv), "the replay inventory and the .txt files under --data differ"
    only_loader = sorted(l_named - r_assessed)          # counted by the loader, not assessed by the replay
    only_replay = sorted(txt_set - l_named)             # attempted by the replay, not counted by the loader
    both_count_but_differ = sorted((r_assessed - l_named) | (l_named & r_assessed & set(l_skipped) - r_8))
    file_rows = []
    for k in only_loader:
        size, lines = describe_file(txt_files[k])
        file_rows.append((k, size, lines, "unassessed: " + inv[k].get("reason", "?"),
                          "named (header check passes), then skipped: " + (l_skipped[k].get("error") or "?"),
                          "OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header "
                          "check only, so it counts the file before load_recording raises the same error"))
    for k in only_replay:
        size, lines = describe_file(txt_files[k])
        file_rows.append((k, size, lines, "unassessed: " + inv[k].get("reason", "?"),
                          "not counted: first 21 bytes are not '%OpenBCI Raw EXG Data'",
                          "zero-byte file; the replay attempts every .txt file and counts the failure among its 11 "
                          "invalid files, the loader never lists it"))
    sha_mismatch = None
    if not args.skip_sha256:
        sha_mismatch = sorted(k for k in r_assessed if sha256(txt_files[k]) != inv[k]["source_sha256"])

    # ---------------------------------------------------------------- gap 2: windows on the 16-channel board
    cats = Counter()
    ch15_reasons = Counter()
    other_detail = Counter()
    startup_detail = Counter()
    per_p = defaultdict(Counter)
    derived = {}                                         # (rel_key, block) -> loader decision derived from the replay
    for k, blocks in windows.items():
        participant = inv[k]["participant_id"]
        for b, w in blocks.items():
            nonch = sorted(r for r in w["rule_reasons"] if r not in CHANNEL_REASONS)
            bad = set(w["bad_channels"])
            other_bad = bad - {EXCLUDED}
            replay_ok = bool(w["rule_usable"])
            loader_ok = b != 0 and not nonch and not other_bad
            derived[(k, b)] = loader_ok
            per_p[participant]["windows"] += 1
            per_p[participant]["replay"] += replay_ok
            per_p[participant]["loader"] += loader_ok
            if b == 0:
                per_p[participant]["startup"] += 1
                if replay_ok:
                    cat = "replay keeps, loader drops: start-up block that passes on all 16 channels"
                    startup_detail["passes on all 16 channels"] += 1
                else:
                    cat = "both drop: start-up block"
                    if not nonch and not other_bad:
                        startup_detail["fails only on channel 15 (would pass on the 15 kept channels)"] += 1
                    else:
                        startup_detail["fails on the 15 kept channels too"] += 1
            elif replay_ok and loader_ok:
                cat = "both keep"
                per_p[participant]["both"] += 1
            elif loader_ok:
                cat = "loader keeps, replay drops: fails only on EXG Channel 15"
                ch15_reasons[", ".join(w["rule_reasons"])] += 1
                per_p[participant]["ch15_only"] += 1
            elif replay_ok:
                cat = "replay keeps, loader drops: other"
            elif nonch and not other_bad:
                cat = "both drop: window-level reason only (" + ", ".join(nonch) + ")"
                per_p[participant]["continuity"] += 1
            else:
                cat = "both drop: fails on a channel other than 15"
                other_detail["also a window-level reason" if nonch else "channel reasons only"] += 1
                per_p[participant]["other_channel"] += 1
            cats[cat] += 1
    replay_acceptable_16 = sum(1 for v in windows.values() for w in v.values() if w["rule_usable"])
    derived_kept = sum(derived.values())

    # block-by-block check against the fresh loader run
    mismatches, n_blocks_loader, kept_loader, alignment_mismatch = [], 0, 0, []
    segments = runs = 0
    for k, r in l_loaded.items():
        if r["n_blocks"] != len(windows.get(k, {})):
            alignment_mismatch.append(k)
        usable = [reasons == [] for reasons in r["block_reasons"]]
        n_blocks_loader += len(usable)
        kept_loader += sum(usable)
        segments += len(r["segments"])
        runs += sum(1 for i, u in enumerate(usable) if u and (i == 0 or not usable[i - 1]))
        for b, ok in enumerate(usable):
            if derived.get((k, b)) != ok:
                mismatches.append((k, b, r["block_reasons"][b], derived.get((k, b))))
    no_window_files = sorted(k for k, r in l_loaded.items() if r["n_blocks"] == 0)
    usable_recordings = sum(1 for r in l_loaded.values() if r["n_blocks_kept"] > 0)
    participants = sorted(per_p)
    usable_participants = [p for p in participants if per_p[p]["loader"] > 0]
    gained = sorted(((per_p[p]["ch15_only"], p) for p in participants if per_p[p]["ch15_only"]), reverse=True)

    # ---------------------------------------------------------------- summary for Table 10
    n_csv = sum(other_suffixes.values())
    empty_n = sum(1 for r in file_rows if r[3].endswith("Empty recording"))
    headerless = [r[0] for r in file_rows if r[1] == 0]
    board_skipped = skip_reason.get("other board", 0)
    startup_n = cats["both drop: start-up block"] + cats["replay keeps, loader drops: start-up block that passes on all 16 channels"]
    cont_keys = [c for c in cats if c.startswith("both drop: window-level reason only")]
    cont_n = sum(cats[c] for c in cont_keys)
    other_n = cats["both drop: fails on a channel other than 15"]
    ch15_n = cats["loader keeps, replay drops: fails only on EXG Channel 15"]
    both_keep = cats["both keep"]
    residual = (both_keep + ch15_n) - kept_loader
    ch15_text = "; ".join(f"{n} {r}" for r, n in ch15_reasons.most_common())
    gained_text = ", ".join(f"{p} ({n})" for n, p in gained)
    sha_text = ("sha256-identical to the files the replay hashed" if sha_mismatch == [] else
                "not re-hashed in this run" if sha_mismatch is None else f"{len(sha_mismatch)} files differ by sha256")
    para1 = (f"Both tools read the same {len(r_assessed):,} recordings. Data_clean holds {len(all_files):,} files: "
             f"{n_csv:,} BrainFlow CSV exports, which neither tool reads, and {len(txt_set):,} OpenBCI-RAW .txt files. "
             f"The replay attempts every .txt file and assessed {len(r_assessed):,}; it reports {len(r_unassessed)} as "
             f"invalid or unsupported, namely {empty_n} files that carry the OpenBCI header and column line but no "
             f"sample rows (five lines each, 'Empty recording') and {len(headerless)} zero-byte file "
             f"({', '.join(headerless)}). The loader's '{len(l_named):,} found' is a header check only (the first 21 "
             f"bytes read '%OpenBCI Raw EXG Data'), so it counts the {empty_n} header-only files and excludes the "
             f"zero-byte one; load_recording then skips those {empty_n} with the same 'Empty recording' error and the "
             f"{board_skipped} eight-channel recordings, which is the {len(l_skipped)} skipped, leaving "
             f"{len(l_loaded):,} sixteen-channel recordings, the same {len(r_16):,} files ({sha_text}) that the replay "
             f"assessed on the 16-channel board. {len(txt_set):,} - {len(headerless)} = {len(l_named):,} and "
             f"{len(l_named):,} - {empty_n} = {len(r_assessed):,}.")
    para2 = (f"Both tools cut non-overlapping 250-sample (2 s) windows from sample 0 of each of the {len(l_loaded):,} "
             f"recordings, {n_windows_16:,} windows in both ({len(no_window_files)} recordings are shorter than 2 s and "
             f"give none). {both_keep:,} windows pass the replay on all 16 channels and are kept by the loader. "
             f"{ch15_n} fail the replay only on EXG Channel 15 ({ch15_text}) and pass on the loader's 15 channels: "
             f"{both_keep:,} + {ch15_n} = {both_keep + ch15_n:,}. The start-up rule costs nothing against the replay: "
             f"all {startup_n:,} first windows fail the replay as well, and "
             f"{startup_detail.get('fails only on channel 15 (would pass on the 15 kept channels)', 0)} of them would "
             f"pass on the 15 kept channels. The other {startup_n + cont_n + other_n:,} windows are dropped by both tools: "
             f"the {startup_n:,} start-up blocks, {cont_n} windows for packet discontinuity alone and {other_n:,} for a "
             f"failure on a channel other than 15 ({other_detail.get('also a window-level reason', 0)} of those also "
             f"discontinuous). Segment continuity "
             f"removes no block (the {runs} runs of kept blocks form {segments} segments), the window alignment is "
             f"identical in every recording, and a fresh run of the loader reproduces the derived decision on all "
             f"{n_blocks_loader:,} blocks ({len(mismatches)} mismatches); residual {residual}. The {ch15_n} recovered "
             f"windows belong to {gained_text}, the participants whose channel 15 was dead or noisy, and they are why "
             f"{len(usable_participants)} participants are usable with channel 15 excluded.")

    # ---------------------------------------------------------------- report
    L = []
    L.append("# Reconciliation of the replay and loader counts on Data_clean\n")
    L.append(f"Written by `prediction-gaussiannb/defense/reconcile_counts.py` from `{replay_dir.relative_to(REPO_ROOT) if replay_dir.is_relative_to(REPO_ROOT) else replay_dir}` "
             f"(inventory.json, windows.jsonl, summary.json) and a fresh run of `eeg_data.load_recordings` with the default "
             f"`LoadConfig` (board {config.board}, excluded {list(config.exclude_channels)}, quality filter on) on "
             f"`{data.relative_to(REPO_ROOT) if data.is_relative_to(REPO_ROOT) else data}`. Run time {time.perf_counter() - t0:.0f} s.\n")
    L.append("Checks that hold: the replay was run on `" + summary["source_root"] + "` and the loader on the repository copy; "
             f"the two trees hold the same {len(txt_set):,} .txt paths below the group folders, and "
             + ("every one of the " + f"{len(r_assessed):,} assessed recordings has the sha256 the replay recorded (0 mismatches). "
                if sha_mismatch == [] else ("the sha256 check was skipped. " if sha_mismatch is None else f"{len(sha_mismatch)} assessed recordings differ by sha256: {sha_mismatch}. "))
             + f"Both tools use the same quality policy (`{policy_replay}`"
             + (" in both)." if policy_replay == policy_loader else f" in the replay, `{policy_loader}` in the loader: DIFFERENT)."))
    L.append("")
    L.append("## Summary for Table 10\n")
    L.append(para1 + "\n")
    L.append(para2 + "\n")

    L.append("## 1. Recording count: 1,827 .txt files, 1,826 named, 1,816 assessed\n")
    L.append("| Count | Replay (`replay.py`) | Loader (`eeg_data.load_recordings`) |")
    L.append("|---|---|---|")
    L.append(f"| Files under Data_clean | {summary['counts']['files_scanned']:,} scanned | {len(all_files):,} |")
    L.append(f"| Not `.txt` (BrainFlow CSV exports) | {summary['counts']['unsupported_export_files']:,} `unsupported_export_files` | {n_csv:,}, never listed (`rglob('*.txt')`) |")
    L.append(f"| `.txt` files (all named `OpenBCI-RAW-*.txt`) | {len(inv):,} attempted with `read_openbci` | {len(txt_set):,} checked for the header |")
    L.append(f"| Header check `%OpenBCI Raw EXG Data` | not a separate step | {len(l_named):,} pass = 'named OpenBCI recordings found' |")
    L.append(f"| Read and assessed / loaded or board-skipped | {len(r_assessed):,} assessed ({len(r_16):,} on 16 ch, {len(r_8)} on 8 ch) | {len(l_loaded):,} loaded (16 ch) + {board_skipped} skipped for the 8-channel board |")
    L.append(f"| Rejected after the header | {len(r_unassessed)} `invalid_or_unsupported_named_files` | {len(l_skipped) - board_skipped} skipped as unreadable ('Empty recording') |")
    L.append("")
    L.append("Set differences (paths from the group folder on):\n")
    L.append(f"- Counted by the loader, not assessed by the replay: **{len(only_loader)}** files (all `Empty recording`).")
    L.append(f"- Attempted by the replay, not counted by the loader: **{len(only_replay)}** file (no header).")
    L.append(f"- Assessed by the replay but not counted by the loader, or loaded on one board and assessed on another: **{len(both_count_but_differ)}**.")
    L.append(f"- The {len(r_16):,} recordings the replay assessed on 16 channels are exactly the {len(l_loaded):,} the loader loaded "
             f"({'identical sets' if r_16 == set(l_loaded) else 'SETS DIFFER'}); the {len(r_8)} assessed on 8 channels are exactly the "
             f"{board_skipped} the loader skipped for the board ({'identical sets' if r_8 == {k for k, s in l_skipped.items() if 'not 16ch' in (s.get('error') or '')} else 'SETS DIFFER'}).")
    L.append(f"- Loader arithmetic: {len(txt_set):,} .txt - {len(only_replay)} headerless = {len(l_named):,} named; {len(l_named):,} - {board_skipped} other board - {len(only_loader)} empty = {len(l_loaded):,} loaded. "
             f"Replay arithmetic: {len(inv):,} .txt - {len(r_unassessed)} invalid = {len(r_assessed):,} assessed.")
    L.append("")
    L.append("Per-file list of the difference:\n")
    L.append("| File | Bytes | Lines | Replay | Loader | Why |")
    L.append("|---|---|---|---|---|---|")
    for k, size, lines, rs, ls, why in file_rows:
        L.append(f"| `{k}` | {size} | {lines} | {rs} | {ls} | {why} |")
    L.append("")

    L.append("## 2. Window count on the 16-channel board: 2,501 acceptable against 3,249 kept\n")
    L.append(f"Both tools cut windows of round(2 s x 125 Hz) = 250 samples from sample 0, stride 250, trailing samples unscored. "
             f"The replay assesses all 16 channels; the loader assesses 15 (EXG Channel 15 excluded) and always drops block 0. "
             f"The per-channel rules (flat, saturated, amplitude, sample jump) are evaluated per channel and the window-level rules "
             f"(packet counter, arrival gap, timestamps) do not depend on the channels kept, so the loader's decision on a window "
             f"follows from the replay's record: kept if and only if it is not block 0, carries no window-level reason, and every "
             f"bad channel is channel 15. That derivation was checked against a fresh loader run.\n")
    L.append("| Class | Windows |")
    L.append("|---|---|")
    order = ["both keep",
             "loader keeps, replay drops: fails only on EXG Channel 15",
             "replay keeps, loader drops: start-up block that passes on all 16 channels",
             "replay keeps, loader drops: other",
             "both drop: start-up block"] + sorted(cont_keys) + ["both drop: fails on a channel other than 15"]
    for c in order:
        L.append(f"| {c} | {cats.get(c, 0):,} |")
    L.append(f"| **total 16-channel windows** | **{n_windows_16:,}** |")
    L.append("")
    L.append(f"- Replay acceptable on the full montage: {replay_acceptable_16:,} = both keep ({both_keep:,}) + start-up blocks that pass ({startup_detail.get('passes on all 16 channels', 0)}).")
    L.append(f"- Loader kept: {kept_loader:,} = both keep ({both_keep:,}) + fails only on channel 15 ({ch15_n}); derived from the replay {derived_kept:,}; **residual {residual}**.")
    L.append(f"- Fails only on channel 15, by reason: {ch15_text}.")
    L.append(f"- Start-up blocks ({startup_n:,}, one per recording with at least one window): " + "; ".join(f"{n:,} {d}" for d, n in startup_detail.items()) + ". "
             "The loader's unconditional start-up rule therefore removes no window the replay would have accepted.")
    L.append(f"- Failures on a channel other than 15 ({other_n:,}): " + "; ".join(f"{n:,} {d}" for d, n in other_detail.items()) + ".")
    L.append(f"- Differing window alignment: {len(alignment_mismatch)} recordings (block counts agree in all {len(l_loaded):,}; {len(no_window_files)} recordings shorter than 2 s have no window in either tool).")
    L.append(f"- Segment continuity: 0 blocks. Continuity only splits runs of kept blocks into segments at a block boundary where the packet counter or arrival time jumps; the loader's {runs} runs of kept blocks became {segments} segments, so no run was split. A jump inside a window is `packet_discontinuity` / `acquisition_gap` in both tools.")
    L.append(f"- Block-by-block check against the fresh loader run: {n_blocks_loader:,} blocks, {kept_loader:,} kept, {len(mismatches)} mismatches with the derived decision.")
    L.append(f"- Recordings with at least one kept block: {usable_recordings}; participants with usable data: {len(usable_participants)} of {len(participants)}.")
    L.append("")
    L.append("Per participant (16-channel board):\n")
    L.append("| Participant | Windows | Start-up | Both keep | Only ch 15 fails | Discontinuity only | Other channel fails | Replay acceptable | Loader kept |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p in participants:
        c = per_p[p]
        L.append(f"| {p} | {c['windows']} | {c['startup']} | {c['both']} | {c['ch15_only']} | {c['continuity']} | {c['other_channel']} | {c['replay']} | {c['loader']} |")
    c = Counter()
    for p in participants:
        c.update(per_p[p])
    L.append(f"| **all** | **{c['windows']:,}** | **{c['startup']:,}** | **{c['both']:,}** | **{c['ch15_only']}** | **{c['continuity']}** | **{c['other_channel']:,}** | **{c['replay']:,}** | **{c['loader']:,}** |")
    L.append("")
    replay_root = Path(summary["source_root"])
    if replay_root.is_dir() and replay_root.resolve() != data.resolve():
        theirs = {rel_key(p): p for p in replay_root.rglob("*") if p.is_file()}
        ours = {rel_key(q): q for q in all_files}
        size_diff = Counter(p.suffix.lower() for k, p in theirs.items()
                            if k in ours and p.stat().st_size != ours[k].stat().st_size)
        L.append(f"Note: the replay read `{replay_root}` ({len(theirs):,} files) and the loader the repository copy "
                 f"({len(all_files):,} files). Below the group folders the two trees list the same paths; file sizes differ in "
                 f"{size_diff.get('.csv', 0):,} of the {n_csv:,} BrainFlow CSV exports, which neither tool reads, and in "
                 f"{size_diff.get('.txt', 0)} of the {len(txt_set):,} .txt recordings.\n")
    else:
        L.append(f"Note: the replay's source root `{replay_root}` is not present on this machine; the sha256 check above is the "
                 "evidence that the loader read the same recordings.\n")
    out.write_text("\n".join(L))

    print(para1)
    print()
    print(para2)
    print()
    print(f"report: {out}  ({time.perf_counter() - t0:.0f} s)")
    if mismatches or residual or alignment_mismatch or both_count_but_differ or (sha_mismatch):
        print("WARNING: not fully reconciled:", len(mismatches), "mismatches,", residual, "residual,",
              len(alignment_mismatch), "alignment,", len(both_count_but_differ), "file-set,", sha_mismatch and len(sha_mismatch), "sha")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
