# Reconciliation of the replay and loader counts on Data_clean

Written by `prediction-gaussiannb/defense/reconcile_counts.py` from `signal_quality/data/real_replay_v3` (inventory.json, windows.jsonl, summary.json) and a fresh run of `eeg_data.load_recordings` with the default `LoadConfig` (board 16ch_125hz, excluded ['EXG Channel 15'], quality filter on) on `Data_clean`. Run time 5 s.

Checks that hold: the replay was run on `/home/prasndhdl/Downloads/Data_clean` and the loader on the repository copy; the two trees hold the same 1,827 .txt paths below the group folders, and every one of the 1,816 assessed recordings has the sha256 the replay recorded (0 mismatches). Both tools use the same quality policy (`avatar-raw-quality-v1-955cc6c1f48fa7c8` in both).

## Summary for Table 10

Both tools read the same 1,816 recordings. Data_clean holds 3,648 files: 1,821 BrainFlow CSV exports, which neither tool reads, and 1,827 OpenBCI-RAW .txt files. The replay attempts every .txt file and assessed 1,816; it reports 11 as invalid or unsupported, namely 10 files that carry the OpenBCI header and column line but no sample rows (five lines each, 'Empty recording') and 1 zero-byte file (G04/person09/Backward/OpenBCI-RAW-2025-03-14_15-56-17.txt). The loader's '1,826 found' is a header check only (the first 21 bytes read '%OpenBCI Raw EXG Data'), so it counts the 10 header-only files and excludes the zero-byte one; load_recording then skips those 10 with the same 'Empty recording' error and the 422 eight-channel recordings, which is the 432 skipped, leaving 1,394 sixteen-channel recordings, the same 1,394 files (sha256-identical to the files the replay hashed) that the replay assessed on the 16-channel board. 1,827 - 1 = 1,826 and 1,826 - 10 = 1,816.

Both tools cut non-overlapping 250-sample (2 s) windows from sample 0 of each of the 1,394 recordings, 6,703 windows in both (13 recordings are shorter than 2 s and give none). 2,501 windows pass the replay on all 16 channels and are kept by the loader. 748 fail the replay only on EXG Channel 15 (724 adc_saturation, flat_channel; 23 excessive_amplitude; 1 excessive_amplitude, excessive_sample_jump) and pass on the loader's 15 channels: 2,501 + 748 = 3,249. The start-up rule costs nothing against the replay: all 1,381 first windows fail the replay as well, and 0 of them would pass on the 15 kept channels. The other 3,454 windows are dropped by both tools: the 1,381 start-up blocks, 230 windows for packet discontinuity alone and 1,843 for a failure on a channel other than 15 (69 of those also discontinuous). Segment continuity removes no block (the 907 runs of kept blocks form 907 segments), the window alignment is identical in every recording, and a fresh run of the loader reproduces the derived decision on all 6,703 blocks (0 mismatches); residual 0. The 748 recovered windows belong to G03/person10 (239), G03/person08 (223), G04/person05 (183), G03/person09 (66), G04/person07 (33), G01/person10 (4), the participants whose channel 15 was dead or noisy, and they are why 18 participants are usable with channel 15 excluded.

## 1. Recording count: 1,827 .txt files, 1,826 named, 1,816 assessed

| Count | Replay (`replay.py`) | Loader (`eeg_data.load_recordings`) |
|---|---|---|
| Files under Data_clean | 3,648 scanned | 3,648 |
| Not `.txt` (BrainFlow CSV exports) | 1,821 `unsupported_export_files` | 1,821, never listed (`rglob('*.txt')`) |
| `.txt` files (all named `OpenBCI-RAW-*.txt`) | 1,827 attempted with `read_openbci` | 1,827 checked for the header |
| Header check `%OpenBCI Raw EXG Data` | not a separate step | 1,826 pass = 'named OpenBCI recordings found' |
| Read and assessed / loaded or board-skipped | 1,816 assessed (1,394 on 16 ch, 422 on 8 ch) | 1,394 loaded (16 ch) + 422 skipped for the 8-channel board |
| Rejected after the header | 11 `invalid_or_unsupported_named_files` | 10 skipped as unreadable ('Empty recording') |

Set differences (paths from the group folder on):

- Counted by the loader, not assessed by the replay: **10** files (all `Empty recording`).
- Attempted by the replay, not counted by the loader: **1** file (no header).
- Assessed by the replay but not counted by the loader, or loaded on one board and assessed on another: **0**.
- The 1,394 recordings the replay assessed on 16 channels are exactly the 1,394 the loader loaded (identical sets); the 422 assessed on 8 channels are exactly the 422 the loader skipped for the board (identical sets).
- Loader arithmetic: 1,827 .txt - 1 headerless = 1,826 named; 1,826 - 422 other board - 10 empty = 1,394 loaded. Replay arithmetic: 1,827 .txt - 11 invalid = 1,816 assessed.

Per-file list of the difference:

| File | Bytes | Lines | Replay | Loader | Why |
|---|---|---|---|---|---|
| `G01/person01/takeoff/OpenBCI-RAW-2025-03-03_11-24-54.txt` | 539 | 5 | unassessed: Empty recording | named (header check passes), then skipped: Empty recording | OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header check only, so it counts the file before load_recording raises the same error |
| `G01/person05/OpenBCISession_backward_5/OpenBCI-RAW-2025-03-08_17-23-33.txt` | 539 | 5 | unassessed: Empty recording | named (header check passes), then skipped: Empty recording | OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header check only, so it counts the file before load_recording raises the same error |
| `G01/person06/OpenBCISession_left_6/OpenBCI-RAW-2025-03-11_21-11-13.txt` | 539 | 5 | unassessed: Empty recording | named (header check passes), then skipped: Empty recording | OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header check only, so it counts the file before load_recording raises the same error |
| `G01/person09/OpenBCISession_Backward_9/OpenBCI-RAW-2025-03-12_14-33-19.txt` | 671 | 5 | unassessed: Empty recording | named (header check passes), then skipped: Empty recording | OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header check only, so it counts the file before load_recording raises the same error |
| `G03/person01/Backward/OpenBCI-RAW-2025-03-11_12-48-03.txt` | 671 | 5 | unassessed: Empty recording | named (header check passes), then skipped: Empty recording | OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header check only, so it counts the file before load_recording raises the same error |
| `G03/person01/Forward/OpenBCI-RAW-2025-03-11_12-34-25.txt` | 671 | 5 | unassessed: Empty recording | named (header check passes), then skipped: Empty recording | OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header check only, so it counts the file before load_recording raises the same error |
| `G03/person01/Forward/OpenBCI-RAW-2025-03-11_12-42-06.txt` | 671 | 5 | unassessed: Empty recording | named (header check passes), then skipped: Empty recording | OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header check only, so it counts the file before load_recording raises the same error |
| `G03/person01/Takeoff/OpenBCI-RAW-2025-03-11_12-26-06.txt` | 671 | 5 | unassessed: Empty recording | named (header check passes), then skipped: Empty recording | OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header check only, so it counts the file before load_recording raises the same error |
| `G03/person08/Takeoff/OpenBCI-RAW-2025-04-16_12-18-43.txt` | 671 | 5 | unassessed: Empty recording | named (header check passes), then skipped: Empty recording | OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header check only, so it counts the file before load_recording raises the same error |
| `G03/person08/Takeoff/OpenBCI-RAW-2025-04-16_12-19-37.txt` | 671 | 5 | unassessed: Empty recording | named (header check passes), then skipped: Empty recording | OpenBCI header and column line but no sample rows; the loader's '1,826 found' is a header check only, so it counts the file before load_recording raises the same error |
| `G04/person09/Backward/OpenBCI-RAW-2025-03-14_15-56-17.txt` | 0 | 0 | unassessed: Requires a named OpenBCI physical acquisition header | not counted: first 21 bytes are not '%OpenBCI Raw EXG Data' | zero-byte file; the replay attempts every .txt file and counts the failure among its 11 invalid files, the loader never lists it |

## 2. Window count on the 16-channel board: 2,501 acceptable against 3,249 kept

Both tools cut windows of round(2 s x 125 Hz) = 250 samples from sample 0, stride 250, trailing samples unscored. The replay assesses all 16 channels; the loader assesses 15 (EXG Channel 15 excluded) and always drops block 0. The per-channel rules (flat, saturated, amplitude, sample jump) are evaluated per channel and the window-level rules (packet counter, arrival gap, timestamps) do not depend on the channels kept, so the loader's decision on a window follows from the replay's record: kept if and only if it is not block 0, carries no window-level reason, and every bad channel is channel 15. That derivation was checked against a fresh loader run.

| Class | Windows |
|---|---|
| both keep | 2,501 |
| loader keeps, replay drops: fails only on EXG Channel 15 | 748 |
| replay keeps, loader drops: start-up block that passes on all 16 channels | 0 |
| replay keeps, loader drops: other | 0 |
| both drop: start-up block | 1,381 |
| both drop: window-level reason only (packet_discontinuity) | 230 |
| both drop: fails on a channel other than 15 | 1,843 |
| **total 16-channel windows** | **6,703** |

- Replay acceptable on the full montage: 2,501 = both keep (2,501) + start-up blocks that pass (0).
- Loader kept: 3,249 = both keep (2,501) + fails only on channel 15 (748); derived from the replay 3,249; **residual 0**.
- Fails only on channel 15, by reason: 724 adc_saturation, flat_channel; 23 excessive_amplitude; 1 excessive_amplitude, excessive_sample_jump.
- Start-up blocks (1,381, one per recording with at least one window): 1,381 fails on the 15 kept channels too. The loader's unconditional start-up rule therefore removes no window the replay would have accepted.
- Failures on a channel other than 15 (1,843): 1,774 channel reasons only; 69 also a window-level reason.
- Differing window alignment: 0 recordings (block counts agree in all 1,394; 13 recordings shorter than 2 s have no window in either tool).
- Segment continuity: 0 blocks. Continuity only splits runs of kept blocks into segments at a block boundary where the packet counter or arrival time jumps; the loader's 907 runs of kept blocks became 907 segments, so no run was split. A jump inside a window is `packet_discontinuity` / `acquisition_gap` in both tools.
- Block-by-block check against the fresh loader run: 6,703 blocks, 3,249 kept, 0 mismatches with the derived decision.
- Recordings with at least one kept block: 883; participants with usable data: 18 of 23.

Per participant (16-channel board):

| Participant | Windows | Start-up | Both keep | Only ch 15 fails | Discontinuity only | Other channel fails | Replay acceptable | Loader kept |
|---|---|---|---|---|---|---|---|---|
| G01/person08 | 293 | 59 | 229 | 0 | 5 | 0 | 229 | 229 |
| G01/person09 | 296 | 59 | 226 | 0 | 3 | 8 | 226 | 226 |
| G01/person10 | 298 | 60 | 205 | 4 | 28 | 1 | 205 | 209 |
| G03/person01 | 192 | 64 | 17 | 0 | 47 | 64 | 17 | 17 |
| G03/person02 | 301 | 60 | 0 | 0 | 0 | 241 | 0 | 0 |
| G03/person03 | 300 | 60 | 226 | 0 | 0 | 14 | 226 | 226 |
| G03/person04 | 300 | 60 | 236 | 0 | 0 | 4 | 236 | 236 |
| G03/person05 | 299 | 60 | 239 | 0 | 0 | 0 | 239 | 239 |
| G03/person06 | 303 | 60 | 243 | 0 | 0 | 0 | 243 | 243 |
| G03/person07 | 303 | 60 | 231 | 0 | 0 | 12 | 231 | 231 |
| G03/person08 | 290 | 58 | 0 | 223 | 0 | 9 | 0 | 223 |
| G03/person09 | 300 | 60 | 0 | 66 | 0 | 174 | 0 | 66 |
| G03/person10 | 300 | 60 | 0 | 239 | 0 | 1 | 0 | 239 |
| G04/person01 | 296 | 60 | 160 | 0 | 53 | 23 | 160 | 160 |
| G04/person02 | 269 | 60 | 112 | 0 | 90 | 7 | 112 | 112 |
| G04/person03 | 302 | 60 | 0 | 0 | 0 | 242 | 0 | 0 |
| G04/person04 | 267 | 61 | 102 | 0 | 0 | 104 | 102 | 102 |
| G04/person05 | 303 | 61 | 50 | 183 | 0 | 9 | 50 | 233 |
| G04/person06 | 295 | 60 | 0 | 0 | 0 | 235 | 0 | 0 |
| G04/person07 | 299 | 60 | 0 | 33 | 2 | 204 | 0 | 33 |
| G04/person08 | 300 | 60 | 0 | 0 | 0 | 240 | 0 | 0 |
| G04/person09 | 293 | 59 | 225 | 0 | 2 | 7 | 225 | 225 |
| G04/person10 | 304 | 60 | 0 | 0 | 0 | 244 | 0 | 0 |
| **all** | **6,703** | **1,381** | **2,501** | **748** | **230** | **1,843** | **2,501** | **3,249** |

Note: the replay read `/home/prasndhdl/Downloads/Data_clean` (3,648 files) and the loader the repository copy (3,648 files). Below the group folders the two trees list the same paths; file sizes differ in 1,820 of the 1,821 BrainFlow CSV exports, which neither tool reads, and in 0 of the 1,827 .txt recordings.
