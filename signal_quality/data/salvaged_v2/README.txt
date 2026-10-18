Salvaged EEG windows (signal_quality.salvage).

5235 windows exported, 1855 of them recovered by excluding dead electrodes.
Per board folder: X.npy [windows, channels, samples] raw µV (excluded channels = NaN), mask.npy [windows, channels], windows.jsonl (provenance, folder labels).

Load:  from signal_quality.salvage import load_salvaged
       data = load_salvaged('<this folder>', '16ch_125hz', montage='common')

WARNING: Folder labels are proxies without cue timing; in Data_clean every class was recorded as one contiguous block, so labels are confounded with recording order. Evaluate on held-out participants; recovered windows do not make the labels more trustworthy.
