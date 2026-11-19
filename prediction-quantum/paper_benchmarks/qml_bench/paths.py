"""Repository paths. Inputs are read from prediction-quantum/results; outputs go to
prediction-quantum/results/paper/qml and temporary caches to a qml_bench folder in the system temp directory."""
import os
import tempfile
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]              # .../prediction-quantum/paper_benchmarks
QUANTUM_PKG = BENCH.parent                               # holds the prediction_quantum package
FEATURE_CACHE_DIR = QUANTUM_PKG / "results" / "cache"    # 570-feature window cache (12,089 windows)
RECORDING_SPLITS = QUANTUM_PKG / "results"               # split_seed{0,1,2}.csv of the original benchmark
RESULTS = QUANTUM_PKG / "results" / "paper" / "qml"
TMP = Path(tempfile.gettempdir()) / "qml_bench"            # scratch outside the repository


def setup():
    """Put the project's quantum package on sys.path and keep library caches inside paper_benchmarks."""
    if str(QUANTUM_PKG) not in sys.path:
        sys.path.insert(0, str(QUANTUM_PKG))
    # setdefault so a shell setting wins; without it matplotlib and fontconfig write under ~/.config and ~/.cache
    os.environ.setdefault("MPLCONFIGDIR", str(TMP / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(TMP / "cache"))
    # the env var is what the loky workers in run.py inherit; the flag below covers the interpreter already running
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True
