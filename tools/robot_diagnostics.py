"""Standalone launcher: python tools/robot_diagnostics.py --mode simulation."""

from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from robot_services.diagnostics import main

    raise SystemExit(main())
