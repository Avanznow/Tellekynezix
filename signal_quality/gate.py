"""the fail-closed prediction gate moved to prediction-gaussiannb/eeg_quality.py on 19 October 2026.

The GUI, the GaussianNB live path and the tests import it from there; this module only keeps the package's
offline tools (dashboard.py) working from one implementation."""
import sys
from pathlib import Path

_GNB = Path(__file__).resolve().parents[1] / "prediction-gaussiannb"
if str(_GNB) not in sys.path:
    sys.path.insert(0, str(_GNB))
from eeg_quality import COMMANDS, QualityGatedPredictor, may_execute, withheld_result  

__all__ = ["COMMANDS", "QualityGatedPredictor", "may_execute", "withheld_result"]
