"""Avatar raw EEG quality assessment (offline tools). The prediction gate and the GUI view moved to
prediction-gaussiannb/eeg_quality.py on 19 October 2026."""

from .assessment import QualityAssessor, QualityConfig

__all__ = ["QualityAssessor", "QualityConfig"]
