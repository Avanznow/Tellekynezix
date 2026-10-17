"""The project's deployed quality policy (default_config.json), as opposed to library defaults."""

import json
from pathlib import Path

from .assessment import QualityConfig

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.json")


def load_default_config(path=DEFAULT_CONFIG_PATH):
    """QualityConfig from default_config.json: verified 60 Hz mains, line ratios recorded but not rejected."""
    return QualityConfig(**json.loads(Path(path).read_text()))
