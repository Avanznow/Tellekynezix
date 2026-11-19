"""Test-session set-up shared by the GaussianNB suites.

prediction-gaussiannb/ holds folders named jax/ and tensorflow/. Every suite puts that directory on sys.path, so on a
machine without the real libraries `import jax` or `import tensorflow` would return an empty namespace package and
the suites' importorskip / find_spec checks would pass, only to fail later with AttributeErrors. Registering None
for such a name makes the import raise ImportError, which is what an absent library looks like; a really installed
library wins over a namespace portion (PEP 420) and is left untouched.
"""
import importlib.util
import sys
from pathlib import Path

GNB_DIR = Path(__file__).resolve().parents[1]
if str(GNB_DIR) not in sys.path:
    sys.path.insert(0, str(GNB_DIR))

for _name in ("jax", "tensorflow"):
    if _name in sys.modules:
        continue
    _spec = importlib.util.find_spec(_name)
    if _spec is not None and _spec.origin is None:          # namespace package: only the shadowing folder exists
        sys.modules[_name] = None
