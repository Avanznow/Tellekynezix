from pathlib import Path

import numpy as np
import torch

import sys


_SVM_MODEL_DIR = Path(__file__).resolve().parent.parent / "prediction-svm" / "pytorch"
if str(_SVM_MODEL_DIR) not in sys.path:
    sys.path.append(str(_SVM_MODEL_DIR))

from svm_model import SVMClassifier


class SVMPytorchPredictor:
    """Load the trained SVM model and return a single command label."""

    class_map = {
        0: "backward",
        1: "forward",
        2: "landing",
        3: "left",
        4: "right",
        5: "takeoff",
    }

    def __init__(self, model_path=None):
        if model_path is None:
            candidates = [
                _SVM_MODEL_DIR / "svm_trained.pkl",
                _SVM_MODEL_DIR / "svm_classifier.pkl",
            ]
            for candidate in candidates:
                if candidate.exists():
                    model_path = candidate
                    break
            if model_path is None:
                model_path = candidates[0]

        self.model = SVMClassifier.load(str(model_path))

    def __call__(self, X):
        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a torch.Tensor")

        predictions = self.model.predict(X)
        if len(predictions) == 0:
            raise ValueError("SVM model returned no predictions")

        first_prediction = predictions[0]
        if isinstance(first_prediction, (np.integer, int)):
            return self.class_map.get(int(first_prediction), f"class_{int(first_prediction)}")

        return str(first_prediction)
