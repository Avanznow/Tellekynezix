"""Avatar local-prediction import path for the window-based EEGNet wrapper.

Capture/extract EEG with BrainFlowDataProcessor, then supply explicitly mapped
electrodes and verified preprocessing metadata. Its legacy get_tensor() applies
an independently fitted scaler and must not feed this predictor.
"""

from prediction_eegnet.predictor import AvatarEEGNetPredictor

__all__ = ["AvatarEEGNetPredictor"]
