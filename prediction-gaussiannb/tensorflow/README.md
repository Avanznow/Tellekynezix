# GaussianNB - TensorFlow implementation

- **gaussiannb_model.py**: `GaussianNBTF`, a Gaussian Naive Bayes whose class statistics and log-likelihoods are computed
  with TensorFlow (CPU is enough), with pickle save / load helpers.
- **gaussiannb_train.ipynb**: the training notebook. Besides its three original experiments it trains, on the repo's
  `Data_clean`, the improved global chain (570 window features -> per-participant z-score -> LDA projection -> one
  `GaussianNBTF`) and the subject-specific chain (435 features -> per-participant z-score -> own LDA + own `GaussianNBTF`
  per participant, global fallback), ported from the PyTorch notebook where they were found and audited. Windows are
  125 samples with a 62-sample stride, cut inside the loader's kept 2 s blocks.
- **gaussiannb_tf_improved_global.pkl / gaussiannb_tf_subject_specific.pkl**: the two chains as trained by the notebook
  (`ParticipantScaler`, projections, `GaussianNBTF` models, label names, window geometry, loader configuration).
- **gaussiannb_tf_protocol_comparison.csv / gaussiannb_tf_subject_per_participant.csv**: their results.
- **subject_gnb_tf.py**: the TensorFlow backend of the live path (`ProjectedGaussianNBTF`, a loader for the pickles,
  which were written with the class name `gaussiannb_model.GaussianNBTF` that collides with the PyTorch module of the
  same name). The live modules and the GUI route live in `../pytorch/` (`improved_gnb_inference.py`,
  `subject_gnb_inference.py`, `backend="tensorflow"`); see `../pytorch/README.md`, section *TensorFlow framework*.

Numbers on `Data_clean` (6 commands, chance 16.7 %, the notebook's test files): best original experiment 22.5 % window
accuracy; improved global chain 51.7 % (65.0 % per file); subject-specific chain 85.9 % (89.8 % per file). Held-out
participants stay at chance for every chain: each command was recorded as one contiguous block per participant, so these
numbers measure within-session separability, as the notebooks explain.
