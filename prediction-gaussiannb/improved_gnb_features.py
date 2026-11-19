"""Improved window features and per-participant normalisation for the GaussianNB EEG pipeline.

The chain used by the PyTorch GaussianNB notebook on Data_clean:

    windows (n, channels, samples)  ->  extract_features  ->  ParticipantScaler (train statistics)
        ->  linear projection fitted on train (LDA, 5 dims, or PCA 95 %)  ->  GaussianNB

Feature blocks (every one is a pure function of the window samples; nothing from labels, file names, recording
order or participant identity):
  logcov   log-Euclidean covariance of the window: C = Xc Xc^T / T (+ 1e-3 tr(C)/ch I), matrix log via eigh,
           upper triangle incl. diagonal, off-diagonals scaled by sqrt(2)                        ch (ch+1) / 2
  dcov     the same covariance block on the first-difference signal diff(X, axis=time)          ch (ch+1) / 2
  tkeo     Teager-Kaiser energy psi[t] = x[t]^2 - x[t-1] x[t+1]: log mean |psi| and mean |psi| / var(x)   2 ch
  hybrid   the notebook's 10 features per channel (mean, std, ptp, energy, line-length and 5 log1p relative band
           powers from a Welch PSD) with log(|x| + 1e-6) applied to std, ptp, energy and line-length     10 ch
  car      the same hybrid block on common-average-referenced windows (x - mean over channels)     10 ch
With 15 channels this is 120 + 120 + 30 + 150 + 150 = 570 features.

Selected on 5-fold recording-grouped cross-validation of Data_clean and confirmed on an independent partition:
window accuracy 0.25 -> 0.49 (6 commands), per-recording accuracy 0.27 -> 0.63. Held-out participants stay at
chance with every chain on this dataset (its commands were recorded as contiguous blocks), so the gain is
within-participant separability; see the notebook for the full discussion.
"""
from pathlib import Path

import numpy as np
from scipy.signal import welch

FEATURE_VERSION = "improved_logcov_dcov_tkeo_hybridlog_car_v1"
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
STAT_NAMES = ["mean", "log_std", "log_ptp", "log_energy", "log_linelen"]
WELCH_NPERSEG = 128


def logcov(X, shrink=1e-3):
    """Windows X (n, ch, t) -> (n, ch (ch + 1) / 2): upper triangle of the matrix log of the shrunk window covariance."""
    X = np.asarray(X, np.float64)
    X = X - X.mean(-1, keepdims=True)
    C = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    d = C.shape[1]
    # Trace-scaled ridge: shrink times the mean eigenvalue regularises a rank-deficient window (a railed channel,
    # say) without visibly moving healthy ones; an all-zero window has zero trace, so it is the 1e-12 floor below
    # that keeps the log finite there.
    C = C + shrink * (np.trace(C, axis1=1, axis2=2) / d)[:, None, None] * np.eye(d)
    w, V = np.linalg.eigh(C)
    L = np.einsum("ncd,nd,ned->nce", V, np.log(np.maximum(w, 1e-12)), V)
    iu = np.triu_indices(d)
    # Off-diagonals get sqrt(2) so the Euclidean norm of the vector equals the Frobenius norm of log(C): distances
    # between windows are then the log-Euclidean ones, which is what the LDA projection downstream should see.
    return L[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


def tkeo(X):
    """Windows (n, ch, t) -> (n, 2 ch): [log(mean |psi| + 1e-9), mean |psi| / var(x)] per channel (Teager-Kaiser energy)."""
    X = np.asarray(X, np.float64)
    X = X - X.mean(-1, keepdims=True)
    # psi needs x[t-1] and x[t+1], so it exists for t = 1 .. T-2: one sample is dropped at each end of the window.
    psi = X[..., 1:-1] ** 2 - X[..., :-2] * X[..., 2:]
    m = np.abs(psi).mean(-1)
    return np.concatenate([np.log(m + 1e-9), m / (X.var(-1) + 1e-10)], -1)


def hybrid_log(X, fs=125, nperseg=WELCH_NPERSEG, bands=BANDS):
    """The notebook's hybrid features per channel, with the four scale statistics in the log domain -> (n, 10 ch).

    Per channel: mean, log std, log peak-to-peak, log energy, log line-length, then log1p of the relative delta,
    theta, alpha, beta and gamma power from a Welch PSD (nperseg = min(nperseg, window length), 50 % overlap),
    relative to the 0.5-45 Hz power. Channel-major order: all 10 features of channel 0, then channel 1, ...
    """
    X = np.asarray(X, np.float64)
    stats = np.stack([X.mean(-1), np.log(X.std(-1) + 1e-6), np.log(np.ptp(X, axis=-1) + 1e-6),
                      np.log((X * X).mean(-1) + 1e-6), np.log(np.abs(np.diff(X, axis=-1)).mean(-1) + 1e-6)], -1)
    # Cap nperseg at the window length so a window shorter than 128 samples still yields a (single-segment) PSD.
    nperseg = min(nperseg, X.shape[-1])
    freqs, psd = welch(X, fs=fs, nperseg=nperseg, noverlap=min(nperseg // 2, nperseg - 1), axis=-1)
    am = (freqs >= 0.5) & (freqs < 45.0)
    # Band powers below are relative to the 0.5-45 Hz total; the floor stops a flat (all-zero) channel giving nan.
    total = np.maximum(np.trapezoid(psd[..., am], freqs[am], axis=-1), 1e-12)
    rel = []
    for lo, hi in bands.values():
        bm = (freqs >= lo) & (freqs < hi)
        bp = np.trapezoid(psd[..., bm], freqs[bm], axis=-1) if bm.any() else np.zeros(X.shape[:-1])
        rel.append(np.log1p(bp / total))
    return np.concatenate([stats, np.stack(rel, -1)], -1).reshape(X.shape[0], -1)


def extract_features(W, fs=125):
    """Windows W (n, ch, t) -> (n, ch (ch + 1) + 22 ch) float32: [logcov | dcov | tkeo | hybrid | car]."""
    W = np.asarray(W, np.float64)
    car = W - W.mean(axis=1, keepdims=True)
    # dcov is logcov of the first difference: differencing is a crude high-pass, so it picks up the beta/gamma
    # structure that the raw covariance, dominated by low-frequency power, tends to hide.
    return np.concatenate([logcov(W), logcov(np.diff(W, axis=-1)), tkeo(W), hybrid_log(W, fs), hybrid_log(car, fs)],
                          axis=1).astype(np.float32)


def feature_names(channels):
    """Column names of extract_features for the given channel names, in the same order."""
    ch = list(channels)
    # Same row-major upper-triangle order (i <= j) that logcov uses, so the names line up with its columns.
    iu = np.triu_indices(len(ch))
    cov = [f"logcov_{ch[i]}_{ch[j]}" for i, j in zip(*iu)]
    dcov = [f"dcov_{ch[i]}_{ch[j]}" for i, j in zip(*iu)]
    tk = [f"{c}_log_tkeo" for c in ch] + [f"{c}_tkeo_over_var" for c in ch]
    hyb = [f"{c}_{s}" for c in ch for s in STAT_NAMES + [f"{b}_relpow" for b in BANDS]]
    car = [f"car_{n}" for n in hyb]
    return cov + dcov + tk + hyb + car


def participant_of(path):
    """'<group>/<person>' of a recording path, as eeg_data.py labels participants (e.g. 'G03/person03')."""
    parts = Path(str(path)).parts
    # Group folders look like 'G03'; the isdigit() check keeps other 'G...' folder names from being read as a group.
    group = next((p for p in parts if p.upper().startswith("G") and p[1:].isdigit()), "G??")
    person = next((p for p in parts if p.lower().startswith("person")), "person??")
    return f"{group.upper()}/{person.lower()}"


class ParticipantScaler:
    """Per-participant feature standardisation with statistics from the training windows only.

    fit(X, participants) stores the global mean / std and, for every participant with at least `min_windows`
    training windows, that participant's own mean / std. transform(X, participants) standardises each window with
    its participant's statistics, or with the global ones for an unknown participant. Nothing is ever fitted on
    test windows. add_participant() registers a new person from an unlabelled calibration recording (deployment).
    """

    def __init__(self, min_windows=2, eps=1e-9):
        self.min_windows = min_windows
        self.eps = eps

    def fit(self, X, participants):
        """Store the global and per-participant mean / std of the training features X (n, features).

        participants is one label per row (as participant_of gives them). A participant with fewer than min_windows
        rows gets no entry and falls back to the global statistics in transform().
        """
        X = np.asarray(X, np.float64)
        participants = np.asarray(participants)
        self.global_mean_ = X.mean(0)
        self.global_std_ = X.std(0) + self.eps
        self.stats_ = {}
        for q in np.unique(participants):
            m = participants == q
            # A single window has std 0, so its scale would be just eps and every later window of that person would
            # explode; hence the min_windows guard.
            if m.sum() >= self.min_windows:
                self.stats_[str(q)] = (X[m].mean(0), X[m].std(0) + self.eps)
        self.n_features_in_ = X.shape[1]
        return self

    def add_participant(self, X_calibration, participant):
        """Statistics for a new participant from their own (unlabelled) calibration windows."""
        X = np.asarray(X_calibration, np.float64)
        if len(X) < self.min_windows:
            raise ValueError(f"need at least {self.min_windows} calibration windows")
        self.stats_[str(participant)] = (X.mean(0), X.std(0) + self.eps)
        return self

    def transform(self, X, participants):
        """Standardise X (n, features) row by row with its participant's statistics; rows of an unknown participant
        use the global mean / std. Returns a float64 (n, features) array."""
        X = np.asarray(X, np.float64)
        participants = np.asarray(participants)
        # Start from the global standardisation; rows of known participants are overwritten below.
        Z = (X - self.global_mean_) / self.global_std_
        for q in np.unique(participants):
            if str(q) in self.stats_:
                mu, sd = self.stats_[str(q)]
                m = participants == q
                Z[m] = (X[m] - mu) / sd
        return Z

    def fit_transform(self, X, participants):
        """fit() on X, then transform() the same X: the usual call for the training split."""
        return self.fit(X, participants).transform(X, participants)

    def known_participants(self):
        """Sorted labels of the participants that have their own statistics (fitted or added)."""
        return sorted(self.stats_)


# ------------------------------------------------------------------------------------------------------------------
# Subject-specific feature set (435): the two log-covariances above plus a 30-45 Hz log-covariance of the RAW signal
# and multitaper absolute band powers of the raw window. Selected for the per-participant chain (subject_gnb.py);
# with per-participant models the notebook's per-channel statistics carry nothing beyond the covariances.
# ------------------------------------------------------------------------------------------------------------------
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from scipy.signal.windows import dpss

SUBJECT_FEATURE_VERSION = "subject_logcov_dcov_gammacov_multitaper_v1"
GAMMA_BAND = (30.0, 45.0)


def multitaper_log_abs_bandpower(W_raw, fs=125, NW=3.0, K=5, bands=BANDS, eps=1e-10):
    """DPSS multitaper PSD (time-bandwidth NW, K tapers) of the RAW demeaned windows (n, ch, t) -> log absolute
    band power in delta, theta, alpha, beta, gamma per channel -> (n, 5 ch)."""
    W = np.asarray(W_raw, np.float64)
    W = W - W.mean(-1, keepdims=True)
    n = W.shape[-1]
    tapers = dpss(n, NW, K)
    # tapers is (K, t); W[..., None, :] * tapers is (n, ch, K, t), one tapered copy of each channel per taper.
    S = np.fft.rfft(W[..., None, :] * tapers, axis=-1)
    # Mean over the K eigenspectra, scaled by 2 / fs to a one-sided density; bins times df then integrates a band.
    p = (np.abs(S) ** 2).mean(-2) * (2.0 / fs)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    df = f[1] - f[0]
    ba = np.stack([p[..., (f >= lo) & (f < hi)].sum(-1) * df for lo, hi in bands.values()], -1)
    return np.log(ba + eps).reshape(W.shape[0], -1)


def bandpass_segment(segment, fs=125, band=GAMMA_BAND, notch_hz=60.0, order=4):
    """Zero-phase Butterworth band-pass (+ mains notch, Q=30) of one RAW segment (samples, channels), demeaned per
    channel first. Applied per kept segment before windowing, like the loader's 1-40 Hz filter."""
    # Second-order sections keep the filter numerically stable: as (b, a) an order-4 band-pass is an 8th-order
    # polynomial whose coefficients are ill-conditioned at a 125 Hz sampling rate.
    sos = butter(order, [band[0], band[1]], btype="bandpass", fs=fs, output="sos")
    s = np.asarray(segment, np.float64)
    s = s - s.mean(axis=0, keepdims=True)
    s = sosfiltfilt(sos, s, axis=0)
    if notch_hz:
        b, a = iirnotch(notch_hz, Q=30.0, fs=fs)
        s = filtfilt(b, a, s, axis=0)
    return s.astype(np.float32)


def extract_subject_features(W_filt, W_raw, W_gamma, fs=125, chunk=3000):
    """Three aligned views of the same windows, each (n, ch, t):
         W_filt  : 1-40 Hz band-pass + 60 Hz notch (the loader's output, as the notebook windows them)
         W_raw   : the same windows cut from the quality-filtered, demeaned, UNFILTERED segments
         W_gamma : the same windows cut from the raw segments after bandpass_segment (30-45 Hz + notch)
       -> (n, ch (ch + 1) * 3 / 2 + 5 ch) float32 = [logcov(W_filt) | logcov(diff W_filt) | logcov(W_gamma) | multitaper(W_raw)]
       (435 for 15 channels)."""
    # Process `chunk` windows at a time: the (n, ch, K, t) multitaper array and the batched eigh would otherwise be
    # built for the whole split in one go.
    def run(fn, W):
        return np.concatenate([fn(np.asarray(W[s:s + chunk], np.float64)) for s in range(0, len(W), chunk)])
    F = np.concatenate([run(logcov, W_filt), run(lambda w: logcov(np.diff(w, axis=-1)), W_filt),
                        run(logcov, W_gamma), run(lambda w: multitaper_log_abs_bandpower(w, fs), W_raw)], 1)
    return F.astype(np.float32)


def subject_feature_names(channels):
    """Column names of extract_subject_features for the given channel names, in the same order."""
    ch = list(channels)
    iu = np.triu_indices(len(ch))
    cov = [f"logcov_{ch[i]}_{ch[j]}" for i, j in zip(*iu)]
    dcov = [f"dcov_{ch[i]}_{ch[j]}" for i, j in zip(*iu)]
    gcov = [f"gammacov_{ch[i]}_{ch[j]}" for i, j in zip(*iu)]
    mt = [f"{c}_mt_log_{b}" for c in ch for b in BANDS]
    return cov + dcov + gcov + mt
