from typing import Any, Dict, Tuple
import numpy as np
import tensorflow as tf
import pickle
from scipy.signal import butter, sosfiltfilt  # welch only if you un-comment PSD code
from scipy.stats import skew, kurtosis


class KMeansTF:
    """
    K-Means clustering implementation using TensorFlow.
    Can be used for classification by mapping clusters to class labels after training.
    
    USAGE: k_means_train_processed.ipynb cells 10-11
      - Cell 10: model = KMeansTF(...); model.fit(X_train, y_train); model.predict(X_test)
      - Cell 11: model.predict_proba(X_test) for confidence analysis
    """
    def __init__(self, n_clusters: int = 6, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.cluster_centers_ = None  # (n_clusters, n_features)
        self.labels_ = None  # (n_samples,)
        self.cluster_to_label_map_ = None  # Maps cluster indices to class labels for classification
        self.n_features_ = None
        self.inertia_ = None  # Sum of squared distances to nearest cluster center

    def _initialize_centers(self, X: tf.Tensor, n_samples: int) -> tf.Tensor:
        """Initialize cluster centers using k-means++ initialization.
        USAGE: Called internally by fit() when init=None (cell 10 indirect)
        """
        tf.random.set_seed(self.random_state)
        n_features = tf.shape(X)[1]
        
        # First center: randomly select one data point
        first_idx = tf.random.uniform([], 0, n_samples, dtype=tf.int32)
        centers = tf.expand_dims(X[first_idx], 0)  # (1, n_features)
        
        # Select remaining centers using k-means++ algorithm
        for _ in range(self.n_clusters - 1):
            # Compute distances from each point to nearest center
            distances = tf.reduce_min(
                tf.reduce_sum(tf.square(X[:, tf.newaxis, :] - centers[tf.newaxis, :, :]), axis=2),
                axis=1
            )  # (n_samples,)
            
            # Convert distances to probabilities (squared distances)
            probs = distances / (tf.reduce_sum(distances) + 1e-10)
            
            # Sample next center based on probabilities
            cumsum_probs = tf.cumsum(probs)
            r = tf.random.uniform([], 0.0, 1.0)
            idx = tf.argmax(tf.cast(cumsum_probs >= r, tf.int32))
            centers = tf.concat([centers, tf.expand_dims(X[idx], 0)], axis=0)
        
        return centers

    def fit(self, X: np.ndarray, y: np.ndarray = None, init: np.ndarray = None):
        """
        Fit K-means clustering model (USAGE: Cell 10 direct)

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Optional labels for creating cluster-to-label mapping
            init: Optional (n_clusters, n_features) initial centers (e.g. class means for supervised).
                  If None, use k-means++ (for kmeans_train). If provided, use for kmeans_supervised_train.
        """
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        n_samples = tf.shape(X_tf)[0] # Number of training windows
        n_features = tf.shape(X_tf)[1] # Number of features per window (e.g. 14 EEG + 3 accel features)

        # Initialize cluster centers: supervised init (kmeans_supervised_train) or k-means++ (kmeans_train)
        if init is not None:
            init = np.asarray(init, dtype=np.float32)
            if init.shape != (self.n_clusters, int(n_features.numpy())):
                raise ValueError(
                    f"init must have shape (n_clusters={self.n_clusters}, n_features={n_features.numpy()}), got {init.shape}"
                )
            centers = tf.convert_to_tensor(init, dtype=tf.float32)
        else:
            centers = self._initialize_centers(X_tf, n_samples)

        # K-means iteration
        for iteration in range(self.max_iter):
            # Assign each point to nearest cluster
            distances = tf.reduce_sum(
                tf.square(X_tf[:, tf.newaxis, :] - centers[tf.newaxis, :, :]),
                axis=2
            )  # (n_samples, n_clusters)
            labels = tf.argmin(distances, axis=1)  # (n_samples,)
            
            # Update cluster centers
            new_centers = tf.zeros_like(centers)
            counts = tf.zeros([self.n_clusters], dtype=tf.int32)
            
            for k in range(self.n_clusters):
                mask = tf.cast(labels == k, tf.float32)  # (n_samples,)
                count = tf.reduce_sum(mask)
                if count > 0:
                    new_centers = tf.tensor_scatter_nd_update(
                        new_centers,
                        [[k]],  # Updates cluster k's center
                        [tf.reduce_sum(X_tf * mask[:, tf.newaxis], axis=0) / count]
                    )
                counts = tf.tensor_scatter_nd_update(counts, [[k]], [tf.cast(count, tf.int32)])
            
            # Check convergence
            center_shift = tf.reduce_sum(tf.square(new_centers - centers))
            centers = new_centers
            
            if center_shift < self.tol:
                break
        
        # Store results
        self.cluster_centers_ = centers.numpy()
        self.labels_ = labels.numpy()
        self.n_features_ = int(n_features.numpy())
        
        # Compute inertia (sum of squared distances)
        distances = tf.reduce_sum(
            tf.square(X_tf[:, tf.newaxis, :] - centers[tf.newaxis, :, :]),
            axis=2
        )
        min_distances = tf.reduce_min(distances, axis=1)
        self.inertia_ = float(tf.reduce_sum(min_distances).numpy())
        
        # Create cluster-to-label mapping if labels provided
        if y is not None:
            self._create_cluster_label_map(y)
        
        return self

    def _create_cluster_label_map(self, y: np.ndarray):
        """Map each cluster to the most common label in that cluster.
        USAGE: Called by fit() when y is provided (cell 10 indirect)
        """
        cluster_to_label = {}
        for k in range(self.n_clusters):
            mask = self.labels_ == k
            if np.any(mask):
                cluster_labels = y[mask]
                # Find most common label in this cluster
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                most_common_label = unique_labels[np.argmax(counts)]
                cluster_to_label[k] = int(most_common_label)
            else:
                # If cluster is empty, assign to first class
                cluster_to_label[k] = 0
        self.cluster_to_label_map_ = cluster_to_label

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data (USAGE: Cell 10 direct).
        If cluster_to_label_map_ exists, returns class labels instead of cluster indices.
        
        Args:
            X: Data of shape (n_samples, n_features)
            
        Returns:
            Predicted cluster indices or class labels (n_samples,)
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        centers = tf.convert_to_tensor(self.cluster_centers_, dtype=tf.float32)
        
        # Compute distances to all cluster centers
        distances = tf.reduce_sum(
            tf.square(X_tf[:, tf.newaxis, :] - centers[tf.newaxis, :, :]),
            axis=2
        )  # (n_samples, n_clusters)
        
        # Assign to nearest cluster
        cluster_indices = tf.argmin(distances, axis=1).numpy().astype(np.int32)
        
        # Map clusters to labels if mapping exists
        if self.cluster_to_label_map_ is not None:
            return np.array([self.cluster_to_label_map_[int(idx)] for idx in cluster_indices], dtype=np.int32)
        
        return cluster_indices

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities based on distance to cluster centers (USAGE: Cell 11 direct).
        Uses softmax of negative distances as probabilities.
        
        Args:
            X: Data of shape (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model must be fitted before making predictions")
        if self.cluster_to_label_map_ is None:
            raise RuntimeError("Model must have cluster-to-label mapping for probability predictions")
        
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        centers = tf.convert_to_tensor(self.cluster_centers_, dtype=tf.float32)
        
        # Compute distances to all cluster centers
        distances = tf.reduce_sum(
            tf.square(X_tf[:, tf.newaxis, :] - centers[tf.newaxis, :, :]),
            axis=2
        )  # (n_samples, n_clusters)
        
        # Convert distances to probabilities using softmax of negative distances
        # (closer = higher probability)
        neg_distances = -distances
        probs = tf.nn.softmax(neg_distances, axis=1).numpy()  # (n_samples, n_clusters)
        
        # Map cluster probabilities to class probabilities
        n_classes = len(set(self.cluster_to_label_map_.values()))
        class_probs = np.zeros((X.shape[0], n_classes), dtype=np.float32)
        
        for cluster_idx, class_idx in self.cluster_to_label_map_.items():
            class_probs[:, class_idx] += probs[:, cluster_idx]
        
        # Normalize to ensure probabilities sum to 1
        class_probs = class_probs / (np.sum(class_probs, axis=1, keepdims=True) + 1e-10)
        
        return class_probs


def save_model(path: str, model: KMeansTF, meta: Dict[str, Any]) -> None:
    """Save K-means model and metadata to file (USAGE: Cell 12 direct).
    Saves pickled cluster centers, label mapping, and preprocessing stats (mu, sd).
    """
    payload: Dict[str, Any] = {
        'model': {
            'n_clusters': int(model.n_clusters),
            'cluster_centers_': model.cluster_centers_,
            'cluster_to_label_map_': model.cluster_to_label_map_,
            'n_features_': model.n_features_,
            'inertia_': model.inertia_,
        },
        'meta': meta or {},
        'format': 'KMeansTF-pickle-v1'
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str) -> Tuple[KMeansTF, Dict[str, Any]]:
    """Load K-means model and metadata from file (NOT USED in current notebook).
    Use for inference workflows to load saved models and make predictions on new EEG data.
    """
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    
    m = KMeansTF(n_clusters=payload['model']['n_clusters'])
    m.cluster_centers_ = np.array(payload['model']['cluster_centers_'], dtype=np.float32)
    m.cluster_to_label_map_ = payload['model'].get('cluster_to_label_map_')
    m.n_features_ = int(payload['model']['n_features_'])
    m.inertia_ = payload['model'].get('inertia_', 0.0)
    
    return m, payload.get('meta', {})

def apply_bandpass_to_signal(data, fs, lowcut=1.0, highcut=50.0):
    """
    Applies a 4th-order Butterworth filter to the whole signal (USAGE: Cell 8 direct).
    Data should be (samples, channels). Filters 1-50 Hz band per-file before windowing.
    """
    if fs <= highcut * 2:  # Nyquist safety check
        highcut = (fs / 2) - 1
    
    sos = butter(4, [lowcut, highcut], btype="band", fs=fs, output="sos")
    # axis=0 filters along the time dimension for each channel
    try:
        return sosfiltfilt(sos, data, axis=0)
    except ValueError:
        # For very short segments, sosfiltfilt can fail if the time dimension
        # is shorter than the required padding length. In that case, just
        # return the input unfiltered so we don't crash the pipeline.
        return data

def get_hjorth_params(sig):
    """Calculates Hjorth Mobility and Complexity temporal features.
    USAGE: Called by extract_window_features() for each EEG channel (cell 8 indirect)
    """
    diff = np.diff(sig)
    diff2 = np.diff(diff)
    
    var0 = np.var(sig)
    var1 = np.var(diff)
    var2 = np.var(diff2)
    
    if var0 == 0 or var1 == 0:
        return 0.0, 0.0
        
    mobility = np.sqrt(var1 / var0)
    complexity = (np.sqrt(var2 / var1)) / mobility
    return mobility, complexity

def extract_window_features(window_eeg, window_accel, fs):
    """
    Computes features for a single 1-second time window (USAGE: Cell 8 direct).
    Per-channel: Hjorth (mobility, complexity) + skewness + kurtosis + accel means.
    """
    row_feats = []
    
    # 1. EEG Features per channel
    for ch in range(window_eeg.shape[1]):
        sig = window_eeg[:, ch]
        
        # Hjorth
        mob, comp = get_hjorth_params(sig)
        # Moments
        sk = skew(sig)
        kt = kurtosis(sig)
        
        # Optional: Uncomment if you want PSD bands back
            # f, psd = welch(sig, file_fs, nperseg=win_size)

            # # Frequency bands
            # d = np.mean(psd[(f >= 0.5) & (f < 4)])
            # t = np.mean(psd[(f >= 4) & (f < 8)])
            # a = np.mean(psd[(f >= 8) & (f < 13)])
            # b = np.mean(psd[(f >= 13) & (f < 30)])
            # g_band = np.mean(psd[(f >= 30) & (f < 50)])

            # total = d + t + a + b + g_band + 1e-12
            # # Relative power in each band
            # row_feats.extend([d / total, t / total, a / total, b / total, g_band / total])
        
        row_feats.extend([mob, comp, sk, kt])
    
    # 2. Accelerometer Features (Mean per axis)
    if window_accel.size > 0:
        row_feats.extend(np.mean(window_accel, axis=0).tolist())
        
    return row_feats