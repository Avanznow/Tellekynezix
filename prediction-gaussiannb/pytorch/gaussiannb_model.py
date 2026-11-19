"""
GaussianNB Naive Bayes Classifier - PyTorch Implementation
This module implements Gaussian Naive Bayes using PyTorch for brain wave classification.
"""

import torch
import torch.nn as nn
import numpy as np


class GaussianNB(nn.Module):
    """
    Gaussian Naive Bayes classifier implemented in PyTorch.
    
    This implementation calculates mean (μ) and variance (σ²) for each feature
    per class during training, then uses the Gaussian Probability Density Function
    and Bayes' theorem for prediction.
    
    No traditional backpropagation is used - parameters are calculated directly
    from the training data.
    """
    
    def __init__(self, num_features, num_classes):
        """
        Initialize the GaussianNB model.
        
        Args:
            num_features (int): Number of input features (e.g., 32 for EEG channels)
            num_classes (int): Number of output classes (e.g., 6 for drone commands)
        """
        super(GaussianNB, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Register buffers (non-trainable parameters)
        # These will store the calculated statistics
        # Buffers move with .to(device) and are saved by state_dict, so a fitted model can be checkpointed like any
        # nn.Module; is_fitted is a tensor rather than a Python bool for the same reason.
        self.register_buffer('class_priors', torch.zeros(num_classes))
        self.register_buffer('means', torch.zeros(num_classes, num_features))
        self.register_buffer('variances', torch.zeros(num_classes, num_features))
        self.register_buffer('is_fitted', torch.tensor(False))
        
    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model.
        
        This calculates the mean and variance for each feature within each class.
        
        Args:
            X (torch.Tensor): Training data of shape (n_samples, n_features)
            y (torch.Tensor): Target labels of shape (n_samples,)
        """
        # Ensure tensors are on the same device as the model
        X = X.to(self.means.device)
        y = y.to(self.means.device).long()
        
        n_samples = X.shape[0]
        
        # Calculate class priors P(y)
        for c in range(self.num_classes):
            class_mask = (y == c)
            self.class_priors[c] = class_mask.sum().float() / n_samples
            
            # Get samples for this class
            X_c = X[class_mask]
            
            # Skip a class with no samples (.mean()/.var() of an empty tensor is NaN) and leave its buffers at zero;
            # subject_gnb.py remaps labels to the classes a participant actually has, so this is only a safeguard.
            if len(X_c) > 0:
                # Calculate mean μ for each feature
                self.means[c] = X_c.mean(dim=0)
                
                # Calculate variance σ² for each feature
                # Add small epsilon to avoid division by zero
                # torch.var is the unbiased (n-1) estimate; the TensorFlow/JAX models use the population variance.
                # 1e-9 is a flat additive constant, not sklearn's var_smoothing * max σ² (subject_gnb.py adds that
                # on top after fit).
                self.variances[c] = X_c.var(dim=0) + 1e-9
        
        self.is_fitted = torch.tensor(True)

    def _joint_log_likelihood(self, X):
        """
        Compute unnormalised log posteriors log(P(x|y=c)) + log(P(y=c)) for all classes at once.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)

        Returns:
            torch.Tensor: Log posteriors of shape (n_samples, n_classes)
        """
        # subject_gnb.py calls this directly for the per-participant log-posteriors; forward/predict_proba wrap it.
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        X = X.to(self.means.device)

        # Log of class prior: log(P(y=c)), shape (C,)
        # 1e-10 keeps the log finite for a class whose prior came out exactly zero.
        log_prior = torch.log(self.class_priors + 1e-10)

        # Log PDF: log(P(x|y=c)) = -0.5 * [log(2πσ²) + (x-μ)²/σ²]
        # The log(2πσ²) term does not depend on x, so sum it once per class.
        const_term = -0.5 * torch.log(2 * np.pi * self.variances).sum(dim=1)  # (C,)

        # (x - μ)² / σ² summed over features for every (sample, class) pair, shape (N, C)
        diff_sq = (X.unsqueeze(1) - self.means.unsqueeze(0)).pow(2)
        quad = -0.5 * (diff_sq / self.variances.unsqueeze(0)).sum(dim=2)

        # Posterior: log(P(y=c|x)) ∝ log(P(x|y=c)) + log(P(y=c))
        return quad + const_term + log_prior

    def forward(self, X):
        """
        Make predictions using the Gaussian Probability Density Function and Bayes' theorem.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)

        Returns:
            torch.Tensor: Predicted class labels of shape (n_samples,)
        """
        # Return class with highest posterior probability
        return torch.argmax(self._joint_log_likelihood(X), dim=1)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)

        Returns:
            torch.Tensor: Class probabilities of shape (n_samples, n_classes)
        """
        log_probs = self._joint_log_likelihood(X)

        # Convert log probabilities to probabilities using softmax
        # Softmax over the joint log likelihoods is exactly Bayes' normalisation exp(jll_c) / Σ_k exp(jll_k), done
        # in log space so the very negative values of a multi-feature Gaussian do not underflow to zero.
        probs = torch.softmax(log_probs, dim=1)
        return probs