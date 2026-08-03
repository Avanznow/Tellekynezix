"""
Support Vector Machine (SVM) Classifier - PyTorch/Scikit-Learn Implementation
This module implements an SVM classifier for brain wave classification.
"""

import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle


class SVMClassifier:
    """
    Support Vector Machine (SVM) classifier for brain wave (EEG) data classification.
    
    This wrapper implements an SVM classifier using scikit-learn's SVC, optimized for
    predicting drone commands from EEG signals. The model includes feature scaling
    to ensure consistent performance across different input ranges.
    
    Attributes:
        - Configurable kernel (linear, rbf, poly)
        - Hyperparameter tuning support
        - Feature standardization
        - GPU-compatible predictions via PyTorch conversion
    """
    
    def __init__(self, num_features, num_classes, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize the SVM classifier.
        
        Args:
            num_features (int): Number of input features (e.g., 32 for EEG channels)
            num_classes (int): Number of output classes (e.g., 6 for drone commands)
            kernel (str): Type of kernel - 'linear', 'rbf', 'poly'. Default: 'rbf'
            C (float): Regularization parameter. Default: 1.0
            gamma (str or float): Kernel coefficient. Default: 'scale'
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        
        # Initialize SVM classifier
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        # Initialize feature scaler
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the SVM model to training data.
        
        This method:
        1. Scales features using StandardScaler
        2. Trains the SVM classifier
        3. Marks the model as fitted
        
        Args:
            X (np.ndarray or torch.Tensor): Training data of shape (n_samples, n_features)
            y (np.ndarray or torch.Tensor): Target labels of shape (n_samples,)
        """
        # Convert torch tensors to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        # Ensure y is in correct format
        y = np.asarray(y, dtype=np.int64)
        
        # Fit and transform the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the SVM model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict(self, X):
        """
        Make predictions using the trained SVM model.
        
        Args:
            X (np.ndarray or torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,)
            
        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Convert torch tensors to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for input data.
        
        Args:
            X (np.ndarray or torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes)
            
        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Convert torch tensors to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Get probability estimates
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def forward(self, X):
        """
        Convenience method for making predictions (similar to PyTorch models).
        
        Args:
            X (np.ndarray or torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels
        """
        return self.predict(X)
    
    def save(self, filepath):
        """
        Save the trained model and scaler to a file.
        
        Args:
            filepath (str): Path to save the model checkpoint
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        checkpoint = {
            'model': self.model,
            'scaler': self.scaler,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    @staticmethod
    def load(filepath):
        """
        Load a trained SVM model from a file.
        
        Args:
            filepath (str): Path to the saved model checkpoint
            
        Returns:
            SVMClassifier: Initialized and fitted SVM classifier
        """
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Recreate model
        model = SVMClassifier(
            num_features=checkpoint['num_features'],
            num_classes=checkpoint['num_classes'],
            kernel=checkpoint['kernel'],
            C=checkpoint['C'],
            gamma=checkpoint['gamma']
        )
        
        # Load trained components
        model.model = checkpoint['model']
        model.scaler = checkpoint['scaler']
        model.is_fitted = True
        
        return model


# Convenience function for quick model creation and training
def create_svm_model(num_features, num_classes, kernel='rbf', C=1.0, gamma='scale'):
    """
    Create and initialize an SVM classifier.
    
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
        kernel (str): Kernel type for SVM
        C (float): Regularization parameter
        gamma (str or float): Kernel coefficient
        
    Returns:
        SVMClassifier: Initialized SVM classifier
    """
    return SVMClassifier(num_features, num_classes, kernel, C, gamma)
