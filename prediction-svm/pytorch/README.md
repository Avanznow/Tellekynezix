# Support Vector Machine (SVM) - PyTorch/Scikit-Learn Implementation

## Overview

This folder contains a Support Vector Machine (SVM) classifier implementation for brain wave (EEG) data classification to predict drone commands. The SVM model uses scikit-learn's `SVC` with a customizable PyTorch-compatible wrapper class that handles data preprocessing, training, and inference.

### Files

- **svm_model.py**: Contains the `SVMClassifier` class - a custom wrapper around scikit-learn's SVC
- **svm_train.ipynb**: Simplified Jupyter notebook for training the model with EEG data
- **SVM_Classifier.ipynb**: Comprehensive notebook with hyperparameter tuning and detailed evaluation
- **svm_trained.pkl**: Trained model file (generated after training)
- **README.md**: This documentation file

## How It Works

### Support Vector Machine Algorithm

SVM is a powerful supervised learning algorithm that finds the optimal hyperplane to separate classes in high-dimensional spaces. For multiclass classification, it uses the One-vs-Rest (OvR) strategy.

**Key Steps:**

1. **Feature Scaling**:
   - All features are standardized using `StandardScaler` (zero mean, unit variance)
   - This is critical for SVM performance as it's distance-based

2. **Training (fit method)**:
   - Finds the optimal separation hyperplane by maximizing the margin between classes
   - Solves the optimization problem:
   ```
   Minimize: (1/2)||w||² + C * Σ(ξᵢ)
   ```
   - C parameter controls the trade-off between margin maximization and training error

3. **Prediction (forward method)**:
   - Scales new input using fitted scaler
   - Calculates distance to the decision boundary
   - Assigns class based on which side of the boundary the point falls

### PyTorch/Scikit-Learn Integration

- **Scikit-learn SVC**: Core classifier implementing the SVM algorithm
- **StandardScaler**: Persistent feature scaling for consistent preprocessing
- **PyTorch Compatibility**: Accepts both numpy arrays and PyTorch tensors as input
- **Model Persistence**: Saves both model and scaler using pickle for reproducibility

## Key Features

- Pure scikit-learn SVM implementation with PyTorch-compatible interface
- Configurable kernels: linear, RBF (Radial Basis Function), polynomial
- Hyperparameter tuning support (C, gamma, kernel, degree)
- Automatic feature scaling with persisted StandardScaler
- Probability estimates for confidence scores
- GPU-compatible tensor input (automatically converted to numpy for processing)
- Can save/load trained models using pickle

## Mathematical Background

### Kernel Functions

**Linear Kernel:**
```
K(x₁, x₂) = x₁ · x₂
```
Best for linearly separable data.

**RBF (Radial Basis Function) Kernel:**
```
K(x₁, x₂) = exp(-γ ||x₁ - x₂||²)
```
Excellent for non-linear patterns. γ (gamma) controls the reach of each support vector.

**Polynomial Kernel:**
```
K(x₁, x₂) = (γ(x₁ · x₂) + r)^d
```
Where d is the degree parameter.

### Optimization Problem

The SVM solves:
```
Minimize: (1/2)||w||² + C * Σ(ξᵢ)
Subject to: yᵢ(wᵀφ(xᵢ) + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

- **||w||²**: Margin to minimize (want hyperplane far from both classes)
- **C**: Regularization parameter (smaller = larger margin, more training errors allowed)
- **ξᵢ**: Slack variables (allow some misclassification)

## Configuration Parameters

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel` | 'rbf' | Kernel type: 'linear', 'rbf', 'poly' |
| `C` | 1.0 | Regularization strength (smaller = stronger regularization) |
| `gamma` | 'scale' | Kernel coefficient for 'rbf', 'poly', 'sigmoid' |
| `degree` | 3 | Degree of polynomial kernel (for 'poly' kernel only) |

### Parameter Tuning Guide

- **C (Regularization)**:
  - Small C (0.1): Wider margin, allows more misclassification (underfitting risk)
  - Large C (100): Narrow margin, fewer misclassifications (overfitting risk)

- **Gamma**:
  - Small gamma: Each support vector has far-reaching influence (smooth decision boundary)
  - Large gamma: Each support vector has close influence (jagged decision boundary)

- **Kernel Selection**:
  - Linear: Fast, good for high-dimensional linear data
  - RBF: Default choice, handles most non-linear cases
  - Poly: Useful for specific domain knowledge patterns

## Usage

### Training

1. **Update data path** in `svm_train.ipynb`:
```python
   DATA_DIR = "/path/to/your/brainwave_readings/"
```

2. **Run the notebook** to:
   - Load brain wave .txt files
   - Extract labels from filenames (backward, forward, landing, left, right, takeoff)
   - Train the SVM model
   - Evaluate accuracy on test set
   - Save trained model

3. **Brain wave data format**:
   - `.txt` files with CSV format (skip first 4 header lines)
   - Filenames must contain command labels
   - Default: 32 EEG feature columns

### Basic Inference

```python
from svm_model import SVMClassifier
import numpy as np

# Create and train model
model = SVMClassifier(num_features=32, num_classes=6, kernel='rbf')
X_train = np.load('training_data.npy')  # Shape: (n_samples, 32)
y_train = np.load('training_labels.npy')
model.fit(X_train, y_train)

# Make predictions
X_new = np.load('new_data.npy')
predictions = model.predict(X_new)  # Returns class indices

# Get probability estimates
probabilities = model.predict_proba(X_new)  # Shape: (n_samples, n_classes)
```

### Loading Pre-trained Model

```python
from svm_model import SVMClassifier

# Load trained model
model = SVMClassifier.load('svm_trained.pkl')

# Use for predictions
predictions = model.predict(X_new)
```

### PyTorch Integration

```python
import torch
from svm_model import SVMClassifier

model = SVMClassifier.load('svm_trained.pkl')

# Input as PyTorch tensor
X_tensor = torch.randn(10, 32)  # 10 samples, 32 features
predictions = model.predict(X_tensor)  # Automatically converted
```

### Hyperparameter Tuning

For detailed hyperparameter tuning, use `SVM_Classifier.ipynb` which includes:
- Grid Search over multiple parameter combinations
- 5-fold cross-validation
- Comprehensive evaluation metrics
- Result visualization

```python
# Grid search is performed in SVM_Classifier.ipynb
# Best parameters are then used for final model training
```

## Performance Characteristics

### Advantages
- Works well in high-dimensional spaces
- Memory efficient (uses only support vectors)
- Versatile with different kernel functions
- Good for both binary and multiclass classification
- Strong theoretical foundation

### Disadvantages
- Slower training on very large datasets (>100k samples)
- Sensitive to feature scaling (why StandardScaler is used)
- Hyperparameter tuning can be computationally expensive
- Less interpretable than decision trees

## Typical Performance

On EEG-based drone command classification:
- **Accuracy**: 85-95% (depending on data quality and class balance)
- **Training Time**: 1-10 seconds for 1000 samples
- **Prediction Time**: <1ms per sample

## Troubleshooting

### Issue: Poor Accuracy
- Check if features are properly scaled
- Try different kernel functions (rbf, linear, poly)
- Tune regularization parameter C

### Issue: Slow Training
- Reduce dataset size for hyperparameter tuning
- Lower C parameter for faster convergence
- Use linear kernel for high-dimensional data

### Issue: Memory Error
- Reduce dataset size
- Use 'sparse' data format if applicable
- Consider data sampling/stratification

## References

- [Scikit-learn SVC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Support Vector Machines - Theory](https://en.wikipedia.org/wiki/Support-vector_machine)
- [EEG Signal Processing for BCI](https://ieeexplore.ieee.org/)

## Authors & Contributors

- BCI/Drone Command Prediction System
- Part of Tellekynezix Project

## License

See LICENSE file in project root.
