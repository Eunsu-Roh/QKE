"""
Iris Dataset Loader

Load Iris dataset for baseline validation
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_iris_2d(test_size=0.2, random_state=42, binary=True, scale=True):
    """
    Load Iris dataset with all 4 features
    
    Args:
        test_size: Fraction of test data (default: 0.2)
        random_state: Random seed (default: 42)
        binary: If True, convert to binary classification (Setosa vs others)
        scale: If True, standardize features
        
    Returns:
        X_train, X_test, y_train, y_test
        
    Example:
        >>> X_train, X_test, y_train, y_test = load_iris_2d()
        >>> print(X_train.shape)  # (120, 4)
    """
    iris = load_iris()
    
    # Use all 4 features (sepal length, sepal width, petal length, petal width)
    X = iris.data
    y = iris.target
    
    if binary:
        # Binary classification: Setosa (0) vs others (1)
        y = (y != 0).astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardization
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def load_iris_binary(test_size=0.2, random_state=42, n_features=2):
    """
    Load Iris dataset for binary classification
    
    Simplified wrapper for load_iris_2d
    
    Args:
        test_size: Test set size
        random_state: Random seed  
        n_features: Number of features (2 or 4)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    iris = load_iris()
    
    if n_features == 2:
        X = iris.data[:, :2]
    elif n_features == 4:
        X = iris.data
    else:
        raise ValueError(f"n_features must be 2 or 4, got {n_features}")
    
    y = (iris.target != 0).astype(int)  # Binary: Setosa vs others
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test
    print("Loading Iris dataset...")
    X_train, X_test, y_train, y_test = load_iris_2d()
    
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    print(f"Class distribution (train): {np.bincount(y_train)}")
    print(f"Class distribution (test): {np.bincount(y_test)}")
    print(f"Feature range: [{X_train.min():.2f}, {X_train.max():.2f}]")
