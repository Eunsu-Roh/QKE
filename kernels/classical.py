"""
Classical Kernel Functions

Wrapper for scikit-learn RBF kernel for consistent interface with quantum kernels.
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def rbf_kernel_wrapper(X, Y=None, gamma=1.0):
    """
    RBF (Radial Basis Function / Gaussian) Kernel
    
    K(x, z) = exp(-gamma * ||x - z||²)
    
    Args:
        X: Training data (n_samples, n_features)
        Y: Test data (m_samples, n_features), if None uses X
        gamma: Kernel coefficient (default: 1.0)
        
    Returns:
        Kernel matrix (n_samples, m_samples) or (n_samples, n_samples)
        
    Example:
        >>> X = np.array([[0, 0], [1, 1]])
        >>> K = rbf_kernel_wrapper(X, gamma=1.0)
        >>> print(K)
        [[1.0, 0.14]
         [0.14, 1.0]]
    """
    if Y is None:
        Y = X
    return rbf_kernel(X, Y, gamma=gamma)


def compute_kernel_matrix(X, Y=None, kernel_type='rbf', **kwargs):
    """
    Unified interface for computing classical kernel matrices
    
    Args:
        X: Training data
        Y: Test data (optional)
        kernel_type: Only 'rbf' supported
        **kwargs: Kernel-specific parameters (gamma for RBF)
        
    Returns:
        Kernel matrix
        
    Example:
        >>> K_rbf = compute_kernel_matrix(X, kernel_type='rbf', gamma=1.0)
    """
    if kernel_type != 'rbf':
        raise ValueError(f"Only 'rbf' kernel is supported. Got: {kernel_type}")
    
    return rbf_kernel_wrapper(X, Y, **kwargs)
