"""
SVM Training Pipeline

Train SVM with precomputed kernels (classical or quantum)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time


def train_svm_with_kernel(K_train, K_test, y_train, y_test, C=1.0, verbose=True):
    """
    Train SVM with precomputed kernel matrix
    
    Args:
        K_train: Training kernel matrix (n_train, n_train)
        K_test: Test kernel matrix (n_test, n_train)
        y_train: Training labels
        y_test: Test labels
        C: SVM regularization parameter (default: 1.0)
        verbose: Print training info
        
    Returns:
        results: Dictionary with accuracy, f1, confusion_matrix, model, time
        
    Example:
        >>> from kernels.classical import rbf_kernel_wrapper
        >>> K_tr = rbf_kernel_wrapper(X_train)
        >>> K_te = rbf_kernel_wrapper(X_test, X_train)
        >>> results = train_svm_with_kernel(K_tr, K_te, y_train, y_test)
    """
    start_time = time.time()
    
    # Train SVM
    svm = SVC(kernel='precomputed', C=C)
    svm.fit(K_train, y_train)
    
    # Predictions
    y_pred_train = svm.predict(K_train)
    y_pred_test = svm.predict(K_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='binary')
    cm = confusion_matrix(y_test, y_pred_test)
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy:  {test_acc:.4f}")
        print(f"Test F1 score:  {test_f1:.4f}")
        print(f"Training time:  {elapsed_time:.3f}s")
        print(f"Support vectors: {len(svm.support_)}/{len(y_train)}")
    
    results = {
        'model': svm,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'confusion_matrix': cm,
        'time': elapsed_time,
        'n_support_vectors': len(svm.support_)
    }
    
    return results


def compare_kernels(X_train, X_test, y_train, y_test, 
                    classical_kernel_fn, quantum_kernel_fn,
                    C=1.0, verbose=True):
    """
    Compare classical and quantum kernels side-by-side
    
    Args:
        X_train, X_test, y_train, y_test: Train/test data
        classical_kernel_fn: Function that computes classical kernel matrix
        quantum_kernel_fn: Function that computes quantum kernel matrix
        C: SVM parameter
        verbose: Print comparison
        
    Returns:
        results_classical, results_quantum: Two result dictionaries
        
    Example:
        >>> from kernels.classical import rbf_kernel_wrapper
        >>> from kernels.quantum import quantum_kernel_matrix
        >>> 
        >>> classical_fn = lambda X, Y=None: rbf_kernel_wrapper(X, Y, gamma=1.0)
        >>> quantum_fn = lambda X, Y=None: quantum_kernel_matrix(X, Y, n_qubits=2)
        >>> 
        >>> res_cl, res_q = compare_kernels(X_train, X_test, y_train, y_test,
        ...                                 classical_fn, quantum_fn)
    """
    if verbose:
        print("=" * 60)
        print("CLASSICAL KERNEL (RBF)")
        print("=" * 60)
    
    # Classical kernel
    K_train_classical = classical_kernel_fn(X_train)
    K_test_classical = classical_kernel_fn(X_test, X_train)
    
    results_classical = train_svm_with_kernel(
        K_train_classical, K_test_classical, 
        y_train, y_test, C=C, verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("QUANTUM KERNEL")
        print("=" * 60)
    
    # Quantum kernel
    K_train_quantum = quantum_kernel_fn(X_train)
    K_test_quantum = quantum_kernel_fn(X_test, X_train)
    
    results_quantum = train_svm_with_kernel(
        K_train_quantum, K_test_quantum,
        y_train, y_test, C=C, verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"Classical accuracy: {results_classical['test_accuracy']:.4f}")
        print(f"Quantum accuracy:   {results_quantum['test_accuracy']:.4f}")
        
        improvement = (results_quantum['test_accuracy'] - 
                      results_classical['test_accuracy']) * 100
        print(f"Improvement:        {improvement:+.1f} percentage points")
        
        speedup = results_classical['time'] / results_quantum['time']
        print(f"\nTime ratio (classical/quantum): {speedup:.2f}x")
        if speedup < 1:
            print("  → Quantum is faster (simulation)")
        else:
            print("  → Classical is faster (expected on simulator)")
    
    return results_classical, results_quantum


def train_and_evaluate(X_train, X_test, y_train, y_test, 
                       kernel_fn, n_layers=2, n_qubits=4, dataset_name='', C=1.0, verbose=False):
    """
    Train and evaluate a single kernel (for depth experiments)
    
    Args:
        X_train, X_test, y_train, y_test: Train/test data
        kernel_fn: Kernel function (quantum_kernel_matrix or rbf_kernel_wrapper)
        n_layers: Number of layers (for quantum kernel)
        n_qubits: Number of qubits (for quantum kernel)
        dataset_name: Name for logging
        C: SVM parameter
        verbose: Print detailed info
        
    Returns:
        test_accuracy: Test accuracy (float)
        results: Full results dictionary
    """
    # Check if quantum kernel
    is_quantum = 'quantum' in str(kernel_fn.__name__).lower()
    
    # Compute kernel matrices
    if is_quantum:
        K_train = kernel_fn(X_train, n_qubits=n_qubits, n_layers=n_layers)
        K_test = kernel_fn(X_test, X_train, n_qubits=n_qubits, n_layers=n_layers)
    else:
        K_train = kernel_fn(X_train)
        K_test = kernel_fn(X_test, X_train)
    
    # Train SVM
    results = train_svm_with_kernel(K_train, K_test, y_train, y_test, C=C, verbose=verbose)
    
    return results['test_accuracy'], results


if __name__ == "__main__":
    # Quick test with Iris
    from datasets.iris import load_iris_2d
    from kernels.classical import rbf_kernel_wrapper
    from kernels.quantum import quantum_kernel_matrix
    
    print("Loading Iris dataset...")
    X_train, X_test, y_train, y_test = load_iris_2d()
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")
    
    # Define kernel functions
    classical_fn = lambda X, Y=None: rbf_kernel_wrapper(X, Y, gamma=1.0)
    quantum_fn = lambda X, Y=None: quantum_kernel_matrix(X, Y, n_qubits=2, n_layers=2)
    
    # Compare
    results_cl, results_q = compare_kernels(
        X_train, X_test, y_train, y_test,
        classical_fn, quantum_fn,
        verbose=True
    )
