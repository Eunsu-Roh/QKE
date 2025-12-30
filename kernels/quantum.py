"""
Quantum Kernel Functions

Implements quantum feature maps and kernel estimation based on:
"Supervised learning with quantum-enhanced feature spaces" (Nature, 2019)
"""

import pennylane as qml
from pennylane import numpy as np


def create_quantum_kernel(n_qubits=2, n_layers=2, device_name='default.qubit'):
    """
    Create a quantum kernel function using IQP-style embedding
    
    Args:
        n_qubits: Number of qubits (should match feature dimension)
        n_layers: Number of repetitions in IQPEmbedding (default: 2)
        device_name: PennyLane device (default: 'default.qubit')
        
    Returns:
        quantum_kernel: Function that computes K(x1, x2) = |<Φ(x1)|Φ(x2)>|²
        
    Example:
        >>> kernel_fn = create_quantum_kernel(n_qubits=2, n_layers=2)
        >>> K_value = kernel_fn([0.5, 1.0], [1.0, 0.5])
    """
    dev = qml.device(device_name, wires=n_qubits)
    
    @qml.qnode(dev)
    def feature_map(x):
        """
        Quantum feature map: |Φ(x)>
        
        Uses IQPEmbedding (Instantaneous Quantum Polynomial):
        - Hadamard layer
        - RZ rotations (data encoding)
        - ZZ interactions (entanglement)
        
        Paper Eq.(2): U_Φ(x) = exp(i Σ φ_S(x) Π_{i∈S} Z_i)
        """
        # Extend features if needed (for oversampling)
        if len(x) < n_qubits:
            # Repeat features cyclically
            x_extended = [x[i % len(x)] for i in range(n_qubits)]
        else:
            x_extended = x[:n_qubits]
        
        # IQP-style embedding
        qml.templates.IQPEmbedding(
            features=x_extended,
            wires=range(n_qubits),
            n_repeats=n_layers
        )
        
        return qml.state()
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        """
        Kernel measurement circuit: U_Φ(x2) U†_Φ(x1) |0>
        
        Measures probability of |00...0> state
        = |<Φ(x1)|Φ(x2)>|²
        """
        # Extend features if needed
        if len(x1) < n_qubits:
            x1_extended = [x1[i % len(x1)] for i in range(n_qubits)]
            x2_extended = [x2[i % len(x2)] for i in range(n_qubits)]
        else:
            x1_extended = x1[:n_qubits]
            x2_extended = x2[:n_qubits]
        
        # U_Φ(x2)
        qml.templates.IQPEmbedding(
            features=x2_extended,
            wires=range(n_qubits),
            n_repeats=n_layers
        )
        
        # U†_Φ(x1) (adjoint)
        qml.adjoint(qml.templates.IQPEmbedding)(
            features=x1_extended,
            wires=range(n_qubits),
            n_repeats=n_layers
        )
        
        # Measure computational basis
        return qml.probs(wires=range(n_qubits))
    
    def quantum_kernel(x1, x2):
        """
        Compute quantum kernel value K(x1, x2)
        
        Args:
            x1, x2: Feature vectors
            
        Returns:
            Kernel value (float): |<Φ(x1)|Φ(x2)>|²
        """
        # Method 1: Direct state inner product (more stable)
        state1 = feature_map(x1)
        state2 = feature_map(x2)
        kernel_value = np.abs(np.vdot(state1, state2))**2
        
        # Method 2: Measurement probability (paper's method)
        # probs = kernel_circuit(x1, x2)
        # kernel_value = probs[0]  # P(|00...0>)
        
        return float(kernel_value)
    
    return quantum_kernel


def quantum_kernel_matrix(X, Y=None, n_qubits=2, n_layers=2):
    """
    Compute quantum kernel matrix
    
    Args:
        X: Training data (n_samples, n_features)
        Y: Test data (m_samples, n_features), if None uses X
        n_qubits: Number of qubits
        n_layers: Number of IQP layers
        
    Returns:
        K: Kernel matrix (n_samples, m_samples) or (n_samples, n_samples)
        
    Example:
        >>> from datasets.synthetic import generate_xor
        >>> X, y = generate_xor(n_samples=4)
        >>> K = quantum_kernel_matrix(X, n_qubits=2)
        >>> print(K.shape)  # (4, 4)
    """
    if Y is None:
        Y = X
    
    # Create kernel function
    kernel_fn = create_quantum_kernel(n_qubits=n_qubits, n_layers=n_layers)
    
    # Use PennyLane's kernel_matrix utility
    K = qml.kernels.kernel_matrix(X, Y, kernel_fn)
    
    return K


# Alias for consistency with classical kernels
quantum_kernel = quantum_kernel_matrix


if __name__ == "__main__":
    # Quick test
    print("Testing Quantum Kernel...")
    
    # Simple 2D data
    X_test = np.array([[0.0, 0.0],
                       [1.0, 1.0],
                       [0.5, 0.5]])
    
    print(f"Data shape: {X_test.shape}")
    
    # Compute kernel matrix
    K = quantum_kernel_matrix(X_test, n_qubits=2, n_layers=2)
    
    print(f"\nQuantum Kernel Matrix:")
    print(K)
    print(f"\nDiagonal (self-similarity): {np.diag(K)}")
    print("✓ All diagonal elements should be 1.0")
    
    # Test single kernel value
    kernel_fn = create_quantum_kernel(n_qubits=2, n_layers=2)
    k_value = kernel_fn(X_test[0], X_test[1])
    print(f"\nK([0,0], [1,1]) = {k_value:.4f}")
