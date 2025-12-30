"""
Iris Dataset Experiment (Baseline Validation)

Test implementation with Iris dataset.
Expected: Similar performance between classical and quantum
          (양자 이점 없음 - 너무 쉬운 문제)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from datasets.iris import load_iris_2d
from kernels.classical import rbf_kernel_wrapper
from kernels.quantum import quantum_kernel_matrix
from train import compare_kernels
from visualization.plots import plot_kernel_matrix

# Create results directory
os.makedirs('../results', exist_ok=True)


def main():
    print("=" * 70)
    print("EXPERIMENT: Iris Dataset (Baseline Validation)")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading Iris dataset...")
    X_train, X_test, y_train, y_test = load_iris_2d(test_size=0.2, random_state=42)
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
    print()
    
    # Define kernel functions
    print("Setting up kernels...")
    print("  - Classical: RBF kernel (gamma=1.0)")
    print("  - Quantum: IQP embedding (2 qubits, 2 layers)")
    print()
    
    classical_kernel_fn = lambda X, Y=None: rbf_kernel_wrapper(X, Y, gamma=1.0)
    quantum_kernel_fn = lambda X, Y=None: quantum_kernel_matrix(X, Y, n_qubits=2, n_layers=2)
    
    # Compare kernels
    results_classical, results_quantum = compare_kernels(
        X_train, X_test, y_train, y_test,
        classical_kernel_fn, quantum_kernel_fn,
        C=1.0, verbose=True
    )
    
    # Visualize kernel matrices
    print("\nGenerating kernel matrix visualizations...")
    
    K_classical = classical_kernel_fn(X_train[:30])
    K_quantum = quantum_kernel_fn(X_train[:30])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_kernel_matrix(K_classical, title="RBF Kernel Matrix (30x30)", 
                      ax=axes[0], cmap='viridis')
    plot_kernel_matrix(K_quantum, title="Quantum Kernel Matrix (30x30)",
                      ax=axes[1], cmap='plasma')
    
    plt.tight_layout()
    save_path = '../results/iris_kernel_matrices.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    # Save results
    print("\nSaving results...")
    results_file = '../results/iris_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("IRIS DATASET EXPERIMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        
        f.write("CLASSICAL (RBF) KERNEL:\n")
        f.write(f"  Accuracy: {results_classical['test_accuracy']:.4f}\n")
        f.write(f"  F1 Score: {results_classical['test_f1']:.4f}\n")
        f.write(f"  Time: {results_classical['time']:.3f}s\n\n")
        
        f.write("QUANTUM KERNEL:\n")
        f.write(f"  Accuracy: {results_quantum['test_accuracy']:.4f}\n")
        f.write(f"  F1 Score: {results_quantum['test_f1']:.4f}\n")
        f.write(f"  Time: {results_quantum['time']:.3f}s\n\n")
        
        improvement = (results_quantum['test_accuracy'] - 
                      results_classical['test_accuracy']) * 100
        f.write(f"IMPROVEMENT: {improvement:+.2f} percentage points\n")
        f.write("\nCONCLUSION:\n")
        f.write("Iris is too easy - no quantum advantage expected.\n")
        f.write("This serves as implementation validation.\n")
    
    print(f"✓ Saved: {results_file}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return results_classical, results_quantum


if __name__ == "__main__":
    results_classical, results_quantum = main()
