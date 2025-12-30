"""
Enhanced Quantum Kernel Experiments

SPIRALS와 CHECKERBOARD를 위한 향상된 양자 커널 실험:
- 더 많은 layers (depth 증가)
- 기존 2 layers vs 5 layers vs 10 layers 비교
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.synthetic import generate_spirals, generate_checkerboard
from kernels.quantum import quantum_kernel_matrix
from kernels.classical import rbf_kernel_wrapper


def experiment_enhanced_quantum(dataset_name, X, y, n_qubits=2):
    """
    Enhanced quantum kernel experiment with varying depth
    
    Args:
        dataset_name: 'SPIRALS' or 'CHECKERBOARD'
        X: Features
        y: Labels
        n_qubits: Number of qubits (should match features)
    
    Returns:
        results: Dictionary with accuracies for each configuration
    """
    print(f"\n{'='*70}")
    print(f"ENHANCED QUANTUM KERNEL EXPERIMENT: {dataset_name}")
    print(f"{'='*70}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    results = {}
    
    # 1. Classical RBF (baseline)
    print(f"\n1. Classical RBF Kernel...")
    K_train_rbf = rbf_kernel_wrapper(X_train, X_train, gamma=1.0)
    K_test_rbf = rbf_kernel_wrapper(X_test, X_train, gamma=1.0)
    
    svm_rbf = SVC(kernel='precomputed')
    svm_rbf.fit(K_train_rbf, y_train)
    y_pred_rbf = svm_rbf.predict(K_test_rbf)
    acc_rbf = accuracy_score(y_test, y_pred_rbf)
    results['Classical RBF'] = acc_rbf
    print(f"   Accuracy: {acc_rbf*100:.2f}%")
    
    # 2. Quantum with different layers
    layer_configs = [2, 5, 10]
    
    for n_layers in layer_configs:
        print(f"\n2. Quantum IQP Kernel (n_layers={n_layers})...")
        print(f"   Computing kernel matrix (this may take time)...")
        
        # Compute quantum kernels
        K_train_q = quantum_kernel_matrix(X_train, X_train, 
                                          n_qubits=n_qubits, 
                                          n_layers=n_layers)
        K_test_q = quantum_kernel_matrix(X_test, X_train, 
                                         n_qubits=n_qubits, 
                                         n_layers=n_layers)
        
        # Train SVM
        svm_q = SVC(kernel='precomputed')
        svm_q.fit(K_train_q, y_train)
        y_pred_q = svm_q.predict(K_test_q)
        acc_q = accuracy_score(y_test, y_pred_q)
        
        results[f'Quantum (layers={n_layers})'] = acc_q
        print(f"   Accuracy: {acc_q*100:.2f}%")
        print(f"   Gates: {n_layers * 14} (14 gates per layer)")
        
        # Estimate depth
        depth = 4 + (n_layers - 1) * 3  # Approximate
        print(f"   Estimated depth: ~{depth}")
    
    return results


def plot_comparison(results_spirals, results_checkerboard, save_path):
    """Plot comparison of different quantum configurations"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SPIRALS
    ax1 = axes[0]
    methods = list(results_spirals.keys())
    accuracies = [results_spirals[m]*100 for m in methods]
    colors = ['skyblue'] + ['salmon', 'coral', 'crimson']
    
    bars1 = ax1.bar(range(len(methods)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('SPIRALS Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # CHECKERBOARD
    ax2 = axes[1]
    methods = list(results_checkerboard.keys())
    accuracies = [results_checkerboard[m]*100 for m in methods]
    
    bars2 = ax2.bar(range(len(methods)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('CHECKERBOARD Dataset', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Enhanced Quantum Kernel Performance\n' +
                 'Effect of Increased Depth (More Layers)',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved comparison plot: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Create output directory
    os.makedirs('results/enhanced', exist_ok=True)
    
    print("="*70)
    print("ENHANCED QUANTUM KERNEL EXPERIMENTS")
    print("="*70)
    print("\n목적: SPIRALS와 CHECKERBOARD에서 더 깊은 양자 회로의 효과 확인")
    print("비교: 2 layers (기본) vs 5 layers vs 10 layers")
    print("이론: 더 많은 layer → 더 복잡한 polynomial features → 더 나은 성능")
    
    # Generate datasets
    print("\n" + "="*70)
    print("데이터 생성...")
    print("="*70)
    
    X_spirals, y_spirals = generate_spirals(n_samples=300, noise=0.1)
    print(f"SPIRALS: {X_spirals.shape[0]} samples, {X_spirals.shape[1]} features")
    
    X_checker, y_checker = generate_checkerboard(n_samples=200, grid_size=4)
    print(f"CHECKERBOARD: {X_checker.shape[0]} samples, {X_checker.shape[1]} features")
    
    # Experiments
    results_spirals = experiment_enhanced_quantum('SPIRALS', X_spirals, y_spirals, n_qubits=2)
    results_checkerboard = experiment_enhanced_quantum('CHECKERBOARD', X_checker, y_checker, n_qubits=2)
    
    # Plot comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'results/enhanced/enhanced_comparison_{timestamp}.png'
    plot_comparison(results_spirals, results_checkerboard, plot_path)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nSPIRALS:")
    for method, acc in results_spirals.items():
        print(f"  {method:30s}: {acc*100:6.2f}%")
    
    print("\nCHECKERBOARD:")
    for method, acc in results_checkerboard.items():
        print(f"  {method:30s}: {acc*100:6.2f}%")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    improvement_spirals = (results_spirals['Quantum (layers=10)'] - 
                          results_spirals['Quantum (layers=2)']) * 100
    improvement_checker = (results_checkerboard['Quantum (layers=10)'] - 
                          results_checkerboard['Quantum (layers=2)']) * 100
    
    print(f"\nLayer 증가 효과 (2 → 10):")
    print(f"  SPIRALS: {improvement_spirals:+.2f}% points")
    print(f"  CHECKERBOARD: {improvement_checker:+.2f}% points")
    
    print("\n결론:")
    if improvement_spirals > 5:
        print("  ✓ SPIRALS에서 depth 증가가 효과적!")
    else:
        print("  ✗ SPIRALS에서 depth 증가만으로는 부족")
        print("    → 다른 entanglement 패턴 필요")
    
    if improvement_checker > 5:
        print("  ✓ CHECKERBOARD에서 depth 증가가 효과적!")
    else:
        print("  ✗ CHECKERBOARD에서 depth 증가만으로는 부족")
        print("    → 문제가 너무 복잡 (다중 영역 분리)")
    
    print("\n" + "="*70)
    print("✓ Enhanced quantum kernel experiments completed!")
    print("="*70)
