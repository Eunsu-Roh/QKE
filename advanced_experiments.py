"""
Advanced Experiments for Complex Datasets

Test deeper circuits for SPIRALS and CHECKERBOARD
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from datasets.synthetic import generate_spirals, generate_checkerboard, prepare_dataset
from kernels.classical import rbf_kernel_wrapper
from kernels.quantum import quantum_kernel_matrix
from train import compare_kernels


def run_depth_experiment(dataset_name, X_train, X_test, y_train, y_test, 
                         layer_configs=[2, 4, 6, 8], timestamp=''):
    """
    Run experiments with varying circuit depths
    
    Args:
        dataset_name: 'spirals' or 'checkerboard'
        X_train, X_test, y_train, y_test: Dataset
        layer_configs: List of n_layers to test
        timestamp: For file naming
    """
    print("\n" + "=" * 70)
    print(f"DEPTH EXPERIMENT: {dataset_name.upper()}")
    print("=" * 70)
    
    results = {
        'layers': [],
        'quantum_train': [],
        'quantum_test': [],
        'rbf_train': [],
        'rbf_test': []
    }
    
    # Classical baseline (once)
    print(f"\n--- Classical RBF Kernel (Baseline) ---")
    rbf_scores = compare_kernels(
        X_train, X_test, y_train, y_test,
        rbf_kernel_wrapper,
        quantum_kernel_matrix,
        n_qubits=4,
        n_layers=2,
        dataset_name=dataset_name,
        timestamp=timestamp
    )
    
    baseline_rbf_train = rbf_scores['classical']['train']
    baseline_rbf_test = rbf_scores['classical']['test']
    
    print(f"Classical RBF - Train: {baseline_rbf_train:.1%}, Test: {baseline_rbf_test:.1%}")
    
    # Test different quantum depths
    for n_layers in layer_configs:
        print(f"\n--- Quantum Circuit: {n_layers} Layers ---")
        print(f"Gates: {n_layers * 14} (={n_layers}×[4H + 4RZ + 6MultiRZ])")
        
        scores = compare_kernels(
            X_train, X_test, y_train, y_test,
            rbf_kernel_wrapper,
            quantum_kernel_matrix,
            n_qubits=4,
            n_layers=n_layers,
            dataset_name=f"{dataset_name}_layer{n_layers}",
            timestamp=timestamp,
            save_plots=False  # Don't save individual plots
        )
        
        results['layers'].append(n_layers)
        results['quantum_train'].append(scores['quantum']['train'])
        results['quantum_test'].append(scores['quantum']['test'])
        results['rbf_train'].append(baseline_rbf_train)
        results['rbf_test'].append(baseline_rbf_test)
        
        print(f"Quantum (L={n_layers}) - Train: {scores['quantum']['train']:.1%}, "
              f"Test: {scores['quantum']['test']:.1%}")
    
    # Plot results
    plot_depth_comparison(results, dataset_name, timestamp)
    
    return results


def plot_depth_comparison(results, dataset_name, timestamp=''):
    """Plot performance vs circuit depth"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = results['layers']
    
    # Quantum performance
    ax.plot(layers, results['quantum_train'], 'o-', label='Quantum (Train)', 
            color='blue', linewidth=2, markersize=8)
    ax.plot(layers, results['quantum_test'], 's-', label='Quantum (Test)', 
            color='darkblue', linewidth=2, markersize=8)
    
    # Classical baseline
    ax.axhline(results['rbf_train'][0], linestyle='--', label='Classical RBF (Train)', 
               color='red', linewidth=2)
    ax.axhline(results['rbf_test'][0], linestyle='--', label='Classical RBF (Test)', 
               color='darkred', linewidth=2)
    
    ax.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Circuit Depth vs Performance: {dataset_name.upper()}\n' + 
                 'Effect of Increasing Polynomial Features',
                 fontsize=13, fontweight='bold')
    
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    
    # Add annotations
    for i, (l, train, test) in enumerate(zip(layers, results['quantum_train'], 
                                               results['quantum_test'])):
        gates = l * 14
        ax.annotate(f'{gates}g', (l, test), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8, color='darkblue')
    
    plt.tight_layout()
    
    filename = f'results/depth_comparison_{dataset_name}{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    plt.close()


def run_all_advanced_experiments(timestamp=''):
    """Run depth experiments for SPIRALS and CHECKERBOARD"""
    print("\n" + "=" * 70)
    print("ADVANCED EXPERIMENTS: Testing Circuit Depth")
    print("=" * 70)
    print("\nGoal: Improve quantum kernel performance on complex datasets")
    print("Method: Increase n_layers → More polynomial features")
    print("Expected: Higher-order interactions help non-linear patterns")
    
    # SPIRALS
    print("\n" + "=" * 70)
    print("Dataset 1: SPIRALS")
    print("=" * 70)
    X, y = generate_spirals(n_samples=300, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = prepare_dataset(X, y, test_size=0.2)
    
    spirals_results = run_depth_experiment(
        'spirals', X_train, X_test, y_train, y_test,
        layer_configs=[2, 4, 6, 8],
        timestamp=timestamp
    )
    
    # CHECKERBOARD
    print("\n" + "=" * 70)
    print("Dataset 2: CHECKERBOARD")
    print("=" * 70)
    X, y = generate_checkerboard(n_samples=200, random_state=42)
    X_train, X_test, y_train, y_test = prepare_dataset(X, y, test_size=0.2)
    
    checker_results = run_depth_experiment(
        'checkerboard', X_train, X_test, y_train, y_test,
        layer_configs=[2, 4, 6, 8],
        timestamp=timestamp
    )
    
    # Summary comparison
    plot_combined_summary(spirals_results, checker_results, timestamp)
    
    return spirals_results, checker_results


def plot_combined_summary(spirals_results, checker_results, timestamp=''):
    """Plot side-by-side comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    datasets = [
        ('SPIRALS', spirals_results, axes[0]),
        ('CHECKERBOARD', checker_results, axes[1])
    ]
    
    for name, results, ax in datasets:
        layers = results['layers']
        
        ax.plot(layers, results['quantum_test'], 'o-', label='Quantum', 
                color='blue', linewidth=2.5, markersize=10)
        ax.axhline(results['rbf_test'][0], linestyle='--', label='Classical RBF', 
                   color='red', linewidth=2.5)
        
        ax.set_xlabel('Layers', fontsize=11, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks(layers)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Best quantum result
        best_idx = np.argmax(results['quantum_test'])
        best_layer = layers[best_idx]
        best_acc = results['quantum_test'][best_idx]
        ax.plot(best_layer, best_acc, 'g*', markersize=15, 
                label=f'Best: L={best_layer}')
    
    plt.suptitle('Circuit Depth Impact on Complex Datasets\n' + 
                 'Testing if more layers improve quantum kernel performance',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f'results/advanced_summary{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("ADVANCED QUANTUM KERNEL EXPERIMENTS")
    print("=" * 70)
    print("\nTesting hypothesis:")
    print("  - Basic circuits (2 layers) work well for simple datasets")
    print("  - Complex datasets need deeper circuits (more layers)")
    print("  - More layers → Higher-order polynomial features")
    
    spirals_results, checker_results = run_all_advanced_experiments(timestamp)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nSPIRALS:")
    for i, l in enumerate(spirals_results['layers']):
        print(f"  L={l}: {spirals_results['quantum_test'][i]:.1%} " + 
              f"(Classical: {spirals_results['rbf_test'][0]:.1%})")
    
    print("\nCHECKERBOARD:")
    for i, l in enumerate(checker_results['layers']):
        print(f"  L={l}: {checker_results['quantum_test'][i]:.1%} " + 
              f"(Classical: {checker_results['rbf_test'][0]:.1%})")
    
    print("\n✓ Advanced experiments complete!")
