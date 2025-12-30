"""
Visualization Functions

Plot decision boundaries, kernel matrices, and performance comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay


def plot_decision_boundary(X, y, model, title="Decision Boundary", 
                           ax=None, resolution=0.02):
    """
    Plot decision boundary for 2D classification
    
    Args:
        X: Data (n_samples, 2)
        y: Labels (n_samples,)
        model: Trained SVM model
        title: Plot title
        ax: Matplotlib axis (creates new if None)
        resolution: Mesh grid resolution
        
    Returns:
        ax: Matplotlib axis
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_decision_boundary(X_test, y_test, svm_model, ax=ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Predict on mesh (note: needs kernel matrix for precomputed kernel)
    # For visualization, we'll just plot the data points
    # Decision boundary requires special handling for precomputed kernels
    
    # Plot data points
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                        edgecolor='black', s=100, linewidth=1.5, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1 (x₁)', fontsize=12)
    ax.set_ylabel('Feature 2 (x₂)', fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add legend
    handles, labels = scatter.legend_elements()
    ax.legend(handles, ['Class 0', 'Class 1'], loc='best', fontsize=10)
    
    return ax


def plot_decision_boundary_with_kernel(X, y, X_train, kernel_fn, model,
                                       title="Decision Boundary", ax=None,
                                       resolution=0.02):
    """
    Plot decision boundary using kernel function (works with precomputed)
    
    Args:
        X: Test data for plotting
        y: Test labels
        X_train: Training data (needed for kernel computation)
        kernel_fn: Kernel function that takes (X_new, X_train)
        model: Trained SVM model
        title: Plot title
        ax: Matplotlib axis
        resolution: Grid resolution
        
    Returns:
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Predict on mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    K_mesh = kernel_fn(mesh_points, X_train)
    Z = model.predict(K_mesh)
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light, levels=1)
    
    # Plot data points
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                        edgecolor='black', s=100, linewidth=1.5, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1 (x₁)', fontsize=12)
    ax.set_ylabel('Feature 2 (x₂)', fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Legend
    handles, labels = scatter.legend_elements()
    ax.legend(handles, ['Class 0', 'Class 1'], loc='best', fontsize=10)
    
    return ax


def plot_kernel_matrix(K, title="Kernel Matrix", ax=None, cmap='viridis'):
    """
    Plot kernel matrix as heatmap (논문 Fig. 4 스타일)
    
    Args:
        K: Kernel matrix (n, n)
        title: Plot title
        ax: Matplotlib axis
        cmap: Colormap
        
    Returns:
        ax: Matplotlib axis
        
    Example:
        >>> K_rbf = rbf_kernel_wrapper(X_train)
        >>> plot_kernel_matrix(K_rbf, title="RBF Kernel Matrix")
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(K, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Kernel Value', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    
    # Show values for small matrices
    if K.shape[0] <= 10:
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                text = ax.text(j, i, f'{K[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
    
    return ax


def plot_performance_comparison(results_dict, title="Performance Comparison", 
                                save_path=None):
    """
    Plot bar chart comparing accuracies across datasets
    
    Args:
        results_dict: Dictionary like:
            {
                'Iris': {'classical': 0.97, 'quantum': 0.96},
                'XOR': {'classical': 0.75, 'quantum': 1.0},
                ...
            }
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        fig, ax
        
    Example:
        >>> results = {
        ...     'Iris': {'classical': 0.97, 'quantum': 0.96},
        ...     'XOR': {'classical': 0.75, 'quantum': 1.0}
        ... }
        >>> plot_performance_comparison(results)
    """
    datasets = list(results_dict.keys())
    classical_scores = [results_dict[d]['classical'] for d in datasets]
    quantum_scores = [results_dict[d]['quantum'] for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, classical_scores, width, label='Classical (RBF)',
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, quantum_scores, width, label='Quantum Kernel',
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0%}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax


def plot_datasets_overview(datasets_dict, save_path=None):
    """
    Plot 4 datasets in one figure (for presentation)
    
    Args:
        datasets_dict: Dictionary like:
            {
                'XOR': (X, y),
                'Circles': (X, y),
                ...
            }
        save_path: Path to save
        
    Returns:
        fig, axes
    """
    n_datasets = len(datasets_dict)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4*n_datasets, 4))
    
    if n_datasets == 1:
        axes = [axes]
    
    for ax, (name, (X, y)) in zip(axes, datasets_dict.items()):
        ax.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Class 0',
                  s=50, edgecolor='black', linewidth=1, alpha=0.7)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Class 1',
                  s=50, edgecolor='black', linewidth=1, alpha=0.7)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('x₁', fontsize=11)
        ax.set_ylabel('x₂', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Non-linear Classification Datasets', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, axes


if __name__ == "__main__":
    # Test visualizations
    from datasets.synthetic import generate_xor, generate_concentric_circles
    from kernels.classical import rbf_kernel_wrapper
    
    print("Testing visualizations...")
    
    # Generate data
    X, y = generate_xor(n_samples=100)
    
    # Kernel matrix
    K = rbf_kernel_wrapper(X, gamma=1.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dataset
    axes[0].scatter(X[y==0, 0], X[y==0, 1], c='red', s=50, label='Class 0')
    axes[0].scatter(X[y==1, 0], X[y==1, 1], c='blue', s=50, label='Class 1')
    axes[0].set_title("XOR Dataset")
    axes[0].legend()
    
    # Kernel matrix
    plot_kernel_matrix(K[:20, :20], title="RBF Kernel Matrix (20x20)", ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('test_visualization.png', dpi=300)
    print("✓ Saved test_visualization.png")
