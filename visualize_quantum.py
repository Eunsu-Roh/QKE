"""
Quantum Circuit and Bloch Sphere Visualization

Visualize the quantum kernel's circuit structure and qubit states
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_quantum_kernel_circuit(n_qubits=2, n_layers=2):
    """
    Draw the quantum kernel circuit used in experiments
    
    Shows IQPEmbedding structure with Hadamard, RZ rotations, and ZZ entanglement
    """
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(features):
        """Quantum feature map circuit"""
        qml.templates.IQPEmbedding(
            features=features,
            wires=range(n_qubits),
            n_repeats=n_layers
        )
        return qml.state()
    
    # Example features
    example_features = [0.5, 1.0] if n_qubits == 2 else [0.5, 1.0, 0.3, 0.8]
    
    # Draw circuit
    fig, ax = qml.draw_mpl(circuit, decimals=2)(example_features)
    
    plt.suptitle(f'Quantum Kernel Circuit (IQPEmbedding)\n{n_qubits} qubits, {n_layers} layers', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Add explanation
    explanation = (
        f"Features: {example_features}\n"
        f"Structure: Hadamard → RZ(features) → ZZ entanglement (repeated {n_layers}×)"
    )
    plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return fig


def draw_bloch_sphere_superposition():
    """
    Draw Bloch sphere showing superposition states
    
    Visualizes how quantum states exist in superposition
    """
    fig = plt.figure(figsize=(14, 5))
    
    # Bloch sphere 1: |0⟩ state
    ax1 = fig.add_subplot(131, projection='3d')
    draw_single_bloch(ax1, state='0', title='|0⟩ State (Ground)')
    
    # Bloch sphere 2: |+⟩ = H|0⟩ (superposition)
    ax2 = fig.add_subplot(132, projection='3d')
    draw_single_bloch(ax2, state='+', title='|+⟩ = (|0⟩+|1⟩)/√2\nSuperposition')
    
    # Bloch sphere 3: General state after encoding
    ax3 = fig.add_subplot(133, projection='3d')
    draw_single_bloch(ax3, state='general', title='After RZ(θ)\nEncoded State')
    
    plt.suptitle('Quantum Superposition on Bloch Sphere', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def draw_single_bloch(ax, state='0', title=''):
    """Draw a single Bloch sphere with a state vector"""
    
    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                    alpha=0.1, color='cyan', edgecolor='none')
    
    # Draw axes
    ax.plot([-1.3, 1.3], [0, 0], [0, 0], 'k-', alpha=0.3, linewidth=1)
    ax.plot([0, 0], [-1.3, 1.3], [0, 0], 'k-', alpha=0.3, linewidth=1)
    ax.plot([0, 0], [0, 0], [-1.3, 1.3], 'k-', alpha=0.3, linewidth=1)
    
    # Axis labels
    ax.text(1.5, 0, 0, 'X', fontsize=12, fontweight='bold')
    ax.text(0, 1.5, 0, 'Y', fontsize=12, fontweight='bold')
    ax.text(0, 0, 1.5, '|0⟩', fontsize=12, fontweight='bold')
    ax.text(0, 0, -1.5, '|1⟩', fontsize=12, fontweight='bold')
    
    # Draw state vector
    if state == '0':
        # |0⟩ at north pole
        ax.quiver(0, 0, 0, 0, 0, 1, color='red', arrow_length_ratio=0.15, linewidth=3)
        ax.text(0, 0, 1.3, 'Pure |0⟩', fontsize=10, ha='center', color='red')
        
    elif state == '+':
        # |+⟩ on equator (X-axis)
        ax.quiver(0, 0, 0, 1, 0, 0, color='green', arrow_length_ratio=0.15, linewidth=3)
        ax.text(1.2, 0, 0, 'Superposition', fontsize=10, ha='left', color='green')
        
        # Show both |0⟩ and |1⟩ components (dashed)
        ax.plot([0, 0], [0, 0], [0, 1], 'r--', alpha=0.3, linewidth=1)
        ax.plot([0, 0], [0, 0], [0, -1], 'b--', alpha=0.3, linewidth=1)
        ax.text(0.1, 0.1, 0.5, '|0⟩\ncomponent', fontsize=8, color='red', alpha=0.7)
        ax.text(0.1, 0.1, -0.5, '|1⟩\ncomponent', fontsize=8, color='blue', alpha=0.7)
        
    elif state == 'general':
        # General state after RZ rotation (example: θ=π/4)
        theta = np.pi / 4
        x = np.sin(theta)
        z = np.cos(theta)
        ax.quiver(0, 0, 0, x, 0, z, color='purple', arrow_length_ratio=0.15, linewidth=3)
        ax.text(x*1.3, 0, z*1.3, f'cos(θ/2)|0⟩\n+sin(θ/2)|1⟩', 
                fontsize=9, ha='center', color='purple')
    
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_box_aspect([1,1,1])
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def visualize_kernel_computation():
    """
    Visualize how quantum kernel computes similarity
    
    Shows the two-state overlap |⟨ψ|φ⟩|²
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
    
    # State 1: |ψ(x₁)⟩
    ax1 = axes[0]
    draw_single_bloch(ax1, state='general', title='State |ψ(x₁)⟩\nFeature x₁ encoded')
    theta1 = np.pi / 4
    x1, z1 = np.sin(theta1), np.cos(theta1)
    ax1.quiver(0, 0, 0, x1, 0, z1, color='red', arrow_length_ratio=0.15, linewidth=3)
    
    # State 2: |φ(x₂)⟩
    ax2 = axes[1]
    draw_single_bloch(ax2, state='general', title='State |φ(x₂)⟩\nFeature x₂ encoded')
    theta2 = np.pi / 3
    x2, z2 = np.sin(theta2), np.cos(theta2)
    ax2.quiver(0, 0, 0, x2, 0, z2, color='blue', arrow_length_ratio=0.15, linewidth=3)
    
    # Overlap visualization
    ax3 = axes[2]
    draw_single_bloch(ax3, state='0', title='Kernel K(x₁,x₂)\n|⟨ψ(x₁)|φ(x₂)⟩|²')
    ax3.quiver(0, 0, 0, x1, 0, z1, color='red', arrow_length_ratio=0.15, 
               linewidth=2, alpha=0.6, label='|ψ(x₁)⟩')
    ax3.quiver(0, 0, 0, x2, 0, z2, color='blue', arrow_length_ratio=0.15, 
               linewidth=2, alpha=0.6, label='|φ(x₂)⟩')
    
    # Draw angle between vectors
    angle = theta2 - theta1
    overlap = np.cos(angle/2)**2
    ax3.text(0, 0, -1.8, f'Overlap = {overlap:.3f}\n(closer → higher similarity)', 
             fontsize=10, ha='center', 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Quantum Kernel Computation: Inner Product of Quantum States', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    import os
    os.makedirs('results/quantum_viz', exist_ok=True)
    
    print("=" * 70)
    print("QUANTUM CIRCUIT & STATE VISUALIZATION")
    print("=" * 70)
    
    # 1. Quantum circuit for 4-qubit (ALL datasets use 4 qubits)
    print("\n1. Drawing 4-qubit quantum kernel circuit...")
    print("   (Used for ALL datasets: Iris, XOR, CIRCLES, SPIRALS, CHECKERBOARD)")
    fig1 = draw_quantum_kernel_circuit(n_qubits=4, n_layers=2)
    fig1.savefig('results/quantum_viz/circuit_4qubits.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/quantum_viz/circuit_4qubits.png")
    plt.close(fig1)
    
    # 2. Bloch sphere superposition
    print("\n2. Drawing Bloch sphere (superposition)...")
    fig2 = draw_bloch_sphere_superposition()
    fig2.savefig('results/quantum_viz/bloch_sphere_superposition.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/quantum_viz/bloch_sphere_superposition.png")
    plt.close(fig2)
    
    # 3. Kernel computation visualization
    print("\n3. Drawing kernel computation (state overlap)...")
    fig3 = visualize_kernel_computation()
    fig3.savefig('results/quantum_viz/kernel_computation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/quantum_viz/kernel_computation.png")
    plt.close(fig3)
    
    print("\n" + "=" * 70)
    print("✓ All quantum visualizations saved in: results/quantum_viz/")
    print("=" * 70)
