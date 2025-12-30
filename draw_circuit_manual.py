"""
Quantum Circuit Visualization

Draw IQPEmbedding circuit with exact structure using matplotlib
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['mathtext.fontset'] = 'dejavusans'


def draw_iqp_circuit_manual(n_qubits=4, n_layers=2, save_path='results/quantum_viz/iqp_circuit.png'):
    """
    Draw IQPEmbedding circuit with exact structure
    
    Each layer:
    - 4 Hadamard gates
    - 4 RZ gates  
    - 6 MultiRZ gates (all qubit pairs)
    
    Total: 28 gates, depth 13
    """
    fig, ax = plt.subplots(figsize=(20, 7))
    
    wire_length = 22
    wire_spacing = 1.0
    gate_size = 0.45
    
    # Draw wires
    for i in range(n_qubits):
        y = (n_qubits - 1 - i) * wire_spacing
        ax.plot([0, wire_length], [y, y], 'k-', linewidth=1.5, zorder=1)
        
        # Qubit labels
        ax.text(-0.5, y, f'$|0\\rangle$', fontsize=11, ha='right', va='center')
        ax.text(-1.0, y, f'q{i}', fontsize=12, ha='right', va='center', fontweight='bold')
        ax.text(wire_length + 0.3, y, '$|\\psi\\rangle$', fontsize=11, ha='left', va='center')
    
    x_pos = 1.0
    
    # Layer 1
    layer_y = n_qubits * wire_spacing + 0.3
    draw_label(ax, x_pos + 4.5, layer_y, 'Layer 1', color='darkblue')
    
    # H gates
    for i in range(n_qubits):
        y = (n_qubits - 1 - i) * wire_spacing
        draw_gate(ax, x_pos, y, 'H', gate_size, color='skyblue')
    x_pos += 1.2
    
    # RZ gates
    for i in range(n_qubits):
        y = (n_qubits - 1 - i) * wire_spacing
        draw_gate(ax, x_pos, y, 'RZ', gate_size, color='wheat')
    x_pos += 1.2
    
    # MultiRZ gates (6 pairs)
    multirz_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for idx, (i, j) in enumerate(multirz_pairs):
        x = x_pos + idx * 0.85
        y_i = (n_qubits - 1 - i) * wire_spacing
        y_j = (n_qubits - 1 - j) * wire_spacing
        draw_multirz(ax, x, y_i, y_j)
    x_pos += len(multirz_pairs) * 0.85 + 0.8
    
    # Layer 2
    draw_label(ax, x_pos + 4.5, layer_y, 'Layer 2', color='darkblue')
    
    # H gates
    for i in range(n_qubits):
        y = (n_qubits - 1 - i) * wire_spacing
        draw_gate(ax, x_pos, y, 'H', gate_size, color='skyblue')
    x_pos += 1.2
    
    # RZ gates
    for i in range(n_qubits):
        y = (n_qubits - 1 - i) * wire_spacing
        draw_gate(ax, x_pos, y, 'RZ', gate_size, color='wheat')
    x_pos += 1.2
    
    # MultiRZ gates (6 pairs)
    for idx, (i, j) in enumerate(multirz_pairs):
        x = x_pos + idx * 0.85
        y_i = (n_qubits - 1 - i) * wire_spacing
        y_j = (n_qubits - 1 - j) * wire_spacing
        draw_multirz(ax, x, y_i, y_j)
    
    # Legend
    legend_x = 1.0
    legend_y = -1.2
    
    ax.text(legend_x, legend_y, 'Gates:', fontsize=11, fontweight='bold')
    
    draw_gate(ax, legend_x + 1.0, legend_y, 'H', 0.35, color='skyblue')
    ax.text(legend_x + 1.5, legend_y, r'Hadamard: $\frac{|0\rangle+|1\rangle}{\sqrt{2}}$', 
            fontsize=9, va='center')
    
    draw_gate(ax, legend_x + 5.5, legend_y, 'RZ', 0.35, color='wheat')
    ax.text(legend_x + 6.0, legend_y, r'Rotation: $R_Z(\theta) = e^{-i\theta Z/2}$', 
            fontsize=9, va='center')
    
    draw_multirz(ax, legend_x + 10.5, legend_y, legend_y - 0.3, small=True)
    ax.text(legend_x + 11.0, legend_y - 0.15, r'MultiRZ: $e^{-i\theta Z_i Z_j}$ (symmetric entanglement)', 
            fontsize=9, va='center')
    
    # Title
    title = 'IQP Embedding Circuit (4 Qubits, 2 Layers)'
    ax.text(wire_length/2, n_qubits * wire_spacing + 1.0, title,
            fontsize=14, ha='center', fontweight='bold')
    
    # Circuit info
    info = 'Structure: [4H + 4RZ + 6MultiRZ] × 2 layers = 28 gates | Depth: 13 | Device: default.qubit (ideal, no noise)'
    ax.text(wire_length/2, -1.8, info, fontsize=9, ha='center', style='italic', color='gray')
    
    ax.set_xlim(-1.5, wire_length + 1.0)
    ax.set_ylim(-2.2, n_qubits * wire_spacing + 1.5)
    ax.axis('off')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    print(f"  Circuit structure: {n_qubits} qubits, {n_layers} layers")
    print(f"  Gates: {n_layers * 14} total ({n_layers}×[4H + 4RZ + 6MultiRZ])")
    print(f"  Depth: 13")
    plt.close()


def draw_gate(ax, x, y, label, size, color='lightblue'):
    """Draw a quantum gate box"""
    rect = FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.03",
        edgecolor='black', facecolor=color, linewidth=1.5, zorder=10
    )
    ax.add_patch(rect)
    ax.text(x, y, label, fontsize=9, ha='center', va='center', fontweight='bold', zorder=11)


def draw_multirz(ax, x, y1, y2, small=False):
    """Draw MultiRZ gate (ZZ interaction with plus-in-circle notation)"""
    circle_size = 0.12 if small else 0.16
    plus_size = circle_size * 0.6
    
    # First qubit (circle with plus)
    circle1 = Circle((x, y1), circle_size, facecolor='white', edgecolor='black', 
                     linewidth=1.5, zorder=10)
    ax.add_patch(circle1)
    # Plus symbol
    ax.plot([x - plus_size, x + plus_size], [y1, y1], 'k-', linewidth=1.5, zorder=11)
    ax.plot([x, x], [y1 - plus_size, y1 + plus_size], 'k-', linewidth=1.5, zorder=11)
    
    # Second qubit (circle with plus - SYMMETRIC!)
    circle2 = Circle((x, y2), circle_size, facecolor='white', edgecolor='black', 
                     linewidth=1.5, zorder=10)
    ax.add_patch(circle2)
    # Plus symbol
    ax.plot([x - plus_size, x + plus_size], [y2, y2], 'k-', linewidth=1.5, zorder=11)
    ax.plot([x, x], [y2 - plus_size, y2 + plus_size], 'k-', linewidth=1.5, zorder=11)
    
    # Connection line
    ax.plot([x, x], [y1, y2], 'k-', linewidth=1.5, zorder=5)


def draw_label(ax, x, y, text, color='darkblue'):
    """Draw layer label"""
    bbox = dict(boxstyle='round,pad=0.4', facecolor='lightcyan', 
                edgecolor=color, linewidth=2)
    ax.text(x, y, text, fontsize=11, ha='center', fontweight='bold',
            color=color, bbox=bbox)

def draw_simple_comparison():
    """Draw a simple comparison: Classical vs Quantum feature mapping"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Classical kernel
    ax1 = axes[0]
    ax1.text(0.5, 0.9, 'Classical RBF Kernel', fontsize=16, fontweight='bold', 
             ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.7, r'$K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma ||\mathbf{x}-\mathbf{z}||^2)$', fontsize=13, 
             ha='center', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.text(0.5, 0.5, r'$\downarrow$', fontsize=30, ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.35, r'Feature Space: $\mathbb{R}^2 \rightarrow \mathbb{R}^{\infty}$ (infinite dim)', 
             fontsize=11, ha='center', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax1.text(0.5, 0.15, 'Gaussian basis functions', fontsize=10, 
             ha='center', transform=ax1.transAxes, style='italic')
    ax1.axis('off')
    
    # Quantum kernel  
    ax2 = axes[1]
    ax2.text(0.5, 0.9, 'Quantum Kernel (IQP)', fontsize=16, fontweight='bold', 
             ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.7, r'$K(\mathbf{x}, \mathbf{z}) = |\langle\psi(\mathbf{x})|\psi(\mathbf{z})\rangle|^2$', fontsize=13, 
             ha='center', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.text(0.5, 0.5, r'$\downarrow$', fontsize=30, ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.35, r'Feature Space: $\mathbb{R}^2 \rightarrow \mathbb{C}^{16}$ (Hilbert space)' + '\n' + r'4 qubits = $2^4$ dimensions', 
             fontsize=11, ha='center', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='pink', alpha=0.3))
    ax2.text(0.5, 0.15, 'Polynomial features via ZZ interactions', fontsize=10, 
             ha='center', transform=ax2.transAxes, style='italic')
    ax2.axis('off')
    
    plt.suptitle('Feature Mapping Comparison', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    path = 'results/quantum_viz/kernel_comparison.png'
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('results/quantum_viz', exist_ok=True)
    
    print("=" * 70)
    print("DRAWING IQP CIRCUIT (Manual matplotlib)")
    print("=" * 70)
    
    # Draw main circuit
    print("\n1. Drawing 4-qubit IQP circuit...")
    draw_iqp_circuit_manual(n_qubits=4, n_layers=2)
    
    # Draw comparison
    print("\n2. Drawing kernel comparison...")
    draw_simple_comparison()
    
    print("\n" + "=" * 70)
    print("✓ Circuit diagrams saved!")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✓ Circuit diagrams saved!")
    print("=" * 70)
