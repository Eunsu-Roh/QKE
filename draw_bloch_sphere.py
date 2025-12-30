"""
Bloch Sphere Visualization (Publication Quality)

Draw quantum states on Bloch sphere in Nature paper style
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


class Arrow3D(FancyArrowPatch):
    """3D arrow for state vectors"""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        return np.min(zs)


def draw_bloch_sphere_base(ax):
    """Draw the basic Bloch sphere structure"""
    # Sphere surface
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, alpha=0.08, color='gray', edgecolor='none')
    
    # Equator circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x_eq = np.cos(theta)
    y_eq = np.sin(theta)
    z_eq = np.zeros_like(theta)
    ax.plot(x_eq, y_eq, z_eq, 'gray', linewidth=1.5, alpha=0.4)
    
    # Meridians
    phi = np.linspace(0, np.pi, 100)
    # XZ plane
    ax.plot(np.sin(phi), np.zeros_like(phi), np.cos(phi), 'gray', linewidth=1, alpha=0.3)
    # YZ plane
    ax.plot(np.zeros_like(phi), np.sin(phi), np.cos(phi), 'gray', linewidth=1, alpha=0.3)
    
    # Coordinate axes
    axis_length = 1.4
    ax.plot([-axis_length, axis_length], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.3)
    ax.plot([0, 0], [-axis_length, axis_length], [0, 0], 'k-', linewidth=1, alpha=0.3)
    ax.plot([0, 0], [0, 0], [-axis_length, axis_length], 'k-', linewidth=1, alpha=0.3)
    
    # Axis labels
    ax.text(axis_length + 0.1, 0, 0, 'X', fontsize=14, fontweight='bold')
    ax.text(0, axis_length + 0.1, 0, 'Y', fontsize=14, fontweight='bold')
    ax.text(0, 0, axis_length + 0.2, '|0⟩', fontsize=16, fontweight='bold')
    ax.text(0, 0, -axis_length - 0.2, '|1⟩', fontsize=16, fontweight='bold')
    
    # Set equal aspect ratio and limits
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect([1, 1, 1])
    
    # Remove grid and ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def add_state_vector(ax, theta, phi, color, label, label_pos='auto'):
    """
    Add a state vector to Bloch sphere
    
    State: |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    Bloch vector: (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))
    """
    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Draw state vector as thick arrow
    arrow = Arrow3D(0, 0, 0, x, y, z, 
                    mutation_scale=20, lw=3, arrowstyle='-|>', color=color)
    ax.add_artist(arrow)
    
    # Add label
    if label_pos == 'auto':
        label_x, label_y, label_z = x * 1.3, y * 1.3, z * 1.3
    else:
        label_x, label_y, label_z = label_pos
    
    ax.text(label_x, label_y, label_z, label, fontsize=13, fontweight='bold',
            color=color, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor=color, linewidth=2, alpha=0.9))


def draw_quantum_states_overview():
    """Draw common quantum states on Bloch sphere"""
    fig = plt.figure(figsize=(16, 5))
    
    # State 1: |0⟩ (North pole)
    ax1 = fig.add_subplot(141, projection='3d')
    draw_bloch_sphere_base(ax1)
    add_state_vector(ax1, theta=0, phi=0, color='red', label='|0⟩')
    ax1.set_title('Ground State |0⟩\nθ=0, φ=0', fontsize=12, fontweight='bold', pad=10)
    ax1.view_init(elev=20, azim=45)
    
    # State 2: |1⟩ (South pole)
    ax2 = fig.add_subplot(142, projection='3d')
    draw_bloch_sphere_base(ax2)
    add_state_vector(ax2, theta=np.pi, phi=0, color='blue', label='|1⟩')
    ax2.set_title('Excited State |1⟩\nθ=π, φ=0', fontsize=12, fontweight='bold', pad=10)
    ax2.view_init(elev=20, azim=45)
    
    # State 3: |+⟩ = (|0⟩+|1⟩)/√2 (X-axis)
    ax3 = fig.add_subplot(143, projection='3d')
    draw_bloch_sphere_base(ax3)
    add_state_vector(ax3, theta=np.pi/2, phi=0, color='green', label='|+⟩')
    ax3.set_title('Superposition |+⟩\n(|0⟩+|1⟩)/√2\nθ=π/2, φ=0', 
                  fontsize=11, fontweight='bold', pad=10)
    ax3.view_init(elev=20, azim=45)
    
    # State 4: |i⟩ = (|0⟩+i|1⟩)/√2 (Y-axis)
    ax4 = fig.add_subplot(144, projection='3d')
    draw_bloch_sphere_base(ax4)
    add_state_vector(ax4, theta=np.pi/2, phi=np.pi/2, color='purple', label='|i⟩')
    ax4.set_title('Complex State |i⟩\n(|0⟩+i|1⟩)/√2\nθ=π/2, φ=π/2', 
                  fontsize=11, fontweight='bold', pad=10)
    ax4.view_init(elev=20, azim=45)
    
    plt.suptitle('Quantum States on Bloch Sphere', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    path = 'results/quantum_viz/bloch_states_overview.png'
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {path}")
    plt.close()


def draw_hadamard_action():
    """Show Hadamard gate effect on Bloch sphere"""
    fig = plt.figure(figsize=(16, 5))
    
    # Before: |0⟩
    ax1 = fig.add_subplot(141, projection='3d')
    draw_bloch_sphere_base(ax1)
    add_state_vector(ax1, theta=0, phi=0, color='red', label='|0⟩')
    ax1.set_title('Initial: |0⟩\n(North Pole)', fontsize=12, fontweight='bold', pad=10)
    ax1.view_init(elev=20, azim=45)
    
    # Arrow
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.text(0.5, 0.5, 0.5, 'H', fontsize=50, ha='center', va='center',
             fontweight='bold', color='blue',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                      edgecolor='blue', linewidth=3))
    ax2.text(0.5, 0.2, 0.5, 'Hadamard\nGate', fontsize=14, ha='center', va='center',
             style='italic')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_zlim([0, 1])
    ax2.axis('off')
    
    # After: |+⟩
    ax3 = fig.add_subplot(143, projection='3d')
    draw_bloch_sphere_base(ax3)
    add_state_vector(ax3, theta=np.pi/2, phi=0, color='green', label='|+⟩')
    # Show original position as faded
    arrow_old = Arrow3D(0, 0, 0, 0, 0, 1, mutation_scale=15, lw=2, 
                       arrowstyle='-|>', color='red', alpha=0.3, linestyle='--')
    ax3.add_artist(arrow_old)
    ax3.set_title('Result: |+⟩\n(Equator, X-axis)', fontsize=12, fontweight='bold', pad=10)
    ax3.view_init(elev=20, azim=45)
    
    # Equation
    ax4 = fig.add_subplot(144, projection='3d')
    equation = (
        'H|0⟩ = |+⟩\n\n'
        '= (|0⟩ + |1⟩)/√2\n\n'
        'Creates equal\n'
        'superposition'
    )
    ax4.text(0.5, 0.5, 0.5, equation, fontsize=14, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                      edgecolor='orange', linewidth=2))
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.set_zlim([0, 1])
    ax4.axis('off')
    
    plt.suptitle('Hadamard Gate: Creating Superposition', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    path = 'results/quantum_viz/bloch_hadamard.png'
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {path}")
    plt.close()


def draw_rz_rotation():
    """Show RZ rotation (feature encoding) on Bloch sphere"""
    fig = plt.figure(figsize=(12, 5))
    
    # Starting from |+⟩
    ax1 = fig.add_subplot(121, projection='3d')
    draw_bloch_sphere_base(ax1)
    add_state_vector(ax1, theta=np.pi/2, phi=0, color='green', label='|+⟩')
    ax1.set_title('Before RZ: |+⟩\n(After Hadamard)', fontsize=12, fontweight='bold', pad=10)
    ax1.view_init(elev=20, azim=45)
    
    # After RZ(θ) rotation
    ax2 = fig.add_subplot(122, projection='3d')
    draw_bloch_sphere_base(ax2)
    
    # Show multiple rotation angles
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    colors = ['green', 'lime', 'yellow', 'orange', 'red']
    
    for angle, color in zip(angles, colors):
        add_state_vector(ax2, theta=np.pi/2, phi=angle, color=color, 
                        label=f'θ={angle:.2f}', label_pos='auto')
    
    # Draw rotation path (circle on equator)
    phi_path = np.linspace(0, np.pi, 50)
    x_path = np.cos(phi_path)
    y_path = np.sin(phi_path)
    z_path = np.zeros_like(phi_path)
    ax2.plot(x_path, y_path, z_path, 'b--', linewidth=2, alpha=0.5, label='RZ rotation')
    
    ax2.set_title('After RZ(θ): Feature Encoding\n(Rotation around Z-axis)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.view_init(elev=20, azim=45)
    
    # Add text explanation
    fig.text(0.5, 0.02, 
             'RZ(θ) rotates the state around Z-axis by angle θ\n' +
             'Different feature values → Different rotation angles → Different quantum states',
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('RZ Rotation: Encoding Classical Data into Quantum States', 
                 fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    path = 'results/quantum_viz/bloch_rz_encoding.png'
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('results/quantum_viz', exist_ok=True)
    
    print("=" * 70)
    print("BLOCH SPHERE VISUALIZATION (Publication Quality)")
    print("=" * 70)
    
    print("\n1. Drawing quantum states overview...")
    draw_quantum_states_overview()
    
    print("\n2. Drawing Hadamard gate action...")
    draw_hadamard_action()
    
    print("\n3. Drawing RZ rotation (feature encoding)...")
    draw_rz_rotation()
    
    print("\n" + "=" * 70)
    print("✓ All Bloch sphere visualizations complete!")
    print("=" * 70)
