"""
Inspect IQPEmbedding circuit structure
"""

import pennylane as qml
from pennylane import numpy as np

# Create device
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def iqp_circuit(x, n_layers=2):
    """IQP embedding circuit"""
    qml.templates.IQPEmbedding(
        features=x,
        wires=range(4),
        n_repeats=n_layers
    )
    return qml.state()

# Sample data
x = np.array([0.5, 1.0, 0.3, 0.7])

print("=" * 70)
print("IQP EMBEDDING CIRCUIT STRUCTURE")
print("=" * 70)
print(f"\n입력 파라미터:")
print(f"  - Qubits: 4")
print(f"  - Layers (n_repeats): 2")
print(f"  - Features: {x}")
print(f"  - Device: default.qubit (노이즈 없음)")

print(f"\n" + "=" * 70)
print("회로 구조 (Text):")
print("=" * 70)
drawer = qml.draw(iqp_circuit)
print(drawer(x, n_layers=2))

print(f"\n" + "=" * 70)
print("상세 게이트 정보:")
print("=" * 70)

# Get circuit operations
with qml.tape.QuantumTape() as tape:
    iqp_circuit(x, n_layers=2)

print(f"\n총 게이트 수: {len(tape.operations)}")
print(f"Circuit depth: {tape.graph.get_depth()}")

print("\n게이트 리스트:")
for idx, op in enumerate(tape.operations):
    if hasattr(op, 'wires'):
        wires = str([w for w in op.wires])
    else:
        wires = "[]"
    
    if hasattr(op, 'parameters') and len(op.parameters) > 0:
        params = [f"{p:.3f}" for p in op.parameters]
        print(f"  {idx+1:2d}. {op.name:15s} on wires {wires:20s} params: {params}")
    else:
        print(f"  {idx+1:2d}. {op.name:15s} on wires {wires}")

print(f"\n" + "=" * 70)
print("IQPEmbedding 구조 설명:")
print("=" * 70)
print("""
IQPEmbedding (Instantaneous Quantum Polynomial):

각 Layer마다:
1. Hadamard (H) gates - 모든 큐비트에 적용 (중첩 상태 생성)
2. RZ rotations - 각 큐비트에 feature 값 인코딩
3. Multi-RZ gates - 큐비트 쌍 간 ZZ entanglement

수식: U_IQP(x) = exp(i * Σ φ(x) * Z⊗Z)

특징:
- Polynomial feature map 생성
- 고전 컴퓨터로 효율적 시뮬레이션 불가능 (일반적으로)
- Nature 2019 논문에서 사용한 정확한 구조
""")

print(f"\n" + "=" * 70)
print("노이즈 모델:")
print("=" * 70)
print("""
현재 설정: default.qubit (이상적 양자 시뮬레이터)
- Gate error: 0%
- Decoherence: 없음
- Readout error: 없음

실제 양자 컴퓨터에서는 노이즈 고려 필요:
- qml.device('default.mixed', wires=4) - 혼합 상태
- qml.DepolarizingChannel() - 게이트 노이즈
- qml.AmplitudeDamping() - T1 relaxation
- qml.PhaseDamping() - T2 dephasing
""")
