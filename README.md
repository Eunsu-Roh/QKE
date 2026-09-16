# Quantum Kernel Estimation (QKE) vs Classical SVM

A project that reproduces the Quantum Kernel Estimation methodology from the Nature paper *"Supervised learning with quantum-enhanced feature spaces"* (Havlíček et al., 2019) using PennyLane, and compares its performance against classical SVM kernels.

## 🎯 Project Objectives

1. **Iris Dataset**: Implementation validation (Baseline)
2. **Non-linear Datasets**: Exploring quantum advantage
   - XOR (4 samples)
   - Concentric Circles
   - Spirals
   - Checkerboard

## 📁 Project Structure

```text
QKE/
├── kernels/
│   ├── __init__.py
│   ├── classical.py       # RBF, Polynomial, Linear kernels
│   └── quantum.py         # Quantum kernel (IQPEmbedding)
├── datasets/
│   ├── __init__.py
│   ├── iris.py            # Iris data loader
│   └── synthetic.py       # XOR, Concentric Circles, Spirals, Checkerboard
├── visualization/
│   ├── __init__.py
│   └── plots.py           # Decision boundaries, kernel matrices, performance comparison
├── experiments/
│   ├── run_iris.py        # Iris experiment
│   └── run_synthetic.py   # Non-linear datasets experiment
├── results/               # Stores experiment results
├── train.py               # SVM training pipeline
├── requirements.txt
└── README.md
```

## 🚀 Installation & Usage

### 1. Activate Conda Environment

```bash
conda activate env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Experiments

#### Iris Dataset (Baseline)
```bash
python experiments/run_iris.py
```

#### Non-linear Datasets (Verifying Quantum Advantage)
```bash
python experiments/run_synthetic.py
```

## 📊 Expected Results

| Dataset | RBF Kernel | Quantum Kernel | Improvement |
|---------|---------|----------|--------|
| Iris | ~97% | ~96% | -1% |
| XOR | ~75% | ~100% | **+25%** |
| Concentric Circles | ~88% | ~97% | **+9%** |
| Spirals | ~82% | ~94% | **+12%** |
| Checkerboard | ~79% | ~96% | **+17%** |

## 🔬 Tech Stack

- **Quantum Computing**: PennyLane
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Numerical Computing**: NumPy

## 📚 References

Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567(7747), 209-212.

## 📝 License

MIT License
