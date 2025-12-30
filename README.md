# Quantum Kernel Estimation (QKE) vs Classical SVM

Nature 논문 "Supervised learning with quantum-enhanced feature spaces" (Havlíček et al., 2019)의 Quantum Kernel Estimation 방법론을 PennyLane으로 재현하고, 고전 SVM 커널과 성능을 비교하는 프로젝트입니다.

## 🎯 프로젝트 목표

1. **Iris 데이터셋**: 구현 검증 (베이스라인)
2. **비선형 데이터셋**: 양자 이점 탐색
   - XOR (4샘플)
   - 동심원 (Concentric Circles)
   - 나선형 (Spirals)
   - 체커보드 (Checkerboard)

## 📁 프로젝트 구조

```
QKE/
├── kernels/
│   ├── __init__.py
│   ├── classical.py       # RBF, Polynomial, Linear 커널
│   └── quantum.py         # 양자 커널 (IQPEmbedding)
├── datasets/
│   ├── __init__.py
│   ├── iris.py           # Iris 데이터 로더
│   └── synthetic.py      # XOR, 동심원, 나선형, 체커보드
├── visualization/
│   ├── __init__.py
│   └── plots.py          # 결정 경계, 커널 행렬, 성능 비교
├── experiments/
│   ├── run_iris.py       # Iris 실험
│   └── run_synthetic.py  # 비선형 데이터셋 실험
├── results/              # 실험 결과 저장
├── train.py             # SVM 학습 파이프라인
├── requirements.txt
└── README.md
```

## 🚀 설치 및 실행

### 1. Conda 가상환경 활성화

```powershell
conda activate env
```

### 2. 의존성 설치

```powershell
pip install -r requirements.txt
```

### 3. 실험 실행

#### Iris 데이터셋 (베이스라인)
```powershell
python experiments/run_iris.py
```

#### 비선형 데이터셋 (양자 이점 검증)
```powershell
python experiments/run_synthetic.py
```

## 📊 예상 결과

| 데이터셋 | RBF 커널 | 양자 커널 | 개선율 |
|---------|---------|----------|--------|
| Iris | ~97% | ~96% | -1% |
| XOR | ~75% | ~100% | **+25%** |
| 동심원 | ~88% | ~97% | **+9%** |
| 나선형 | ~82% | ~94% | **+12%** |
| 체커보드 | ~79% | ~96% | **+17%** |

## 🔬 기술 스택

- **양자 컴퓨팅**: PennyLane
- **머신러닝**: scikit-learn
- **시각화**: matplotlib, seaborn
- **수치 계산**: NumPy

## 📚 참고 문헌

Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567(7747), 209-212.

## 📝 라이선스

MIT License
