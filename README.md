# Fashion JSON Encoder v5

패션 이미지와 JSON 메타데이터를 정렬하는 대조 학습 시스템

---

## 🚀 빠른 시작

### 설치
```bash
pip install -r requirements.txt
```

### 데이터 구조
```
K-fashion/
├── Training/
│   ├── 원천데이터/  (이미지)
│   └── 라벨링데이터/  (JSON)
└── Validation/
    ├── 원천데이터/
    └── 라벨링데이터/
```

### 데이터 경로 설정
1. `utils/config.py` 열기
2. `DataConfig` 클래스의 `data_root` 수정
```python
data_root: Path = Path("C:/실제/데이터/경로")
```

### 학습
```bash
# RTX 4090 최적화 버전 (권장)
python scripts/training/create_baseline_v5_rtx4090_optimized.py
```

---

## 📁 프로젝트 구조

```
├── models/          # JSON Encoder, Contrastive Learner
├── data/            # 데이터 로더, 전처리
├── training/        # 학습 시스템
├── scripts/         # 학습/평가 스크립트
├── checkpoints/     # 학습된 모델
├── tests/           # 테스트
└── requirements.txt
```

---

## 🎨 핵심 특징

- **FashionCLIP**: 패션 특화 이미지 인코더
- **Data Augmentation**: 6가지 증강 기법
- **Class Balancing**: 불균형 데이터 해결
- **RTX 4090 최적화**: Mixed Precision, TF32

---


## 🧪 테스트

```bash
python -m pytest tests/ -v
```

---

## 🛠️ 기술 스택

- PyTorch 2.0+, Transformers
- FashionCLIP (ViT-B/32, frozen)
- Custom JSON Encoder (305K params)
- InfoNCE Loss

---

## 📄 라이선스

MIT License
