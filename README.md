# Fashion JSON Encoder v5
## K-Fashion 이미지-메타데이터 정렬 시스템

패션 이미지와 JSON 메타데이터를 정렬하는 대조 학습 기반 시스템. FashionCLIP과 JSON Encoder를 결합하여 고성능 패션 검색 및 추천을 제공합니다.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 핵심 성과

### v5 (최신 - 프로덕션 배포 준비 완료) ✅

| 지표 | 성능 | 개선 |
|------|------|------|
| **Top-1 Accuracy** | **47.1%** | v3 대비 +16.8%p |
| **Top-5 Accuracy** | **87.1%** | v3 대비 +15.6%p |
| **MRR** | **0.638** | v3 대비 +31.8% |
| **Val Loss** | **1.744** | v3 대비 -24.8% |

### 버전 히스토리

```
v1: Top-1  1.0% (초기 베이스라인)
v2: Top-1 22.2% (Standard CLIP)
v3: Top-1 30.3% (FashionCLIP Frozen) ← 이전 최고
v4: Top-1 29.4% (Fine-tuning 실패)
v5: Top-1 47.1% (Augmentation + Class Balancing) ← 🏆 현재 최고
```

### 재현성 검증 ✅

4개의 다른 random seed로 검증 완료:
- **변동계수 (CV)**: Top-1 2.56%, Top-5 1.31%
- **평가**: 매우 안정적 (CV < 5%)
- **결론**: Seed에 robust하며 재현 가능

---

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone <repository-url>
cd fashion-json-encoder

# 의존성 설치
pip install -r requirements.txt
```

### 기본 학습 (v5)

```bash
# v5 학습 (Data Augmentation + Class Balancing)
python scripts/training/create_baseline_v5_augmented.py

# 간단한 학습 스크립트 사용
python scripts/train.py --dataset_path C:/sample/라벨링데이터

# 데이터 경로 수정이 필요한 경우
# scripts/training/create_baseline_v5_augmented.py 파일에서
# dataset_path = "C:/sample/라벨링데이터" 수정
```

### 추론 및 평가

```python
from training.trainer import FashionTrainer
from data.fashion_dataset import FashionDataModule

# 데이터 로드
data_module = FashionDataModule(
    dataset_path="C:/sample/라벨링데이터",
    target_categories=['레트로', '로맨틱', '리조트']
)
data_module.setup()

# 모델 로드
trainer = FashionTrainer(...)
trainer.load_checkpoint("checkpoints/baseline_v5_best_model.pt")

# 평가
results = trainer._final_evaluation(data_module.val_dataloader())
print(f"Top-1: {results['top1_accuracy']*100:.1f}%")
print(f"Top-5: {results['top5_accuracy']*100:.1f}%")
```

---

## 📁 프로젝트 구조

```
fashion-json-encoder/
├── 📂 models/                    # 핵심 모델
│   ├── json_encoder.py           # JSON 메타데이터 인코더
│   └── contrastive_learner.py    # 대조 학습 시스템
│
├── 📂 data/                      # 데이터 파이프라인
│   ├── dataset_loader.py         # K-Fashion 데이터 로더
│   ├── fashion_dataset.py        # PyTorch Dataset
│   ├── processor.py              # 전처리
│   └── class_balanced_sampler.py # 클래스 밸런싱
│
├── 📂 training/                  # 학습 시스템
│   └── trainer.py                # FashionTrainer 클래스
│
├── 📂 scripts/                   # 실험 스크립트
│   ├── training/                 # 학습 스크립트
│   │   ├── create_baseline_v3_fashionclip.py
│   │   ├── create_baseline_v4_finetuned.py
│   │   ├── create_baseline_v5_augmented.py  ← 최신
│   │   └── create_baseline_v5_seed_validation.py
│   ├── analysis/                 # 분석 도구
│   └── testing/                  # 테스트 스크립트
│
├── 📂 results/                   # 실험 결과
│   ├── baseline_v3_final_report.md
│   ├── baseline_v3_vs_v4_analysis.md
│   ├── baseline_v5_final_report.md  ← 최신
│   ├── v3_vs_v5_comparison.json
│   └── v5_seed_validation_report.md
│
├── 📂 checkpoints/               # 모델 체크포인트
│   ├── baseline_v3_best_model.pt
│   ├── baseline_v4_best_model.pt
│   └── baseline_v5_best_model.pt  ← 프로덕션 배포용
│
├── 📂 tests/                     # 테스트 스위트
├── 📂 docs/                      # 문서
├── 📂 .kiro/specs/               # 프로젝트 명세
│
├── 📂 docs/                      # 문서
│   ├── CHANGELOG.md              # 변경 이력
│   ├── PROJECT_SUMMARY.md        # 프로젝트 요약
│   ├── CODE_QUALITY_IMPROVEMENTS.md
│   └── CLEANUP_COMPLETE.md
│
├── 📂 scripts/                   # 실행 스크립트
│   ├── train.py                  # 간단한 학습 스크립트
│   ├── main.py                   # 메인 시스템
│   └── start_api_server.py       # API 서버
│
├── README.md                     # 이 파일
├── requirements.txt              # 의존성
└── setup.py                      # 설치 스크립트
```

---

## 🎨 v5 핵심 개선 사항

### 1. 강화된 데이터 증강 ⭐⭐⭐

```python
# 적용된 augmentation
- RandomResizedCrop (scale 0.8-1.0)
- ColorJitter (brightness/contrast/saturation 0.2)
- RandomRotation (10 degrees)
- RandomAffine (translate 5%, scale 5%)
- GaussianBlur (30% probability)
- RandomErasing (20% probability)
```

**효과:**
- 데이터 다양성 증가 → 일반화 능력 향상
- 실제 환경 변화 대응력 향상
- 과적합 방지

### 2. 클래스 밸런싱 ⭐⭐⭐

**문제:**
- 레트로: 147개 (8.5%) - 심각한 불균형
- 로맨틱: 786개 (45.3%)
- 리조트: 804개 (46.3%)

**해결:**
- ClassBalancedSampler 적용
- 소수 클래스 오버샘플링
- 배치당 클래스별 최소 샘플 수 보장

**효과:**
- 모든 카테고리 균등 학습
- Top-1 accuracy 대폭 향상 (+16.8%p)

### 3. FashionCLIP 활용 ⭐⭐⭐

- **모델**: `patrickjohncyh/fashion-clip`
- **상태**: Frozen (fine-tuning 불필요)
- **효과**: 패션 도메인 특화 특징 추출

---

## 📊 상세 성능 분석

### v3 vs v5 비교

| 항목 | v3 (Baseline) | v5 (Augmented) | 개선율 |
|------|---------------|----------------|--------|
| **Top-1** | 30.3% | **47.1%** | **+55.4%** |
| **Top-5** | 71.5% | **87.1%** | **+21.8%** |
| **MRR** | 0.484 | **0.638** | **+31.8%** |
| **Val Loss** | 2.319 | **1.744** | **-24.8%** |
| **학습 시간** | 25분 | 35분 | +10분 |
| **Epochs** | 10 | 15 | +5 |

### 학습 곡선 (v5)

```
Epoch  Train Loss  Val Loss  Top-1   Top-5   MRR
  1      2.369      2.163    33.3%   77.0%   0.515
  5      1.672      1.835    41.7%   85.3%   0.604
  9      1.515      1.760    47.1%   87.7%   0.637
 12      1.464      1.748    47.3%   87.1%   0.639
 15      1.447      1.744    47.1%   87.1%   0.638  ← 최종
```

**관찰:**
- Epoch 1부터 v3 초과 성능
- Epoch 9에서 성능 안정화
- 과적합 없이 안정적 수렴

---

## 🔧 사용 가이드

### 학습 설정

```python
from utils.config import TrainingConfig

config = TrainingConfig(
    batch_size=16,           # v5 기본값
    learning_rate=1e-4,
    max_epochs=15,
    temperature=0.1,         # 최적값
    embedding_dim=128,
    hidden_dim=256,
    output_dim=512,
    dropout_rate=0.1,
    weight_decay=1e-5
)
```

### 데이터 증강 커스터마이징

```python
from torchvision import transforms

# v5 augmentation
enhanced_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])
```

### 클래스 밸런싱 활용

```python
from data.class_balanced_sampler import create_balanced_dataloader

# 클래스 밸런싱 DataLoader
train_loader = create_balanced_dataloader(
    dataset=train_dataset,
    batch_size=16,
    oversample_minority=True,
    min_samples_per_class=2
)
```

---

## 🧪 테스트

### 테스트 실행

```bash
# 전체 테스트
python -m pytest tests/ -v

# 특정 모듈
python -m pytest tests/test_json_encoder.py -v
python -m pytest tests/test_contrastive_learner.py -v
```

### 테스트 커버리지

- **JSON Encoder**: 100% 통과
- **Contrastive Learner**: 100% 통과
- **Data Pipeline**: 100% 통과
- **Training**: 100% 통과

---

## 📈 향후 개선 방향

### v6: 배치 크기 증가 ⭐⭐

```
현재: Batch 16
변경: Batch 32-64
예상 효과: Top-5 +1-2%, 더 안정적 학습
```

### v7: Temperature 튜닝 ⭐

```
현재: 0.1
실험: 0.07, 0.15
예상 효과: Top-1 +1-2%
```

### v8: Ensemble ⭐⭐⭐

```
방법: v3 + v5 앙상블
예상 효과: Top-1 +2-3%, 더 robust
```

### 장기: 데이터 확장 ⭐⭐⭐

```
현재: 2,172 items
목표: 5,000-10,000 items
예상 효과: Top-1 50%+
```

---

## 📚 주요 문서

### 실험 결과
- [v5 최종 리포트](results/baseline_v5_final_report.md) - v5 상세 분석
- [v3 vs v5 비교](results/v3_vs_v5_comparison.json) - 버전 비교
- [Seed 검증 리포트](results/v5_seed_validation_report.md) - 재현성 검증
- [v3 vs v4 분석](results/baseline_v3_vs_v4_analysis.md) - Fine-tuning 실패 분석

### 프로젝트 문서
- [프로젝트 요약](docs/PROJECT_SUMMARY.md) - 전체 요약
- [변경 이력](docs/CHANGELOG.md) - 버전 히스토리
- [코드 품질 개선](docs/CODE_QUALITY_IMPROVEMENTS.md) - 개선 사항
- [정리 완료](docs/CLEANUP_COMPLETE.md) - 프로젝트 정리

### 프로젝트 명세
- [요구사항](.kiro/specs/fashion-json-encoder/requirements.md)
- [설계 문서](.kiro/specs/fashion-json-encoder/design.md)
- [작업 계획](.kiro/specs/fashion-json-encoder/tasks.md)

---

## 🛠️ 기술 스택

### 핵심 라이브러리
- **PyTorch 2.0+**: 딥러닝 프레임워크
- **Transformers**: FashionCLIP 모델
- **torchvision**: 이미지 증강
- **NumPy**: 수치 연산

### 모델 아키텍처
- **Image Encoder**: FashionCLIP (ViT-B/32, 87M params, frozen)
- **JSON Encoder**: Custom Transformer (305K params, trainable)
- **Loss**: InfoNCE (Contrastive Learning)
- **Temperature**: 0.1

### 데이터
- **Dataset**: K-Fashion (2,172 items)
- **Categories**: 레트로, 로맨틱, 리조트
- **Split**: 80% train, 20% validation
- **Augmentation**: 6가지 기법 적용

---

## 🎯 프로덕션 배포

### 배포 준비 체크리스트

- [x] 성능 목표 달성 (Top-1 ≥ 32%, Top-5 ≥ 73%)
- [x] 재현성 검증 완료 (4개 seed 테스트)
- [x] 안정성 확인 (과적합 없음)
- [x] 클래스 밸런싱 (모든 카테고리 균등)
- [x] 체크포인트 저장 (`checkpoints/baseline_v5_best_model.pt`)
- [x] 문서화 완료

### 배포 권장사항

1. **모델**: `checkpoints/baseline_v5_best_model.pt` 사용
2. **Seed**: 42 고정 (재현성)
3. **설정**: v5 기본 설정 유지
4. **모니터링**: Top-1, Top-5, MRR 추적

---

## 🤝 기여

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

---

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.

---

## 🎉 주요 성과 요약

### ✅ v5 달성 사항

1. **압도적인 성능**: Top-1 47.1%, Top-5 87.1%
2. **목표 초과 달성**: 목표 대비 +15%p 초과
3. **재현성 검증**: 4개 seed로 안정성 확인
4. **프로덕션 준비**: 즉시 배포 가능

### 🚀 다음 단계

1. **즉시**: v5 프로덕션 배포
2. **단기**: 실제 환경 테스트 및 모니터링
3. **중기**: Ensemble 시스템 구축 (v8)
4. **장기**: 데이터 확장 (5,000+ items)

---

**Made with ❤️ for Fashion AI**

*Last Updated: 2026-02-10*
