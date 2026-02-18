# Multi-Domain Training 실행 가이드

Domain Gap 문제 해결을 위한 Multi-Domain Training 구현 가이드

---

## 📋 개요

**문제:** 9oz 평면 제품 → K-Fashion 모델 착용 → Naver 모델 착용 (Domain Gap)

**해결:** 평면 제품 + 모델 착용 이미지를 모두 학습

---

## 🚀 빠른 시작

### 1단계: DeepFashion2 다운로드 준비

```bash
# 다운로드 안내 및 폴더 구조 생성
python scripts/data_collection/download_deepfashion2.py

# 또는 샘플 데이터셋 생성 (테스트용)
python scripts/data_collection/download_deepfashion2.py --create_sample
```

### 2단계: DeepFashion2 수동 다운로드

1. [DeepFashion2 GitHub](https://github.com/switchablenorms/DeepFashion2) 방문
2. [Google Drive 링크](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok) 접속
3. 다운로드:
   - `train/image/` (일부만 - 10K 이미지 권장)
   - `train/annos/` (JSON 어노테이션)
4. 압축 해제: `data/deepfashion2/`

### 3단계: 데이터 파싱 및 통합

```bash
# DeepFashion2 파싱
python scripts/data_collection/parse_deepfashion2.py \
    --deepfashion_dir data/deepfashion2 \
    --split train \
    --output_dir data/multi_domain

# Multi-Domain 데이터셋 생성
python scripts/data_collection/parse_deepfashion2.py \
    --create_multi_domain \
    --deepfashion_dir data/deepfashion2 \
    --output_dir data/multi_domain
```

### 4단계: 학습 (예정)

```bash
# Multi-Domain Contrastive Learning
python scripts/training/train_multi_domain.py \
    --config configs/multi_domain_v1.yaml \
    --batch_size 256 \
    --epochs 50
```

---

## 📁 생성된 파일 구조

```
c:\Mainproject_DataHSH\
├── data/
│   ├── deepfashion2/                    # DeepFashion2 원본 (수동 다운로드)
│   │   ├── train/
│   │   │   ├── image/
│   │   │   └── annos/
│   │   └── validation/
│   │       ├── image/
│   │       └── annos/
│   │
│   └── multi_domain/                    # 통합 데이터셋
│       └── multi_domain_dataset.csv     # 통합 CSV
│
├── src/data/
│   └── multi_domain_dataset.py          # MultiDomainFashionDataset
│
├── scripts/
│   └── data_collection/
│       ├── download_deepfashion2.py     # 다운로드 도구
│       └── parse_deepfashion2.py        # 파싱 도구
│
└── docs/
    ├── domain_gap_solutions.md          # 해결 방안 상세
    └── multi_domain_training_guide.md   # 이 파일
```

---

## 🧪 테스트

### 1. MultiDomainFashionDataset 테스트

```python
from src.data.multi_domain_dataset import MultiDomainFashionDataset

# Dataset 로드
dataset = MultiDomainFashionDataset(
    csv_path='data/multi_domain/multi_domain_dataset.csv',
    domain_augment=True
)

print(f"Total items: {len(dataset)}")

# 샘플 확인
item = dataset[0]
print(f"Image: {item['image'].shape}")
print(f"Category: {item['category']}")
print(f"Image Type: {item['image_type']}")
```

### 2. DataLoader 테스트

```python
from src.data.multi_domain_dataset import create_multi_domain_dataloader

dataloader = create_multi_domain_dataloader(
    dataset,
    batch_size=32,
    shuffle=True
)

batch = next(iter(dataloader))
print(f"Batch size: {batch['images'].shape[0]}")
print(f"Domains: {batch['domains']}")
```

---

## 📊 예상 결과

### Before (현재)
```
Query: 9oz 평면 제품
Target: Naver 모델 착용
Top-5 Accuracy: ~40% (추정)
```

### After (Multi-Domain Training)
```
Query: 9oz 평면 제품
Target: Naver 모델 착용
Top-5 Accuracy: 70-80% (목표)
```

---

## 🔧 트러블슈팅

### 문제 1: DeepFashion2 너무 큼
**해결:** 일부만 다운로드 (10K 이미지)

### 문제 2: 메모리 부족
**해결:** 배치 크기 감소, num_workers 조정

### 문제 3: GPU 메모리 부족
**해결:**
- Mixed Precision (FP16) 사용
- Gradient Accumulation
- 이미지 크기 축소 (224 → 192)

---

## 📝 다음 단계

1. ✅ DeepFashion2 다운로드 스크립트
2. ✅ 데이터 파싱 및 전처리
3. ✅ MultiDomainFashionDataset 구현
4. ⏳ **학습 스크립트 수정** (진행 중)
5. ⏳ 성능 검증

---

## 🎯 성공 지표

- [ ] Multi-Domain Dataset 생성 (20K+ 이미지)
- [ ] 학습 완료 (50 epochs)
- [ ] Top-5 Accuracy > 70%
- [ ] Domain별 성능 분석
- [ ] 9oz → Naver 검색 정확도 향상

---

## 📚 참고 자료

- [DeepFashion2 Paper](https://arxiv.org/abs/1901.07973)
- [Contrastive Learning](https://arxiv.org/abs/2002.05709)
- [Domain Adaptation](https://arxiv.org/abs/1409.7495)
