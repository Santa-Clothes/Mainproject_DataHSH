# 코드 리뷰 가이드 - Fashion JSON Encoder

## 🎯 코드 리뷰 시 제출할 핵심 파일

### **메인 진입점 (필수)**
1. **`main.py`** ⭐ - 전체 시스템 통합 및 진입점
   - `FashionEncoderSystem` 클래스: 모든 컴포넌트 통합
   - 데이터 로딩, 학습, 평가 파이프라인
   - CLI 인터페이스

2. **`train.py`** - 학습 전용 스크립트
   - 간단한 학습 실행용
   - main.py의 경량 버전

### **핵심 모델 (필수)**
3. **`models/json_encoder.py`** ⭐⭐⭐ - JSON Encoder 모델
   - 512차원 임베딩 생성
   - 다중 범주형 필드 처리
   - 프로젝트의 핵심 혁신

4. **`models/contrastive_learner.py`** ⭐⭐ - Contrastive Learning
   - InfoNCE Loss 구현
   - 이미지-JSON 임베딩 정렬
   - Temperature 최적화 (0.1)

### **데이터 처리 (필수)**
5. **`data/fashion_dataset.py`** ⭐ - PyTorch Dataset
   - K-Fashion 데이터 로딩
   - 전처리 파이프라인

6. **`data/dataset_loader.py`** - 데이터셋 로더
   - JSON 파싱 및 필터링
   - Vocabulary 구축

### **학습 시스템 (필수)**
7. **`training/trainer.py`** ⭐⭐ - 학습 트레이너
   - 학습 루프 구현
   - 평가 메트릭 계산
   - 체크포인트 관리

### **설정 및 유틸리티 (선택)**
8. **`utils/config.py`** - 설정 관리
9. **`utils/validators.py`** - 입력 검증

---

## 📊 코드 리뷰 우선순위

### **Tier 1: 핵심 알고리즘 (반드시 리뷰)**
```
models/json_encoder.py          ⭐⭐⭐ 가장 중요
models/contrastive_learner.py   ⭐⭐⭐ 가장 중요
training/trainer.py             ⭐⭐  매우 중요
```

### **Tier 2: 시스템 통합 (중요)**
```
main.py                         ⭐⭐  매우 중요
data/fashion_dataset.py         ⭐   중요
data/dataset_loader.py          ⭐   중요
```

### **Tier 3: 지원 코드 (선택)**
```
utils/config.py                 선택사항
utils/validators.py             선택사항
examples/*.py                   참고용
```

---

## 🎓 코드 리뷰 시나리오별 추천

### **시나리오 1: 교수님께 전체 시스템 리뷰**
제출 파일 (6개):
1. `main.py` - 시스템 개요
2. `models/json_encoder.py` - 핵심 모델
3. `models/contrastive_learner.py` - 학습 알고리즘
4. `training/trainer.py` - 학습 시스템
5. `data/fashion_dataset.py` - 데이터 처리
6. `README.md` - 프로젝트 설명

### **시나리오 2: 동료 학생과 코드 리뷰**
제출 파일 (3개):
1. `models/json_encoder.py` - 핵심 모델만
2. `models/contrastive_learner.py` - 학습 알고리즘
3. `main.py` - 사용 방법

### **시나리오 3: 알고리즘 중심 리뷰**
제출 파일 (2개):
1. `models/json_encoder.py` - JSON Encoder 구현
2. `models/contrastive_learner.py` - Contrastive Learning

### **시나리오 4: 전체 프로젝트 리뷰 (GitHub 등)**
제출 파일 (전체):
- 전체 프로젝트 폴더
- 특히 `models/`, `training/`, `data/` 폴더 강조

---

## 📝 각 파일의 핵심 내용

### `main.py` (약 400줄)
```python
class FashionEncoderSystem:
    """전체 시스템 통합"""
    - setup_data()      # 데이터 로딩
    - setup_trainer()   # 트레이너 초기화
    - train()           # 학습 실행
    - evaluate()        # 평가 실행
```

### `models/json_encoder.py` (약 200줄) ⭐⭐⭐
```python
class JSONEncoder(nn.Module):
    """JSON → 512차원 임베딩"""
    - __init__()        # 모델 구조 정의
    - forward()         # 순전파
    - _process_field()  # 필드별 처리
```
**핵심 혁신**:
- 다중 범주형 필드 처리 (mean pooling)
- 512차원 출력 + L2 정규화
- 단순하지만 효과적인 구조

### `models/contrastive_learner.py` (약 150줄) ⭐⭐⭐
```python
class ContrastiveLearner(nn.Module):
    """이미지-JSON 정렬"""
    - forward()                      # 학습
    - compute_contrastive_loss()     # InfoNCE Loss
    - get_embeddings()               # 임베딩 추출
```
**핵심 혁신**:
- Temperature 0.1 최적화
- In-batch negative sampling
- FashionCLIP 통합

### `training/trainer.py` (약 500줄) ⭐⭐
```python
class FashionTrainer:
    """학습 시스템"""
    - train_epoch()         # 에포크 학습
    - validate()            # 검증
    - _final_evaluation()   # 최종 평가
```
**핵심 기능**:
- Top-1/Top-5 정확도 계산
- MRR, Recall@K 메트릭
- 체크포인트 관리

---

## 💡 코드 리뷰 시 강조할 포인트

### **1. 핵심 혁신 (main.py + models/)**
- **임베딩 중심성 기반 베스트셀러 Proxy**
- **Query-Aware 평가 시스템**
- **Temperature 최적화 (0.1)**

### **2. 성능 지표 (training/trainer.py)**
- Top-5 정확도: 64.1%
- Temperature 0.1에서 최적 성능
- 2,172개 아이템 학습 완료

### **3. 코드 품질**
- 명확한 클래스 구조
- 타입 힌트 사용
- 상세한 docstring

### **4. 실용성**
- CLI 인터페이스 (main.py)
- 체크포인트 관리
- 로깅 시스템

---

## 🚀 빠른 코드 리뷰 체크리스트

### **리뷰어가 확인할 핵심 사항**

#### ✅ 모델 구조 (`models/json_encoder.py`)
- [ ] 512차원 출력 확인
- [ ] L2 정규화 적용 확인
- [ ] 다중 범주형 필드 처리 (mean pooling)
- [ ] 단순하고 효과적인 구조

#### ✅ 학습 알고리즘 (`models/contrastive_learner.py`)
- [ ] InfoNCE Loss 구현
- [ ] Temperature 0.1 사용
- [ ] FashionCLIP frozen 상태 유지
- [ ] In-batch negative sampling

#### ✅ 학습 시스템 (`training/trainer.py`)
- [ ] Top-1/Top-5 정확도 계산
- [ ] MRR, Recall@K 메트릭
- [ ] 체크포인트 저장/로드
- [ ] 검증 데이터 평가

#### ✅ 데이터 처리 (`data/fashion_dataset.py`)
- [ ] K-Fashion 데이터 로딩
- [ ] BBox 크롭 처리
- [ ] Vocabulary 구축
- [ ] 학습/검증 분할 (80/20)

---

## 📦 코드 리뷰 패키지 준비

### **Option 1: 핵심 파일만 (추천)**
```bash
# 6개 핵심 파일 압축
zip code_review.zip \
    main.py \
    models/json_encoder.py \
    models/contrastive_learner.py \
    training/trainer.py \
    data/fashion_dataset.py \
    README.md
```

### **Option 2: 전체 프로젝트**
```bash
# 전체 프로젝트 압축 (불필요한 파일 제외)
zip -r project_review.zip . \
    -x "*.pyc" \
    -x "__pycache__/*" \
    -x ".git/*" \
    -x "checkpoints/*" \
    -x "logs/*" \
    -x "results/*"
```

### **Option 3: GitHub 링크**
- 전체 프로젝트를 GitHub에 업로드
- README.md에 핵심 파일 경로 명시
- 리뷰어가 직접 탐색 가능

---

## 🎯 결론: 코드 리뷰 시 제출 파일

### **최소 필수 (3개)**
1. `models/json_encoder.py` ⭐⭐⭐
2. `models/contrastive_learner.py` ⭐⭐⭐
3. `main.py` ⭐⭐

### **권장 (6개)**
1. `main.py`
2. `models/json_encoder.py`
3. `models/contrastive_learner.py`
4. `training/trainer.py`
5. `data/fashion_dataset.py`
6. `README.md`

### **전체 리뷰 (전체 프로젝트)**
- 모든 파일 제출
- 특히 `models/`, `training/`, `data/` 강조

---

## 💬 리뷰 요청 시 멘트 예시

### **교수님께**
"교수님, 패션 추천 시스템 코드 리뷰를 부탁드립니다. 특히 `models/json_encoder.py`의 JSON Encoder 구현과 `models/contrastive_learner.py`의 Contrastive Learning 알고리즘을 중점적으로 봐주시면 감사하겠습니다. 현재 Top-5 정확도 64.1%를 달성했습니다."

### **동료 학생에게**
"코드 리뷰 좀 부탁해! `models/json_encoder.py`랑 `models/contrastive_learner.py` 두 파일만 봐주면 돼. JSON을 512차원 벡터로 바꾸는 부분이랑 이미지랑 매칭하는 부분이야."

### **GitHub Issue/PR**
"Implemented Fashion JSON Encoder with Contrastive Learning. Key files:
- `models/json_encoder.py`: JSON → 512D embedding
- `models/contrastive_learner.py`: Image-JSON alignment
- `training/trainer.py`: Training pipeline

Current performance: Top-5 accuracy 64.1%"
