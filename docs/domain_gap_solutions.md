# Domain Gap 해결 방안 상세 분석

## 현재 문제

```
파이프라인:
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  9oz 평면 제품   │ --> │  K-Fashion   │ --> │ Naver 모델 착용  │
│  (쿼리 입력)     │     │  (학습 데이터) │     │  (검색 대상)     │
└─────────────────┘     └──────────────┘     └─────────────────┘
      🔲                      👤                      👤

❌ Gap 1: 쿼리 도메인 ≠ 학습 도메인
❌ Gap 2: 쿼리 도메인 ≠ 검색 도메인
```

---

## 옵션 1: Multi-Domain Training (권장 ⭐⭐⭐⭐⭐)

### 📋 핵심 아이디어
- **평면 제품**과 **모델 착용** 이미지를 **모두** 학습에 사용
- FashionCLIP이 두 도메인을 자연스럽게 학습하도록 함
- "같은 제품"이면 평면이든 착용이든 임베딩이 비슷해지도록 학습

### 🔧 구현 단계

#### 1단계: 평면 제품 데이터셋 수집

**데이터셋 후보:**

| 데이터셋 | 규모 | 특징 | 다운로드 |
|---------|------|------|---------|
| **DeepFashion2** | 491K 이미지 | 평면+착용 혼합, bbox/pose 제공 | [링크](https://github.com/switchablenorms/DeepFashion2) |
| **Product-10K** | 10K 제품 | 평면 제품 중심, 고품질 | [Kaggle](https://www.kaggle.com/competitions/product-10k) |
| **Fashion-MNIST** | 70K 이미지 | 28x28 그레이스케일, 간단 | Built-in |
| **Zalando** | 45K 이미지 | 패션 이커머스, 평면 | [공개 데이터](https://research.zalando.com/) |

**추천 조합:**
```
학습 데이터 = K-Fashion (모델 착용) + DeepFashion2 평면 부분 (20K)
```

#### 2단계: 데이터 구성

```python
# 학습 데이터 비율
training_data = {
    'k_fashion_model_wearing': 2172,      # 기존
    'deepfashion2_flat_product': 10000,   # 새로 추가
    'deepfashion2_model_wearing': 10000,  # 새로 추가
}

# 총 22K 이미지 (현재 2K → 10배 증가)
```

**데이터 라벨링:**
```python
class FashionItem:
    image_path: str
    image_type: str  # 'flat_product' | 'model_wearing' | 'mannequin'
    category: str
    style: List[str]
    # ... 기타 필드
```

#### 3단계: Contrastive Learning 확장

**현재 방식:**
```python
# 이미지-JSON 쌍만 학습
positive_pairs = [
    (image_embedding, json_embedding),  # 같은 제품
]
```

**새로운 방식 (Multi-View Contrastive Learning):**
```python
# 같은 제품의 여러 뷰를 positive로 설정
positive_pairs = [
    (flat_image, json_metadata),        # 평면 이미지 - 메타데이터
    (model_image, json_metadata),       # 착용 이미지 - 메타데이터
    (flat_image, model_image),          # 평면 ↔ 착용 (핵심!)
]

# Loss 함수
loss = contrastive_loss(flat_emb, json_emb) + \
       contrastive_loss(model_emb, json_emb) + \
       contrastive_loss(flat_emb, model_emb)  # 도메인 간 정렬
```

#### 4단계: Domain-Specific Augmentation

```python
def augment_flat_product(image):
    """평면 제품 전용 augmentation"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(15),  # 평면은 회전 가능
        AddShadow(p=0.3),               # 그림자 추가
        AddBackground(p=0.5),           # 배경 변경
    ])

def augment_model_wearing(image):
    """모델 착용 전용 augmentation"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(5),   # 모델은 회전 제한
        transforms.RandomErasing(p=0.2), # 일부 가림
    ])
```

### 📊 예상 성능

**Before (현재):**
```
Query: 9oz 평면 제품
Target: Naver 모델 착용

Top-5 Accuracy: ~40% (추정)
Cosine Similarity: 0.5~0.6
```

**After (Multi-Domain Training):**
```
Query: 9oz 평면 제품
Target: Naver 모델 착용

Top-5 Accuracy: 70~80% (목표)
Cosine Similarity: 0.75~0.85
```

### 💰 비용 분석

| 항목 | 비용 |
|------|------|
| 데이터셋 다운로드 | 무료 (DeepFashion2는 공개) |
| 데이터 전처리 | 3~5일 (스크립트 작성) |
| GPU 학습 시간 | RTX 4090: 2~3일 (22K 이미지, 100 epochs) |
| 스토리지 | ~50GB (이미지 + 체크포인트) |
| **총 시간** | **1주일** |

### ✅ 장점
- ✅ **근본적 해결**: 모델이 두 도메인을 모두 이해
- ✅ **확장 가능**: 다른 도메인(마네킹, 3D 렌더링 등) 추가 가능
- ✅ **추가 인프라 불필요**: 기존 학습 파이프라인만 수정
- ✅ **검증 가능**: 학습 중 validation으로 성능 측정

### ❌ 단점
- ❌ **데이터 수집 필요**: 평면 제품 이미지 10K+ 필요
- ❌ **학습 시간 증가**: 데이터 10배 → 학습 시간 3~5배
- ❌ **메타데이터 매칭 어려움**: 평면 이미지에도 category, style 필요

### 🛠️ 구현 예시

```python
# src/data/multi_domain_dataset.py
class MultiDomainFashionDataset(Dataset):
    def __init__(self,
                 k_fashion_path: str,
                 deepfashion_path: str,
                 image_size: int = 224):

        # K-Fashion 데이터 (모델 착용)
        self.k_fashion = load_kfashion_data(k_fashion_path)

        # DeepFashion2 데이터 (평면 + 착용)
        self.deepfashion = load_deepfashion2_data(deepfashion_path)

        # 통합
        self.items = self.k_fashion + self.deepfashion

    def __getitem__(self, idx):
        item = self.items[idx]

        # 도메인별 augmentation
        if item.image_type == 'flat_product':
            image = augment_flat_product(item.image)
        else:
            image = augment_model_wearing(item.image)

        return {
            'image': image,
            'json_data': item.metadata,
            'domain': item.image_type
        }
```

---

## 옵션 2: Virtual Try-On (실험적 ⭐⭐⭐)

### 📋 핵심 아이디어
- 9oz 평면 제품 이미지를 **가상으로 모델이 착용한 것처럼** 변환
- 변환된 이미지로 검색 수행
- 검색 대상(Naver)과 도메인 일치

### 🔧 구현 단계

#### 1단계: Virtual Try-On 모델 선택

| 모델 | 품질 | 속도 | 설치 난이도 |
|------|------|------|------------|
| **VITON-HD** | ⭐⭐⭐⭐ | 느림 (5초/이미지) | 높음 |
| **DM-VTON** | ⭐⭐⭐⭐⭐ | 매우 느림 (10초) | 매우 높음 |
| **TryOnDiffusion** | ⭐⭐⭐⭐⭐ | 느림 | 높음 |

**추천: VITON-HD** (품질과 속도 균형)

#### 2단계: 파이프라인 구성

```
┌─────────────────┐
│  9oz 평면 제품   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Virtual Try-On │  <-- VITON-HD 모델
│  (평면 → 착용)   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 가상 착용 이미지 │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  FashionCLIP    │
│  임베딩 생성     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Naver 검색      │
└─────────────────┘
```

#### 3단계: VITON-HD 설치

```bash
# 1. 저장소 클론
git clone https://github.com/shadow2496/VITON-HD.git
cd VITON-HD

# 2. 환경 설정
conda create -n viton python=3.8
conda activate viton
pip install torch torchvision
pip install opencv-python pillow

# 3. 체크포인트 다운로드 (사전 학습 모델)
# ~5GB, Google Drive에서 다운로드
wget https://drive.google.com/...

# 4. 테스트
python test.py --dataroot ./datasets/test \
               --name test \
               --gpu_ids 0
```

#### 4단계: API 통합

```python
# api/virtual_tryon.py
class VirtualTryOnService:
    def __init__(self, checkpoint_path: str):
        self.model = load_viton_model(checkpoint_path)

    def convert_flat_to_wearing(self,
                                 flat_image: Image.Image,
                                 model_image: Optional[Image.Image] = None) -> Image.Image:
        """
        평면 제품을 가상 착용 이미지로 변환

        Args:
            flat_image: 평면 제품 이미지
            model_image: 기준 모델 포즈 (없으면 기본 포즈 사용)

        Returns:
            가상 착용 이미지
        """
        # 1. 평면 이미지 전처리
        cloth = preprocess_cloth(flat_image)

        # 2. 모델 포즈 선택
        if model_image is None:
            model_image = self.default_model_poses[0]

        # 3. Virtual Try-On 수행
        result = self.model.inference(cloth, model_image)

        return result

# 검색 API에 통합
@app.post("/api/recommend/top10_to_new")
async def recommend_with_tryon(file: UploadFile):
    # 1. 9oz 평면 이미지 로드
    flat_image = Image.open(file.file)

    # 2. 가상 착용 변환
    wearing_image = virtual_tryon_service.convert_flat_to_wearing(flat_image)

    # 3. 임베딩 생성 (변환된 이미지로)
    embedding = fashion_clip.encode_image(wearing_image)

    # 4. Naver 검색
    results = search_naver_products(embedding)

    return results
```

### 📊 예상 성능

**Before:**
```
9oz 평면 → FashionCLIP → Naver
Accuracy: ~40%
```

**After:**
```
9oz 평면 → Virtual Try-On → FashionCLIP → Naver
Accuracy: 60~70% (목표)
```

### 💰 비용 분석

| 항목 | 비용 |
|------|------|
| VITON-HD 설치 | 2~3일 (환경 구성, 디버깅) |
| 체크포인트 다운로드 | ~5GB |
| GPU 추론 시간 | **5초/이미지** (매우 느림!) |
| 배치 처리 | 4621개 → 6.4시간 (GPU 1대) |
| API 응답 시간 | **+5초** (실시간 불가능) |

### ✅ 장점
- ✅ **추가 학습 불필요**: 사전 학습된 모델 사용
- ✅ **시각적으로 명확**: 변환 결과를 눈으로 확인 가능
- ✅ **9oz 데이터만 처리**: 4621개만 변환하면 됨

### ❌ 단점
- ❌ **매우 느림**: 5초/이미지 (실시간 검색 불가능)
- ❌ **품질 불안정**: 복잡한 옷은 변환 실패 가능
- ❌ **GPU 필수**: CPU로는 너무 느림 (30초/이미지)
- ❌ **설치 복잡**: 의존성 문제 많음

### 🎯 적용 시나리오

**시나리오 1: 배치 처리 (권장)**
```python
# 밤에 4621개를 미리 변환
python scripts/batch_virtual_tryon.py \
    --input data/csv/internal_products_rows.csv \
    --output data/virtual_tryon_results/

# 변환된 이미지를 검색에 사용
```

**시나리오 2: 실시간 (비권장)**
```python
# API 호출 시마다 변환 → 너무 느림!
# 5초 대기는 UX 관점에서 허용 불가
```

---

## 옵션 3: Dual Encoder (고급 ⭐⭐⭐⭐)

### 📋 핵심 아이디어
- **두 개의 인코더** 사용:
  - Encoder A: 평면 제품 전용
  - Encoder B: 모델 착용 전용
- 두 인코더가 같은 임베딩 공간에 매핑되도록 학습
- "같은 제품"은 평면이든 착용이든 임베딩이 가깝게

### 🔧 구현 단계

#### 1단계: 아키텍처 설계

```
         ┌───────────────────┐
         │   Flat Product    │
         │   Encoder         │
         │   (ViT-B/16)      │
         └─────────┬─────────┘
                   │
                   v
         ┌─────────────────┐
         │  Embedding Space │ (512-dim)
         │  (공유 공간)      │
         └─────────────────┘
                   ^
                   │
         ┌─────────┴─────────┐
         │  Model Wearing    │
         │  Encoder          │
         │  (ViT-B/16)       │
         └───────────────────┘
```

#### 2단계: Alignment Loss

```python
class DualEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat_encoder = ViT_B_16()
        self.wearing_encoder = ViT_B_16()

    def forward(self, flat_img, wearing_img):
        flat_emb = self.flat_encoder(flat_img)
        wearing_emb = self.wearing_encoder(wearing_img)

        return flat_emb, wearing_emb

# Loss 함수
def alignment_loss(flat_emb, wearing_emb, json_emb, labels):
    """
    같은 제품이면 임베딩이 가깝게
    """
    # 1. Contrastive Loss (평면 - JSON)
    loss1 = contrastive_loss(flat_emb, json_emb, labels)

    # 2. Contrastive Loss (착용 - JSON)
    loss2 = contrastive_loss(wearing_emb, json_emb, labels)

    # 3. Alignment Loss (평면 - 착용)
    # 같은 제품이면 거리가 0이 되도록
    loss3 = mse_loss(flat_emb, wearing_emb)

    return loss1 + loss2 + 0.5 * loss3
```

#### 3단계: 학습 데이터 구성

**필요 데이터:**
- "같은 제품"의 평면 이미지 + 착용 이미지 쌍 (최소 5K 쌍)

```python
training_data = [
    {
        'product_id': 'ABC123',
        'flat_image': 'flat_ABC123.jpg',
        'wearing_image': 'model_ABC123.jpg',
        'metadata': {...}
    },
    # ...
]
```

**문제:** 9oz와 Naver는 **다른 제품**이라 직접 쌍을 만들 수 없음!

**해결책:**
1. DeepFashion2에서 쌍 데이터 사용 (평면+착용 모두 있음)
2. 또는 상용 데이터셋 구매 (Zalando 등)

#### 4단계: 추론

```python
# 검색 시
query_flat_image = load_9oz_image(...)
query_embedding = dual_encoder.flat_encoder(query_flat_image)

# Naver 이미지는 wearing encoder로 인코딩되어 있음
naver_embeddings = [dual_encoder.wearing_encoder(img) for img in naver_images]

# 같은 공간에 있으므로 직접 비교 가능
similarities = cosine_similarity(query_embedding, naver_embeddings)
```

### 📊 예상 성능

**성능:**
```
Top-5 Accuracy: 75~85%
Cosine Similarity: 0.8~0.9
```

### 💰 비용 분석

| 항목 | 비용 |
|------|------|
| 데이터셋 수집 | **5K+ 쌍 필요** (가장 큰 장벽) |
| 모델 구현 | 2~3일 |
| 학습 시간 | RTX 4090: 3~5일 |
| 메모리 사용량 | 2배 (인코더 2개) |
| **총 시간** | **2주** |

### ✅ 장점
- ✅ **이론적으로 최고 성능**: 도메인별 최적화
- ✅ **확장 가능**: 인코더 추가 가능 (마네킹용 등)
- ✅ **추론 속도**: Virtual Try-On보다 훨씬 빠름

### ❌ 단점
- ❌ **데이터 수집 어려움**: 평면+착용 쌍이 필요
- ❌ **복잡도 증가**: 모델 2배, 학습 로직 복잡
- ❌ **메모리 2배**: GPU 메모리 부족 가능

---

## 🏆 종합 비교

| 항목 | Multi-Domain | Virtual Try-On | Dual Encoder |
|------|-------------|---------------|-------------|
| **구현 난이도** | ⭐⭐ (쉬움) | ⭐⭐⭐⭐ (어려움) | ⭐⭐⭐ (보통) |
| **데이터 수집** | 평면 10K | 불필요 | 평면+착용 쌍 5K |
| **학습 시간** | 2~3일 | 없음 | 3~5일 |
| **추론 속도** | 빠름 (100ms) | **매우 느림 (5초)** | 빠름 (100ms) |
| **예상 정확도** | 70~80% | 60~70% | 75~85% |
| **비용** | 낮음 | 낮음 | 중간 |
| **유지보수** | 쉬움 | 어려움 | 보통 |

## 🎯 최종 권장 사항

### 1차 추천: **Multi-Domain Training**
- ✅ 구현 쉬움
- ✅ 근본적 해결
- ✅ 확장 가능
- ✅ 빠른 추론

### 2차 옵션: **Dual Encoder**
- 데이터 수집 가능하면 최고 성능

### 비추천: **Virtual Try-On**
- 너무 느림 (실시간 불가)
- 배치 처리용으로만 고려

---

## 📝 다음 단계

1. **Multi-Domain Training 시작**
   ```bash
   # DeepFashion2 다운로드
   python scripts/download_deepfashion2.py

   # 데이터 전처리
   python scripts/prepare_multi_domain_data.py

   # 학습
   python scripts/training/train_multi_domain.py
   ```

2. **성능 측정**
   - Validation set으로 정확도 측정
   - Domain별 성능 비교

3. **점진적 개선**
   - 데이터 추가
   - Augmentation 튜닝
   - Hyperparameter 최적화
