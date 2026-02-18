# DeepFashion2 vs K-Fashion 비교

## 📊 데이터 규모 비교

| 항목 | K-Fashion | DeepFashion2 |
|------|-----------|--------------|
| **총 이미지 수** | 2,172 | **491,000** (225배) |
| **카테고리 수** | 23개 (스타일) | 13개 (의류 타입) |
| **어노테이션** | JSON (메타데이터) | JSON (bbox, pose, 세그멘테이션) |
| **이미지 타입** | 모델 착용만 | **평면 + 모델 착용** 🎯 |
| **크기** | ~5GB | **30GB+** |
| **라이센스** | 연구용 | 연구용 |

## 🗂️ 데이터 구조 비교

### K-Fashion 구조
```
K-fashion/
├── Training/
│   └── 라벨링데이터/
│       ├── 레트로/
│       │   ├── image/        # 이미지
│       │   └── label/        # JSON (메타데이터)
│       ├── 로맨틱/
│       └── ...
└── Validation/
```

**JSON 구조 (복잡):**
```json
{
  "데이터셋 정보": {
    "데이터셋 상세설명": {
      "라벨링": {
        "스타일": [{"스타일": "레트로"}],
        "상의": [{"카테고리": "셔츠"}],
        "polygon": [[x1,y1], [x2,y2], ...]
      }
    }
  }
}
```

### DeepFashion2 구조 (더 간단! ✅)
```
deepfashion2/
├── train/
│   ├── image/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── annos/
│       ├── 000001.json
│       ├── 000002.json
│       └── ...
└── validation/
    ├── image/
    └── annos/
```

**JSON 구조 (훨씬 간단! ✅):**
```json
{
  "source": "user",                    # 'user' (모델 착용) or 'shop' (평면)
  "pair_id": 123,
  "item": {
    "1": {
      "category_name": "short_sleeved_shirt",
      "category_id": 1,
      "bounding_box": [x1, y1, x2, y2],
      "landmarks": [[x1,y1], [x2,y2], ...],  # 옷의 키포인트
      "segmentation": [[...]]                 # 세그멘테이션 마스크
    }
  }
}
```

## ✅ 장점: DeepFashion2가 더 사용하기 쉬운 이유

### 1. 구조가 단순함
```python
# K-Fashion (복잡)
data = json.load(f)
style = data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['스타일'][0]['스타일']

# DeepFashion2 (간단!)
data = json.load(f)
category = data['item']['1']['category_name']
```

### 2. 이미지 타입 구분이 명확
```python
# 'source' 필드로 바로 구분 가능
if data['source'] == 'shop':
    image_type = 'flat_product'  # 평면 제품
elif data['source'] == 'user':
    image_type = 'model_wearing'  # 모델 착용
```

### 3. Bounding Box가 이미 계산되어 있음
```python
# K-Fashion: polygon을 bbox로 변환해야 함 (귀찮음)
polygon = [[10,20], [50,20], [50,80], [10,80]]
bbox = polygon_to_bbox(polygon)  # 직접 구현 필요

# DeepFashion2: 이미 bbox 제공! (편함)
bbox = data['item']['1']['bounding_box']  # [x, y, w, h]
```

### 4. 카테고리가 표준화됨
```python
# K-Fashion: 한글, 스타일 중심
categories = ['레트로', '로맨틱', '리조트', ...]  # 23개

# DeepFashion2: 영어, 의류 타입
categories = [
    'short_sleeved_shirt',   # 반팔 셔츠
    'long_sleeved_shirt',    # 긴팔 셔츠
    'trousers',              # 바지
    'skirt',                 # 치마
    # ... 13개
]
```

## ❌ 단점

### 1. 데이터가 너무 큼 (30GB+)
**해결책:**
- 전체 다운로드 대신 **10K 샘플만** 사용
- 우리는 20K면 충분 (K-Fashion 2K → 10배 증가)

### 2. 수동 다운로드 필요
**이유:**
- Google Drive에서만 제공 (직접 다운로드 링크 없음)
- API 없음

**해결책:**
```bash
# 1. 브라우저에서 다운로드 (한 번만)
# 2. 압축 해제
# 3. 스크립트로 파싱 (자동화)
```

### 3. 메타데이터가 부족
- K-Fashion: 스타일, 소재, 디테일 등 풍부
- DeepFashion2: 카테고리만 (스타일 정보 없음)

**해결책:**
- 카테고리만 사용 (충분!)
- 또는 스타일은 K-Fashion 것 사용

## 🎯 우리 프로젝트에서 사용 방법

### 간단한 3단계!

#### 1단계: 다운로드 (한 번만, 10분)
```bash
# 브라우저에서:
# https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok

# train/image/ 폴더에서 10K 이미지만 선택
# train/annos/ 폴더 전체 다운로드
# → data/deepfashion2/ 에 압축 해제
```

#### 2단계: 파싱 (자동, 5분)
```bash
# 우리가 만든 스크립트 실행
python scripts/data_collection/parse_deepfashion2.py \
    --deepfashion_dir data/deepfashion2 \
    --create_multi_domain

# 결과: data/multi_domain/multi_domain_dataset.csv 생성
```

#### 3단계: 학습 (자동)
```python
from src.data.multi_domain_dataset import MultiDomainFashionDataset

# 이미 우리가 만든 클래스로 바로 사용!
dataset = MultiDomainFashionDataset(
    csv_path='data/multi_domain/multi_domain_dataset.csv'
)

# 끝! 학습 시작
```

## 📝 파싱 스크립트 설명

이미 만들어둔 `parse_deepfashion2.py`가 알아서 다 해줍니다:

```python
# 스크립트가 하는 일:
def parse_deepfashion2_annotation(anno_path):
    # 1. JSON 로드
    data = json.load(open(anno_path))

    # 2. 이미지 타입 자동 판별
    if data['source'] == 'shop':
        image_type = 'flat_product'  # 평면 ✅
    else:
        image_type = 'model_wearing'  # 착용 ✅

    # 3. 카테고리 변환
    category = data['item']['1']['category_name']
    kfashion_category = CATEGORY_MAPPING[category]  # 자동 매핑

    # 4. CSV로 저장 (간단!)
    return {
        'image_path': '...',
        'image_type': image_type,      # flat_product / model_wearing
        'category': kfashion_category,  # 상의 / 하의 / 아우터
        'bbox': [x, y, w, h]
    }
```

**결과 CSV:**
```csv
image_path,image_type,category,bbox
data/deepfashion2/train/image/000001.jpg,flat_product,상의,"[10,20,100,150]"
data/deepfashion2/train/image/000002.jpg,model_wearing,하의,"[15,30,120,180]"
```

## 🚀 학습 필요 없음!

### 우리가 이미 다 만들어뒀습니다! ✅

1. ✅ **다운로드 가이드** - `download_deepfashion2.py`
2. ✅ **파싱 스크립트** - `parse_deepfashion2.py` (JSON → CSV)
3. ✅ **Dataset 클래스** - `MultiDomainFashionDataset` (바로 사용)
4. ✅ **DataLoader** - `create_multi_domain_dataloader()` (바로 사용)

### 당신이 할 일:

1. **브라우저에서 다운로드** (10분)
   - 클릭만 하면 됨

2. **스크립트 실행 2줄** (5분)
   ```bash
   python scripts/data_collection/parse_deepfashion2.py --create_multi_domain
   python scripts/training/train_multi_domain.py  # (다음에 만들 예정)
   ```

3. **끝!**

## 💡 요약

| 질문 | 답변 |
|------|------|
| **구조화 잘 되어 있음?** | ✅ **K-Fashion보다 더 간단!** |
| **데이터양 많음?** | ✅ **491K (K-Fashion의 225배)** |
| **따로 학습 필요?** | ❌ **필요 없음! 스크립트가 다 해줌** |

## 🎯 추천: 샘플로 테스트

전체 다운로드 전에 샘플로 테스트:

```bash
# 1. 샘플 폴더 생성
python scripts/data_collection/download_deepfashion2.py --create_sample

# 2. K-Fashion 이미지 몇 개를 복사해서 테스트
cp data/images/sample_flat_*.jpg data/deepfashion2/sample/train/flat_product/
cp data/images/sample_model_*.jpg data/deepfashion2/sample/train/model_wearing/

# 3. MultiDomainFashionDataset 테스트
python src/data/multi_domain_dataset.py
```

## 📚 참고

- DeepFashion2는 **K-Fashion보다 쉬움!**
- **JSON이 더 단순**하고 **이미지 타입 구분이 명확**
- 우리가 만든 스크립트로 **5분이면 파싱 완료**
- **학습 코드 추가 없음** - 기존 코드 그대로 사용!
