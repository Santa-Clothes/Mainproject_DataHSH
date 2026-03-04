# K-Fashion 이미지 기반 유사 상품 검색 시스템

---

## 1. 프로젝트 개요

**목표**: 패션 이미지를 업로드하면 네이버 쇼핑에서 시각적으로 유사한 상품을 실시간으로 찾아주는 검색 시스템 구축

**핵심 아이디어**: 텍스트가 아닌 **이미지의 시각적 특징(벡터)** 으로 유사도를 측정

```
사용자 이미지 업로드
        ↓
  FashionCLIP 모델
        ↓
  768차원 임베딩 벡터
        ↓
  FAISS 고속 벡터 검색
        ↓
  네이버 쇼핑 유사 상품 반환
```

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        클라이언트                            │
│              웹 UI (search.html) / Spring 백엔드             │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP REST
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI 서버 (포트 8001)                    │
│                    search_api.py                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ /search     │  │ /embed       │  │ /analyze           │  │
│  │ /upload     │  │ 768차원 벡터 │  │ 임베딩 + 스타일    │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬─────────┘  │
│         └────────────────┼─────────────────────┘            │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐   │
│  │              SearchPipeline (search_pipeline.py)       │   │
│  │                                                        │   │
│  │  ┌─────────────────────┐   ┌────────────────────────┐ │   │
│  │  │ FashionCLIP         │   │ FaissVectorIndex       │ │   │
│  │  │ (embedding_         │   │ (vector_index.py)      │ │   │
│  │  │  generator.py)      │   │                        │ │   │
│  │  │  이미지→768차원     │   │  IndexFlatIP           │ │   │
│  │  │  CLIPVisionModel    │   │  7,538 벡터            │ │   │
│  │  └─────────────────────┘   └────────────────────────┘ │   │
│  └───────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐   │
│  │         Supabase (PostgreSQL)                          │   │
│  │   naver_products 테이블: 상품 메타데이터 조회          │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 모델: FashionCLIP

### 3-1. 모델 선택 이유

| 모델 | 학습 도메인 | 패션 이미지 적합성 |
|------|------------|------------------|
| OpenAI CLIP | 범용 (웹 이미지) | 보통 |
| **FashionCLIP** (`patrickjohncyh/fashion-clip`) | **패션 특화** | **높음** |

- CLIP을 **패션 이미지-텍스트 쌍**으로 파인튜닝한 모델
- ViT (Vision Transformer) 백본
- 패션 속성(색상, 소재, 실루엣)에 민감한 임베딩 생성

### 3-2. 임베딩 생성 과정

```
이미지 (PIL)
    ↓
Resize (224×224) + Normalize
    ↓
CLIPVisionModel.forward()
    ↓
pooler_output  →  768차원 벡터
    ↓
L2 Normalize  →  단위 구 위의 벡터 (코사인 유사도 계산 가능)
```

```python
# models/embedding_generator.py 핵심 코드
outputs = self.model(pixel_values=image_tensor)
embedding = outputs.pooler_output        # [1, 768]
embedding = F.normalize(embedding, p=2)  # L2 정규화
```

### 3-3. 왜 768차원인가?

- ViT-B/32 기준 hidden size = **768**
- 고차원일수록 더 세밀한 시각 특징 표현 가능
- FAISS IndexFlatIP로 코사인 유사도 검색 가능 (정규화 벡터 내적 = 코사인 유사도)

---

## 4. 벡터 검색: FAISS

### 4-1. 인덱스 구조

| 항목 | 값 |
|------|----|
| 인덱스 타입 | `IndexFlatIP` (Inner Product) |
| 저장 벡터 수 | 7,538개 (Naver 상품) |
| 벡터 차원 | 768 |
| 검색 방식 | 완전 탐색 (Exact Search) |
| 인덱스 파일 | `data/indexes/naver.index` (23MB) |

### 4-2. 검색 과정

```
쿼리 임베딩 (768차원, L2 정규화)
        ↓
FAISS IndexFlatIP.search(query, k=100)
        ↓
Inner Product = 코사인 유사도 (정규화 벡터이므로)
        ↓
상위 100개 product_id + 점수 반환
        ↓
카테고리 필터링 (BL/OP/SK 등 K-Fashion 코드)
        ↓
최종 Top-K 결과
```

### 4-3. 검색 속도

- FAISS 완전 탐색: 7,538개 기준 **수 ms 이내**
- 전체 응답 시간: 이미지 업로드 포함 약 **100~300ms**

---

## 5. 스타일 분류기

### 5-1. 구조

```
FashionCLIP visual_projection 출력 (512차원)
        ↓
Linear(512 → 256)
        ↓
ReLU
        ↓
Dropout(0.3)
        ↓
Linear(256 → 23)  ← 23개 K-Fashion 스타일
        ↓
Softmax
        ↓
23개 확률을 10개 대분류로 합산
```

### 5-2. 23개 세부 스타일 → 10개 대분류 매핑

| 대분류 | 세부 스타일 |
|--------|------------|
| 트래디셔널 | 클래식, 프레피 |
| 매니시 | 매니시, 톰보이 |
| 페미닌 | 페미닌, 로맨틱, 섹시 |
| 에스닉 | 히피, 웨스턴, 오리엔탈 |
| 컨템포러리 | 모던, 소피스트케이티드, 아방가르드 |
| 내추럴 | 컨트리, 리조트 |
| 젠더리스 | 젠더리스 |
| 스포티 | 스포티 |
| 서브컬처 | 레트로, 키치, 힙합, 펑크 |
| 캐주얼 | 밀리터리, 스트리트 |

### 5-3. 성능

| 지표 | 수치 |
|------|------|
| Top-1 Accuracy | 45.3% |
| Top-3 Accuracy | 69.6% |

---

## 6. 데이터

| 데이터 | 설명 | 규모 |
|--------|------|------|
| 나인온스 (Nine Oz) | 쿼리용 내부 상품 (Supabase) | 4,616개 |
| 네이버 쇼핑 | 검색 대상 상품 (Supabase) | 6,778개 |
| FAISS 인덱스 | 네이버 상품 임베딩 벡터 | 7,538개 |

**카테고리 코드** (K-Fashion 표준):
`BL`(블라우스), `OP`(원피스), `SK`(스커트), `PT`(팬츠), `JK`(재킷),
`CT`(코트), `KN`(니트), `TS`(티셔츠), `JP`(점프수트), `SH`(셔츠)

---

## 7. 검색 파이프라인 전체 흐름

```
[입력] 이미지 파일 (JPG/PNG)
        ↓
[1] PIL로 읽기 → RGB 변환
        ↓
[2] FashionCLIP 임베딩 생성
        이미지 → 224×224 리사이즈 → ViT 인코딩 → 768차원 → L2 정규화
        ↓
[3] FAISS 검색 (top_k=100)
        Inner Product로 코사인 유사도 계산 → 상위 100개 추출
        ↓
[4] 카테고리 필터링 (선택)
        BL, OP, SK 등 K-Fashion 코드로 동일 카테고리만 필터
        ↓
[5] 최종 랭킹
        유사도 점수 내림차순 → Top-K 반환
        ↓
[6] Supabase 메타데이터 조회
        product_id → title, price, image_url, category_id, style_id
        ↓
[출력] JSON 응답 (상품 정보 + 유사도 점수)
```

---

## 8. API 엔드포인트

| 메서드 | 경로 | 설명 | 주요 입력 |
|--------|------|------|----------|
| GET | `/health` | 서버 상태 확인 | - |
| POST | `/search/upload` | 이미지 업로드 → 유사 상품 검색 | `file`, `top_k`, `category_filter` |
| GET | `/search` | 나인온스 인덱스 → 검색 | `query_index`, `final_k` |
| POST | `/embed` | 이미지 → 768차원 임베딩 벡터 | `file` |
| POST | `/analyze` | 이미지 → 임베딩 + 스타일 분류 Top-3 | `file` |
| GET | `/categories` | 카테고리 통계 | - |
| GET | `/stats` | 데이터셋 통계 | - |

### 호출 예시

```bash
# 이미지 검색
curl -X POST http://localhost:8001/search/upload \
  -F "file=@image.jpg" \
  -F "top_k=9"

# 응답 예시
{
  "results": [
    {
      "product_id": "12345",
      "title": "오버사이즈 블라우스",
      "price": 39000,
      "image_url": "https://...",
      "category_id": "BL",
      "score": 0.892
    }, ...
  ],
  "metrics": {
    "search_time_ms": 48,
    "faiss_enabled": true
  }
}

# 임베딩 + 스타일 분류
curl -X POST http://localhost:8001/analyze -F "file=@image.jpg"
# → { "embedding": [...768개...], "styles": [{"style":"페미닌","score":0.41}, ...] }
```

---

## 9. 검색 성능

| 지표 | 수치 |
|------|------|
| Top-1 Accuracy | 44% |
| Top-5 Accuracy | 78% |
| Top-10 Accuracy | 88% |
| MRR (Mean Reciprocal Rank) | 0.58 |
| 검색 대상 상품 수 | 7,538개 |
| 평균 검색 시간 | < 100ms |

> **MRR 0.58** = 평균적으로 정답 상품이 약 2번째 위치 이내에 등장

---

## 10. 임베딩 공간 시각화 (t-SNE / UMAP)

`visualize_tsne_umap.png` 참조

- **t-SNE**: 768차원 → 2차원, perplexity=30
- **UMAP**: 768차원 → 2차원, cosine metric, n_neighbors=15
- 샘플: 2,000개, 카테고리(BL/OP/SK 등)별 색상 구분
- 관찰: FashionCLIP은 카테고리보다 **시각적 스타일/색감** 기준으로 군집 형성
  → 색상·소재가 유사한 상품끼리 벡터 공간에서 가깝게 위치

---

## 11. 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 / 프레임워크 | Python 3.11, FastAPI |
| AI 모델 | FashionCLIP (ViT, HuggingFace) |
| 딥러닝 | PyTorch, torchvision |
| 벡터 검색 | FAISS (faiss-cpu, IndexFlatIP) |
| 데이터베이스 | Supabase (PostgreSQL) |
| 웹 UI | HTML/CSS/JS (Vanilla) |
| 서버 | Uvicorn (ASGI) |
| Spring 연동 | REST API (multipart/form-data) |

---

## 12. 프로젝트 구조

```
FinalProject_v2/
├── api/
│   ├── search_api.py        # FastAPI 서버 (엔드포인트 정의)
│   ├── search_pipeline.py   # 검색 파이프라인 (임베딩→FAISS→필터→랭킹)
│   └── vector_index.py      # FAISS 인덱스 로드/검색
├── models/
│   └── embedding_generator.py   # FashionCLIP 임베딩 생성기
├── utils/
│   ├── config.py            # 환경변수 기반 시스템 설정
│   └── supabase_loader.py   # Supabase 데이터 로더
├── checkpoints/
│   └── style_classifier.pt  # 학습된 MLP 스타일 분류기
├── data/
│   └── indexes/
│       ├── naver.index      # FAISS 벡터 인덱스 (7,538 × 768)
│       └── naver.ids.npy    # FAISS 인덱스 ↔ product_id 매핑
├── static/
│   └── search.html          # 웹 검색 UI
├── visualize.py             # t-SNE / UMAP 시각화 스크립트
├── .env                     # 환경변수 (Supabase URL/KEY 등)
└── requirements.txt
```

---

## 13. 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 서버 실행
uvicorn api.search_api:app --host 0.0.0.0 --port 8001 --reload

# 3. 웹 UI 접속
http://localhost:8001/static/search.html

# 4. API 문서
http://localhost:8001/docs
```

---

## 14. 한계 및 향후 개선 방향

| 한계 | 개선 방향 |
|------|----------|
| Top-1 정확도 44%로 제한적 | 도메인 특화 파인튜닝 추가 학습 |
| 카테고리 필터 후 결과 수 감소 가능 | 카테고리 분류기 별도 학습 |
| 네이버 상품 7,538개로 검색 풀 제한 | 크롤링으로 검색 풀 확대 |
| FAISS 완전 탐색 (대규모 시 속도 저하) | IVF/HNSW 인덱스로 전환 |
| 스타일 분류 Top-1 45.3% | 더 많은 학습 데이터 확보 |
