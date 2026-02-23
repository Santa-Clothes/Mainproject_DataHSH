# Fashion Search - Spring 백엔드 연동 가이드

**대상:** Spring 백엔드 담당자
**FastAPI 서버:** `http://localhost:8001` (AI 임베딩 전용)

---

## 아키텍처

```
[Frontend]
     │  이미지 파일
     ▼
[Spring]  ← 모든 비즈니스 로직 + DB 접근 여기서만
     │
     ├─── POST http://localhost:8001/embed    ──▶  [FastAPI AI 서버]
     │         이미지 → 768차원 벡터 반환              FashionCLIP 모델
     │
     ├─── POST http://localhost:8001/analyze  ──▶  [FastAPI AI 서버]
     │         이미지 → 768차원 벡터 + K-Fashion 스타일 분류 (Top-3)
     │         (나인오즈 이미지 분석용)
     │
     └─── Supabase RPC 호출 (pgvector)
              match_naver_products(embedding, 100)
              유사도 높은 상품 top-N 반환
     │
     ▼
[Frontend] 결과 반환
```

### 역할 분리

| 담당 | 역할 |
|---|---|
| **FastAPI** | 이미지 → 768차원 임베딩 + 스타일 분류 담당 (DB 접근 없음) |
| **Spring** | 임베딩/분석 요청, Supabase 조회, 결과 가공, 프론트 응답 |
| **Supabase** | pgvector로 벡터 유사도 검색 |

---

## 1단계: Supabase DB 설정 (최초 1회)

Supabase 대시보드 → SQL Editor에서 실행:

```sql
-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 유사도 검색 함수 생성
CREATE OR REPLACE FUNCTION match_naver_products(
  query_embedding vector(768),
  match_count int DEFAULT 100
)
RETURNS TABLE (
  product_id text,
  title text,
  price int,
  image_url text,
  category_id text,
  kfashion_category text,
  similarity float
)
LANGUAGE sql STABLE AS $$
  SELECT
    product_id::text,
    title,
    price,
    image_url,
    category_id,
    kfashion_item_category AS kfashion_category,
    1 - (embedding <=> query_embedding) AS similarity
  FROM naver_products
  WHERE embedding IS NOT NULL
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;

-- 검색 성능용 인덱스 (선택, 데이터 많을 때 유효)
CREATE INDEX IF NOT EXISTS naver_products_embedding_idx
  ON naver_products
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
```

---

## 2단계: FastAPI 임베딩 서버 API

### POST /embed — 이미지 → 임베딩 벡터 (검색용)

```
URL:     http://localhost:8001/embed
Method:  POST
Body:    multipart/form-data  { file: 이미지파일 }
```

**응답:**
```json
{
  "embedding": [0.023, -0.145, 0.872, ...],  // 768개 float
  "dimension": 768
}
```

**curl 테스트:**
```bash
curl -X POST http://localhost:8001/embed \
  -F "file=@./image.jpg"
```

---

### POST /analyze — 이미지 → 임베딩 + 스타일 분류 (나인오즈 이미지 분석용)

입력 이미지의 K-Fashion 스타일을 분석하고 임베딩을 함께 반환합니다.
FashionCLIP + MLP 분류기 (K-Fashion 23개 스타일, Top-1 정확도 45.3%).

```
URL:     http://localhost:8001/analyze
Method:  POST
Body:    multipart/form-data  { file: 이미지파일 }
Params:  top_k=3  (반환할 스타일 수, 기본값 3)
```

**응답:**
```json
{
  "embedding": [0.023, -0.145, 0.872, ...],  // 768개 float (검색용 동일 벡터)
  "dimension": 768,
  "styles": [
    { "style": "밀리터리", "score": 0.2702 },
    { "style": "스트리트", "score": 0.1220 },
    { "style": "스포티",   "score": 0.1012 }
  ]
}
```

> **score 해석:** softmax 확률값. 23개 클래스 기준으로 랜덤 = 0.043(4.3%).
> 0.27(27%)이면 랜덤 대비 약 6배 높은 신뢰도.

**curl 테스트:**
```bash
curl -X POST http://localhost:8001/analyze \
  -F "file=@./image.jpg"
```

**K-Fashion 23개 스타일:**
레트로, 로맨틱, 리조트, 매니시, 모던, 밀리터리, 섹시, 소피스트케이티드,
스트리트, 스포티, 아방가르드, 오리엔탈, 웨스턴, 젠더리스, 컨트리,
클래식, 키치, 톰보이, 펑크, 페미닌, 프레피, 히피, 힙합

---

### GET /health — 서버 상태
```bash
curl http://localhost:8001/health
# {"status": "healthy", "model_loaded": true, "style_classifier": true}
```

---

## 3단계: Spring 구현

`integration/examples/` 폴더의 파일을 프로젝트에 복사해서 사용.

```
integration/examples/
├── EmbeddingApiService.java        # FastAPI /embed + /analyze 호출
├── NaverProductService.java        # Supabase pgvector 검색
├── FashionSearchController.java    # 전체 흐름 조합 컨트롤러
└── RestTemplateConfig.java         # HTTP 클라이언트 설정
```

### application.yml 설정

```yaml
fashion:
  embed:
    url: http://localhost:8001   # FastAPI AI 서버

supabase:
  url: https://fjoylosbfvojioljibku.supabase.co
  key: YOUR_SUPABASE_SERVICE_ROLE_KEY   # service_role key (서버에서만 사용)
```

### 이미지 검색 흐름 (기존, /embed 사용)

```java
// 1. 이미지 → 임베딩 (FastAPI /embed 호출)
float[] embedding = embeddingApiService.getEmbedding(imageFile);

// 2. 임베딩 → 유사 상품 검색 (Supabase pgvector)
List<NaverProduct> results = naverProductService.searchSimilar(embedding, topK);

// 3. 결과 반환
return ResponseEntity.ok(results);
```

### 스타일 분석 + 검색 흐름 (신규, /analyze 사용)

```java
// 1. 이미지 → 임베딩 + 스타일 (FastAPI /analyze 호출)
EmbeddingApiService.AnalyzeResponse analysis = embeddingApiService.getAnalysis(imageFile);

// 2. 스타일 정보 활용 (예: 응답에 포함)
List<StylePrediction> styles = analysis.getStyles();
// styles[0] = { style: "밀리터리", score: 0.27 }

// 3. 임베딩으로 유사 상품 검색
float[] embedding = analysis.getEmbeddingArray();
List<NaverProduct> results = naverProductService.searchSimilar(embedding, topK);

// 4. 스타일 + 검색 결과 함께 반환
return ResponseEntity.ok(Map.of("styles", styles, "products", results));
```

---

## Supabase REST API 형식 (pgvector RPC)

Spring에서 Supabase RPC 호출 방법:

```
POST https://{project}.supabase.co/rest/v1/rpc/match_naver_products
Headers:
  apikey: YOUR_SUPABASE_KEY
  Authorization: Bearer YOUR_SUPABASE_KEY
  Content-Type: application/json

Body:
{
  "query_embedding": [0.023, -0.145, 0.872, ...],
  "match_count": 20
}
```

**응답:**
```json
[
  {
    "product_id": "90233826193",
    "title": "오버핏 후드 티셔츠",
    "price": 29900,
    "image_url": "https://...",
    "category_id": "50000803",
    "kfashion_category": "스트리트",
    "similarity": 0.891
  },
  ...
]
```

---

## 트러블슈팅

### FastAPI 서버 연결 안됨
```bash
# 서버 상태 확인
curl http://localhost:8001/health

# AI 서버 재시작 (FinalProject_v2 폴더에서)
uvicorn api.search_api:app --host 0.0.0.0 --port 8001 --reload
```

### Supabase pgvector 함수 없음 오류
SQL Editor에서 1단계 SQL을 다시 실행.

### 임베딩 차원 불일치
FastAPI `/embed` 응답의 `dimension`이 768인지 확인.
Supabase `naver_products.embedding` 컬럼이 `vector(768)` 타입인지 확인.
