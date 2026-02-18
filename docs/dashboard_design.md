# Fashion Search Dashboard 설계

## 📊 KPI (Key Performance Indicators)

### 1. 검색 성능 지표

#### A. Top-K Accuracy
- **Top-1 Accuracy**: 첫 번째 결과가 같은 카테고리일 확률
- **Top-5 Accuracy**: 상위 5개 중 같은 카테고리 포함 확률
- **Top-10 Accuracy**: 상위 10개 중 같은 카테고리 포함 확률

#### B. Precision & Recall
- **Precision@K**: 상위 K개 중 관련 상품 비율
- **Recall@K**: 전체 관련 상품 중 찾은 비율
- **F1@K**: Precision과 Recall의 조화평균

#### C. Ranking Metrics
- **MAP (Mean Average Precision)**: 평균 정밀도
- **MRR (Mean Reciprocal Rank)**: 평균 역순위
- **NDCG@K**: 정규화 할인 누적 이득

### 2. 시스템 성능 지표

#### A. 응답 시간
- **평균 검색 시간**: 임베딩 생성 + FAISS 검색
- **P50, P95, P99 Latency**: 백분위 응답 시간
- **FAISS vs Numpy 속도 비교**

#### B. 처리량
- **QPS (Queries Per Second)**: 초당 처리 쿼리 수
- **일일 검색 수**
- **동시 접속자 수**

#### C. 리소스 사용
- **GPU 메모리 사용률**
- **CPU 사용률**
- **API 서버 메모리 사용량**

### 3. 비즈니스 지표

#### A. 사용자 참여도
- **CTR (Click-Through Rate)**: 검색 결과 클릭률
- **Conversion Rate**: 구매 전환율 (향후)
- **재검색률**: 같은 세션 내 재검색 비율

#### B. 카테고리별 성능
- **카테고리별 Top-5 Accuracy**
- **인기 검색 카테고리**
- **카테고리별 평균 유사도 점수**

#### C. 데이터 품질
- **이미지 로드 실패율**
- **임베딩 생성 실패율**
- **FAISS 인덱스 커버리지**: 전체 DB 대비 인덱싱 비율

---

## 🎨 대시보드 레이아웃

### 1. Overview Dashboard (메인)

```
┌─────────────────────────────────────────────────────────────┐
│  Fashion Search System - Overview                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📊 Real-time Metrics (24h)                                  │
│  ┌───────────┬───────────┬───────────┬───────────┐          │
│  │ Total     │ Avg       │ Top-5     │ Avg       │          │
│  │ Searches  │ Latency   │ Accuracy  │ Score     │          │
│  │ 12,453    │ 0.51s     │ 87.3%     │ 0.623     │          │
│  └───────────┴───────────┴───────────┴───────────┘          │
│                                                               │
│  📈 Performance Trends                                        │
│  [Line Chart: QPS, Latency, Accuracy over time]             │
│                                                               │
│  🎯 Top-K Accuracy (Current)                                 │
│  Top-1:  ████████░░ 78.5%                                   │
│  Top-5:  ███████████░ 87.3%                                 │
│  Top-10: ████████████░ 92.1%                                │
│                                                               │
│  🔥 Hot Categories                                           │
│  [Bar Chart: Search volume by category]                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2. Performance Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  Search Performance Analytics                                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ⏱️ Latency Distribution                                     │
│  [Histogram: Response time distribution]                     │
│  P50: 0.32s  │  P95: 0.78s  │  P99: 1.24s                   │
│                                                               │
│  📊 Precision & Recall                                        │
│  [Line Chart: P@K and R@K for K=1,5,10,20,50]               │
│                                                               │
│  🎲 Score Distribution                                        │
│  [Violin Plot: Similarity score distribution by rank]        │
│                                                               │
│  🔄 FAISS Performance                                         │
│  │ Indexed Vectors: 7,538                                    │
│  │ Index Size: 23 MB                                         │
│  │ Avg Search Time: 0.003s (2000x faster than numpy)        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3. Category Analysis Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  Category Performance Analysis                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📋 Category Breakdown                                        │
│  ┌──────────┬──────────┬──────────┬──────────┐             │
│  │ Category │ Searches │ Top-5    │ Avg      │             │
│  │          │ (24h)    │ Accuracy │ Score    │             │
│  ├──────────┼──────────┼──────────┼──────────┤             │
│  │ BL       │ 2,345    │ 89.2%    │ 0.645    │             │
│  │ OP       │ 1,876    │ 85.1%    │ 0.612    │             │
│  │ SK       │ 1,543    │ 91.3%    │ 0.668    │             │
│  │ ...      │ ...      │ ...      │ ...      │             │
│  └──────────┴──────────┴──────────┴──────────┘             │
│                                                               │
│  🎯 Category Confusion Matrix                                │
│  [Heatmap: Query category vs Retrieved category]            │
│                                                               │
│  📊 Category Trends                                           │
│  [Stacked Area Chart: Category search volume over time]     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔌 API 응답 데이터 구조

### 현재 (기본)
```json
{
  "results": [
    {
      "product_id": "12345",
      "title": "상품명",
      "price": 29900,
      "image_url": "https://...",
      "category_id": "BL",
      "score": 0.856
    }
  ]
}
```

### 개선된 구조 (대시보드용)
```json
{
  "query": {
    "query_id": "uuid-xxx",
    "timestamp": "2026-02-12T10:30:00Z",
    "image_info": {
      "size": "224x298",
      "format": "PNG"
    }
  },
  "results": [
    {
      "rank": 1,
      "product_id": "12345",
      "title": "상품명",
      "price": 29900,
      "image_url": "https://...",
      "category_id": "BL",
      "category_name": "블라우스",
      "score": 0.856,
      "metadata": {
        "brand": "브랜드명",
        "tags": ["여름", "시원한"]
      }
    }
  ],
  "metrics": {
    "total_results": 100,
    "filtered_results": 45,
    "final_results": 10,
    "search_time_ms": 512,
    "embedding_time_ms": 245,
    "faiss_search_time_ms": 3,
    "postprocess_time_ms": 264
  },
  "stats": {
    "avg_score": 0.623,
    "max_score": 0.856,
    "min_score": 0.445,
    "score_distribution": {
      "0.8-1.0": 2,
      "0.6-0.8": 5,
      "0.4-0.6": 3
    }
  }
}
```

---

## 💾 데이터베이스 스키마

### 1. search_logs 테이블 (검색 로그)
```sql
CREATE TABLE search_logs (
    id SERIAL PRIMARY KEY,
    query_id UUID NOT NULL,
    timestamp TIMESTAMP NOT NULL,

    -- 쿼리 정보
    query_image_url TEXT,
    query_image_hash VARCHAR(64),  -- 중복 검색 감지
    query_category VARCHAR(10),

    -- 검색 설정
    top_k INTEGER,
    category_filter VARCHAR(10),

    -- 성능 메트릭
    total_time_ms INTEGER,
    embedding_time_ms INTEGER,
    search_time_ms INTEGER,

    -- 결과 통계
    num_results INTEGER,
    avg_score FLOAT,
    max_score FLOAT,

    -- 시스템 정보
    model_version VARCHAR(50),
    faiss_enabled BOOLEAN,
    device VARCHAR(10)
);

CREATE INDEX idx_timestamp ON search_logs(timestamp);
CREATE INDEX idx_query_category ON search_logs(query_category);
```

### 2. search_results 테이블 (검색 결과 상세)
```sql
CREATE TABLE search_results (
    id SERIAL PRIMARY KEY,
    query_id UUID NOT NULL REFERENCES search_logs(query_id),
    rank INTEGER NOT NULL,

    -- 결과 상품 정보
    product_id VARCHAR(50) NOT NULL,
    title TEXT,
    price INTEGER,
    category_id VARCHAR(10),

    -- 유사도
    similarity_score FLOAT,

    -- 사용자 상호작용 (향후)
    clicked BOOLEAN DEFAULT FALSE,
    click_timestamp TIMESTAMP,
    converted BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_query_id ON search_results(query_id);
CREATE INDEX idx_product_id ON search_results(product_id);
```

### 3. daily_metrics 테이블 (일별 집계)
```sql
CREATE TABLE daily_metrics (
    date DATE PRIMARY KEY,

    -- 검색 통계
    total_searches INTEGER,
    unique_users INTEGER,
    avg_results_per_search FLOAT,

    -- 성능 지표
    avg_latency_ms FLOAT,
    p50_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    p99_latency_ms FLOAT,

    -- 정확도 지표
    top_1_accuracy FLOAT,
    top_5_accuracy FLOAT,
    top_10_accuracy FLOAT,
    avg_score FLOAT,

    -- 카테고리별 (JSON)
    category_stats JSONB
);
```

---

## 🚀 FastAPI 서버 운영

### 1. 서버 실행
```bash
# 개발 환경
py api/search_api.py

# 프로덕션 환경 (Gunicorn + Uvicorn)
gunicorn api.search_api:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --timeout 300 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### 2. 서버 모니터링
- **헬스체크**: `GET /health`
- **메트릭 수집**: Prometheus + Grafana
- **로그 수집**: ELK Stack (Elasticsearch, Logstash, Kibana)

### 3. 백엔드 데이터 흐름

```
[Frontend]
    │
    ├─ 이미지 업로드
    │
    ▼
[FastAPI Server]
    │
    ├─ 1. 임베딩 생성 (FashionCLIP)
    ├─ 2. FAISS 검색
    ├─ 3. 결과 후처리
    ├─ 4. 메트릭 계산
    │
    ├─ → [Database] 검색 로그 저장
    │
    ▼
[Frontend]
    │
    └─ 결과 표시 + 사용자 상호작용 추적
         │
         └─ → [Database] 클릭/전환 로그 저장
```

---

## 📦 프론트엔드에 전달할 데이터

### 실시간 검색 응답
- ✅ 검색 결과 (상품 정보, 이미지, 가격, 점수)
- ✅ 메트릭 (검색 시간, 결과 수, 평균 점수)
- ⭐ 추천 사항 (이 카테고리의 인기 상품)
- ⭐ A/B 테스트 variant (다양한 랭킹 알고리즘)

### 대시보드용 집계 데이터
- 📊 일별/주별 성능 추이
- 🎯 Top-K Accuracy 변화
- 🔥 인기 카테고리/상품
- ⚡ 시스템 성능 (Latency, QPS)

### 분석용 Raw Data
- 🗄️ 검색 로그 (query_id, timestamp, category)
- 👆 클릭 로그 (product_id, rank, timestamp)
- 💰 전환 로그 (purchase events)

---

## 🎯 다음 단계

1. ✅ **평가 스크립트 실행** → Top-K Accuracy 측정
2. **로깅 시스템 추가** → 모든 검색 요청 기록
3. **데이터베이스 연동** → PostgreSQL/MySQL 연결
4. **대시보드 구현** → Streamlit/Grafana
5. **A/B 테스트 프레임워크** → 랭킹 알고리즘 비교
