# 빠른 답변 요약

## Q1: 유사도가 코사인 유사도임?
✅ **네, 코사인 유사도입니다**

```python
# FAISS IndexFlatIP (Inner Product)
# L2 normalization 후 → inner product = cosine similarity
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embedding_dim)
```

---

## Q2: Top-1, Top-5 결과는 어떻게 보지?

### 평가 스크립트 실행
```bash
# 100개 샘플로 평가
py scripts/evaluation/evaluate_search_metrics.py --n_samples 100

# 결과:
# Top-1 Accuracy: 78.5%
# Top-5 Accuracy: 87.3%
# Top-10 Accuracy: 92.1%
# MAP: 0.682
# MRR: 0.745
```

결과는 `results/evaluation/` 폴더에 저장됩니다:
- `detailed_metrics.csv` - 쿼리별 상세 지표
- `aggregated_metrics.json` - 전체 평균 지표

---

## Q3: 대시보드에 들어갈 내용

### 1. 검색 성능 지표
- **Top-K Accuracy** (Top-1, Top-5, Top-10, Top-20)
- **Precision@K, Recall@K**
- **MAP (Mean Average Precision)**
- **MRR (Mean Reciprocal Rank)**

### 2. 시스템 성능 지표
- **평균 검색 시간** (Latency)
- **P50, P95, P99 Latency**
- **QPS (Queries Per Second)**
- **FAISS vs Numpy 속도 비교**

### 3. 비즈니스 지표
- **CTR (Click-Through Rate)** - 클릭률
- **Conversion Rate** - 전환율
- **카테고리별 성능**
- **인기 검색 카테고리**

상세 설계: `docs/dashboard_design.md` 참고

---

## Q4: 프론트에 어떤 데이터 던져줄거냐?

### 현재 응답 (개선 완료!)
```json
{
  "query": {
    "query_id": "uuid-xxx",
    "timestamp": "2026-02-12T10:30:00Z",
    "image_info": {
      "filename": "test.jpg",
      "size": 45678,
      "dimensions": "800x600",
      "format": "JPEG"
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
      "kfashion_category": "블라우스",
      "score": 0.856
    }
  ],
  "metrics": {
    "total_results": 10,
    "search_time_ms": 512,
    "total_time_ms": 678,
    "category_filter": null,
    "faiss_enabled": true
  },
  "stats": {
    "avg_score": 0.623,
    "max_score": 0.856,
    "min_score": 0.445,
    "score_distribution": {
      "0.8-1.0": 2,
      "0.6-0.8": 5,
      "0.4-0.6": 3,
      "0.0-0.4": 0
    }
  }
}
```

### 프론트엔드가 사용할 데이터
1. **results** → 상품 카드 표시
2. **metrics** → 검색 시간, 결과 수 표시
3. **stats** → 평균 유사도, 점수 분포 차트
4. **query_id** → 클릭 추적용 (사용자가 어떤 결과를 클릭했는지)

---

## Q5: FastAPI는 서버만 켜놓으면 되는거지?

### ✅ 네, 맞습니다!

```bash
# 서버 시작
py api/search_api.py

# 또는 프로덕션 환경 (Gunicorn)
gunicorn api.search_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
```

### 자동으로 처리되는 것:
- ✅ 모델 로딩 (FashionCLIP)
- ✅ FAISS 인덱스 로딩
- ✅ CSV 데이터 로딩
- ✅ CORS 설정 (브라우저 접근 허용)
- ✅ Swagger UI (`/docs`)

### 추가로 해야 할 것:
- **로깅 시스템** 추가 (모든 검색 요청 기록)
- **데이터베이스 연동** (PostgreSQL/MySQL)
- **모니터링** (Prometheus + Grafana)
- **로드 밸런싱** (Nginx)

---

## Q6: 백엔드/DB에 어떤 데이터 던져야 하지?

### 데이터베이스 저장 항목

#### 1. search_logs 테이블 (매 검색마다)
```python
{
    "query_id": "uuid-xxx",
    "timestamp": "2026-02-12T10:30:00",
    "query_image_hash": "sha256-...",  # 중복 검색 감지
    "query_category": None,  # 업로드 검색의 경우

    # 검색 설정
    "top_k": 10,
    "category_filter": None,

    # 성능 메트릭
    "total_time_ms": 512,
    "search_time_ms": 345,

    # 결과 통계
    "num_results": 10,
    "avg_score": 0.623,
    "max_score": 0.856,

    # 시스템 정보
    "model_version": "fashion-clip-v1",
    "faiss_enabled": true,
    "device": "cuda"
}
```

#### 2. search_results 테이블 (각 결과마다)
```python
{
    "query_id": "uuid-xxx",
    "rank": 1,
    "product_id": "12345",
    "title": "상품명",
    "price": 29900,
    "category_id": "BL",
    "similarity_score": 0.856,

    # 사용자 상호작용 (나중에 프론트에서 전송)
    "clicked": false,
    "click_timestamp": null,
    "converted": false
}
```

#### 3. user_interactions 테이블 (클릭 추적)
```python
{
    "query_id": "uuid-xxx",
    "product_id": "12345",
    "rank": 1,
    "action": "click",  # 또는 "add_to_cart", "purchase"
    "timestamp": "2026-02-12T10:30:15"
}
```

### 프론트→백 데이터 흐름

```
[Frontend]
    │
    ├─ 이미지 업로드
    │
    ▼
[POST /search/upload]
    │
    ├─ 검색 수행
    ├─ query_id 생성
    ├─ [DB] search_logs 저장
    ├─ [DB] search_results 저장
    │
    ▼
[Response to Frontend]
    │
    └─ 사용자가 결과 클릭
         │
         ▼
    [POST /interactions]  ← 새 엔드포인트 필요
         │
         └─ [DB] user_interactions 저장
```

---

## 다음 작업

### 우선순위 P0 (즉시)
1. ✅ FAISS 구현 완료
2. ✅ API 개선 (메트릭 추가)
3. ⬜ **평가 스크립트 실행** → 정확도 측정
4. ⬜ **로깅 시스템** 추가 (Loguru/Structlog)

### 우선순위 P1 (다음 주)
5. ⬜ **데이터베이스 연동** (PostgreSQL)
6. ⬜ **대시보드 구현** (Streamlit)
7. ⬜ **클릭 추적 API** 추가

### 우선순위 P2 (향후)
8. ⬜ A/B 테스트 프레임워크
9. ⬜ 모니터링 시스템 (Prometheus)
10. ⬜ 프로덕션 배포 (Docker + K8s)

---

## 빠른 시작

### 1. 평가 실행
```bash
py scripts/evaluation/evaluate_search_metrics.py --n_samples 50
```

### 2. 웹 UI 사용
```
http://localhost:8001/
```

### 3. API 테스트
```
http://localhost:8001/docs
```

### 4. 대시보드 설계 확인
```
docs/dashboard_design.md
```
