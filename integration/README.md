# Fashion Search API - 프론트엔드 통합 가이드

**상태:** 연동 검증 완료 ✅
**API:** http://localhost:8001
**데이터:** Nine Oz 4,621개 + Naver 7,538개 = 총 12,159개 제품

---

## 전체 흐름

```
[Next.js 프론트]
      │  이미지 파일 (multipart/form-data)
      ▼
[FastAPI 백엔드 :8001]
      │  FashionCLIP → 이미지 임베딩(768차원) 생성
      │  FAISS 벡터 인덱스로 유사 상품 검색
      ▼
[Supabase DB]
      │  naver_products 테이블 (7,538개, 임베딩 포함)
      ▼
[FastAPI 백엔드]
      │  유사도 순 정렬, JSON 응답
      ▼
[Next.js 프론트]
      │  상품 목록 + 스타일 분포 표시
```

---

## Next.js 통합 (필수)

### 1단계: API 클라이언트 복사

```
integration/examples/fashionSearch.ts  →  (Next.js 프로젝트)/lib/fashionSearch.ts
```

### 2단계: 환경 변수

```env
# .env.local
NEXT_PUBLIC_FASHION_SEARCH_API=http://localhost:8001
```

### 3단계: 사용

```tsx
import { searchByImage, getStyleDistribution } from '@/lib/fashionSearch';

const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
  const file = e.target.files?.[0];
  if (!file) return;

  const data = await searchByImage(file, 20); // 20개 요청할수록 스타일 분포 정확

  // 상품 목록
  console.log(data.results);

  // 스타일 분포 (예: [{style: "스트리트", percent: 55}, {style: "캐주얼", percent: 30}, ...])
  const styles = getStyleDistribution(data.results);
  console.log(styles);
};
```

---

## API 엔드포인트

### POST /search/upload — 이미지 검색 (핵심)

```bash
curl -X POST "http://localhost:8001/search/upload?top_k=10" \
  -F "file=@./image.jpg"
```

**Query Parameters:**

| 파라미터 | 필수 | 기본값 | 설명 |
|---------|------|--------|------|
| `top_k` | 선택 | 10 | 반환할 결과 수 (최대 100) |
| `category_filter` | 선택 | 없음 | K-Fashion 카테고리 필터 |

**Response:**
```json
{
  "query": {
    "query_id": "uuid",
    "timestamp": "2026-02-19T10:30:00Z",
    "image_info": { "filename": "image.jpg", "size": 245678, "dimensions": "800x600", "format": "JPEG" }
  },
  "results": [
    {
      "rank": 1,
      "product_id": "90233826193",
      "title": "오버핏 후드 티셔츠",
      "price": 29900,
      "image_url": "https://shopping-phinf.pstatic.net/...",
      "category_id": "50000803",
      "kfashion_category": "스트리트",
      "score": 0.891
    }
  ],
  "metrics": {
    "total_results": 10,
    "search_time_ms": 480,
    "total_time_ms": 510,
    "category_filter": null,
    "faiss_enabled": true
  },
  "stats": {
    "avg_score": 0.743,
    "max_score": 0.891,
    "min_score": 0.612,
    "score_distribution": { "0.8-1.0": 3, "0.6-0.8": 6, "0.4-0.6": 1, "0.0-0.4": 0 }
  }
}
```

### GET /health — 서버 상태 확인

```bash
curl http://localhost:8001/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "nineoz_count": 4621,
  "naver_count": 7538
}
```

### GET /docs — Swagger UI

브라우저에서 http://localhost:8001/docs 접속

---

## 스타일 분포 (프론트 집계)

백엔드 수정 없이 `getStyleDistribution()` 함수로 바로 계산 가능.
FashionCLIP 임베딩 자체에 스타일 정보가 반영되어 있어 결과의 `kfashion_category` + `score`를 집계하면 됨.

```ts
// fashionSearch.ts에 포함된 함수
const styles = getStyleDistribution(data.results);
// [{ style: "스트리트", percent: 55 }, { style: "캐주얼", percent: 30 }, { style: "모던", percent: 15 }]
```

> top_k를 20~30으로 설정할수록 분포가 더 정확해짐.

---

## 제공 파일

```
integration/
├── README.md                          # 이 파일 (통합 가이드)
└── examples/
    ├── fashionSearch.ts               # ⭐ Next.js API 클라이언트 (필수)
    ├── FashionSearchComponent.tsx     # UI 참고 예시 (Tailwind CSS)
    ├── RestTemplateConfig.java        # Spring Boot: HTTP 클라이언트 설정
    ├── FashionSearchService.java      # Spring Boot: API 호출 서비스
    └── FashionSearchController.java   # Spring Boot: REST 컨트롤러
```

**Next.js 팀은 `fashionSearch.ts` 하나만 필요.**
Spring Boot 파일은 백엔드 프록시가 필요한 경우 선택적으로 사용.

---

## 트러블슈팅

### API 연결 안됨
```bash
curl http://localhost:8001/health
# 응답 없으면 API 재시작:
# cd c:\FinalProject_v2 && python -m uvicorn api.search_api:app --port 8001
```

### CORS 에러
API는 모든 오리진(`*`) 허용으로 설정됨. CORS 에러 발생 시 시크릿 모드에서 재테스트.

### 검색 결과가 없거나 score가 낮음
- 이미지가 패션 아이템(의류)인지 확인
- 파일 형식: JPG, PNG 권장
- 서버 상태 확인: `/health` 에서 `model_loaded: true` 확인
