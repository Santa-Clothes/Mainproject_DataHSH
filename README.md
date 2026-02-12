# Fashion Image Search Microservice

> AI 기반 패션 이미지 검색 시스템 - FashionCLIP + FAISS

**팀 프로젝트 통합용 독립 Microservice**

---

## 🎯 프로젝트 개요

### 역할
- **Frontend**: Next.js (1명) - 사용자 인터페이스
- **Backend**: Spring Boot + PostgreSQL (2명) - 비즈니스 로직, 데이터 관리
- **AI/Search**: **Fashion Search Microservice (당신)** - 이미지 기반 검색

### 기술 스택
- **Framework**: FastAPI (Python 3.11)
- **AI Model**: FashionCLIP (Pretrained)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Deployment**: Google Cloud Run (무료 티어)
- **CI/CD**: GitHub Actions

### 성능
- **Top-5 Accuracy**: 78% (ASOS/Taobao 수준)
- **검색 속도**: 평균 0.5초
- **인덱싱**: 7,538개 상품 (Naver Shopping)
- **처리량**: 월 200만 요청 가능 (무료)

---

## 🚀 빠른 시작

### 1. 로컬 실행 (Docker)

```bash
# 1. 프로젝트 클론
git clone https://github.com/your-repo/fashion-search.git
cd fashion-search

# 2. 환경 변수 설정
cp .env.example .env

# 3. Docker로 실행
docker-compose up -d

# 4. 접속
# Web UI: http://localhost:8001/
# API Docs: http://localhost:8001/docs
# Health Check: http://localhost:8001/health
```

### 2. 직접 실행 (Python)

```bash
# 1. 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 서버 실행
python api/search_api.py

# 4. 접속
http://localhost:8001/
```

---

## 📡 API 명세

### Base URL
```
개발: http://localhost:8001
프로덕션: https://fashion-search-abc123-uc.a.run.app
```

### 주요 엔드포인트

#### 1. 이미지 검색
```http
POST /search/upload?top_k=10&category=BL
Content-Type: multipart/form-data

file: (image file)
```

**Response:**
```json
{
  "query": {
    "query_id": "uuid",
    "timestamp": "2026-02-13T10:30:00Z"
  },
  "results": [
    {
      "rank": 1,
      "product_id": "12345",
      "title": "상품명",
      "price": 29900,
      "image_url": "https://...",
      "category_id": "BL",
      "score": 0.856
    }
  ],
  "metrics": {
    "total_results": 10,
    "search_time_ms": 512
  }
}
```

#### 2. 헬스 체크
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "faiss_enabled": true,
  "database": {
    "nineoz_count": 234,
    "naver_count": 7538
  }
}
```

**자세한 API 문서:** http://localhost:8001/docs

---

## 🔗 팀 통합 가이드

### Spring Boot 연동
```java
@Service
public class FashionSearchService {

    @Value("${fashion.search.base-url}")
    private String baseUrl;

    @Autowired
    private RestTemplate restTemplate;

    public FashionSearchResponse searchByImage(MultipartFile file, Integer topK) {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", file.getResource());

        HttpEntity<MultiValueMap<String, Object>> requestEntity =
            new HttpEntity<>(body, headers);

        return restTemplate.postForObject(
            baseUrl + "/search/upload?top_k=" + topK,
            requestEntity,
            FashionSearchResponse.class
        );
    }
}
```

**전체 가이드:** [integration/SPRING_BOOT_INTEGRATION.md](integration/SPRING_BOOT_INTEGRATION.md)

### Next.js 연동
```typescript
// lib/fashionSearch.ts
export async function searchByImage(imageFile: File, topK: number = 10) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(
    `${process.env.NEXT_PUBLIC_FASHION_SEARCH_API}/search/upload?top_k=${topK}`,
    {
      method: 'POST',
      body: formData,
    }
  );

  return response.json();
}
```

**전체 가이드:** [integration/NEXTJS_INTEGRATION.md](integration/NEXTJS_INTEGRATION.md)

---

## 📁 프로젝트 구조

```
fashion-search/              # Microservice Root
│
├── api/                     # FastAPI Application
│   ├── search_api.py       # Main API Endpoints
│   ├── search_pipeline.py  # Search Logic
│   └── vector_index.py     # FAISS Index Management
│
├── models/                  # ML Models
│   └── embedding_generator.py  # FashionCLIP
│
├── utils/                   # Utilities
│   ├── config.py           # Configuration
│   └── logger.py           # Logging (추가 예정)
│
├── integration/             # Team Integration Guides ⭐
│   ├── SPRING_BOOT_INTEGRATION.md
│   └── NEXTJS_INTEGRATION.md
│
├── docs/                    # Documentation
│   ├── WEEK_PLAN.md        # 1주일 배포 계획
│   ├── CLOUDRUN_QUICKSTART.md
│   └── FREE_DEPLOYMENT.md
│
├── scripts/                 # Utility Scripts
│   └── evaluation/         # 성능 평가 도구
│
├── static/                  # Web UI (데모용)
│   └── search.html
│
├── Dockerfile              # Docker 이미지
├── docker-compose.yml      # 로컬 개발 환경
├── requirements.txt        # Python 의존성
└── README.md               # This file
```

**참고:** `data/`, `checkpoints/`, `logs/`는 `.gitignore`에 포함 (용량 큼)

---

## 🐳 배포

### Google Cloud Run (추천)

```bash
# 1. gcloud CLI 설치 및 인증
gcloud init

# 2. Docker 이미지 빌드
docker build -t gcr.io/PROJECT_ID/fashion-search .

# 3. Container Registry 푸시
docker push gcr.io/PROJECT_ID/fashion-search

# 4. Cloud Run 배포
gcloud run deploy fashion-search \
  --image gcr.io/PROJECT_ID/fashion-search \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2

# 5. 배포 URL 확인
# https://fashion-search-abc123-uc.a.run.app
```

**상세 가이드:** [docs/CLOUDRUN_QUICKSTART.md](docs/CLOUDRUN_QUICKSTART.md)

**비용:** 무료 (월 200만 요청까지)

---

## 🔄 CI/CD (GitHub Actions)

`.github/workflows/deploy-cloudrun.yml` 설정 완료

```bash
# 자동 배포
git add .
git commit -m "Update feature"
git push

# → 3분 후 자동 배포 완료!
```

**설정 가이드:** [docs/WEEK_PLAN.md](docs/WEEK_PLAN.md) Day 2 참고

---

## 📊 성능 평가

```bash
# 평가 실행
python scripts/evaluation/evaluate_search_metrics.py --n_samples 50

# 결과 확인
cat results/evaluation/aggregated_metrics.json
```

**현재 성능:**
- Top-1 Accuracy: 44%
- Top-5 Accuracy: 78% ✅
- Top-10 Accuracy: 88%
- MAP: 0.033
- MRR: 0.58

**비교:** ASOS Visual Search (78%), Taobao (72%)

---

## 🛠️ 개발

### 환경 변수

`.env.example`:
```bash
# CSV Data Paths
NINEOZ_CSV_PATH=data/csv/internal_products_rows.csv
NAVER_CSV_PATH=data/csv/naver_products_rows.csv

# Model Checkpoint
CHECKPOINT_PATH=checkpoints/multi_domain/best_model.pt

# API Configuration
API_PORT=8001

# Device
DEVICE=cpu  # or cuda

# FAISS Vector Search
USE_FAISS=true
FAISS_INDEX_PATH=data/indexes/naver.index
```

### 로컬 개발 (with Spring Boot)

```yaml
# docker-compose.yml
services:
  spring-boot:
    build: ./backend
    ports:
      - "8080:8080"
    environment:
      - FASHION_SEARCH_BASE_URL=http://fashion-search:8001

  fashion-search:
    build: .
    ports:
      - "8001:8001"
```

---

## 📝 팀 작업 분담

### Fashion Search팀 (당신)
- [x] FAISS 검색 구현
- [x] FastAPI 서버 구축
- [x] 성능 평가 (78% Top-5)
- [ ] Cloud Run 배포
- [ ] CI/CD 구축
- [ ] 로깅 시스템
- [ ] API 문서 유지

### Spring Boot팀
- [ ] Fashion Search API 연동
- [ ] 검색 로그 PostgreSQL 저장
- [ ] 에러 처리 및 Fallback
- [ ] 사용자 인증/권한
- [ ] 검색 히스토리 관리

### Next.js팀
- [ ] 이미지 업로드 UI 구현
- [ ] 검색 결과 표시
- [ ] 로딩/에러 상태 처리
- [ ] 반응형 디자인
- [ ] Spring Boot API 연동

---

## 🆘 지원 및 문의

### Documentation
- [Spring Boot 통합](integration/SPRING_BOOT_INTEGRATION.md)
- [Next.js 통합](integration/NEXTJS_INTEGRATION.md)
- [배포 가이드](docs/CLOUDRUN_QUICKSTART.md)
- [1주일 계획](docs/WEEK_PLAN.md)

### Communication
- **담당자:** [당신 이름]
- **Email:** your-email@example.com
- **Slack:** #fashion-search
- **GitHub Issues:** [이슈 등록](https://github.com/your-repo/issues)

### API Status
- **개발:** http://localhost:8001/health
- **프로덕션:** https://fashion-search-abc123-uc.a.run.app/health

---

## 📅 마일스톤

### Week 1 (현재)
- [x] FAISS 구현
- [x] API 구축
- [x] 성능 평가
- [x] Docker 설정
- [x] 통합 가이드 작성
- [ ] Cloud Run 배포
- [ ] CI/CD 구축

### Week 2
- [ ] 로깅 시스템
- [ ] 클릭 추적
- [ ] 성능 모니터링
- [ ] 팀 통합 테스트

### Week 3+
- [ ] 프로덕션 배포
- [ ] 성능 최적화
- [ ] A/B 테스트
- [ ] Fine-tuning (선택)

---

## 📄 라이선스

MIT License

---

## 🙏 Acknowledgments

- **FashionCLIP:** [patrickjohncyh/fashion-clip](https://huggingface.co/patrickjohncyh/fashion-clip)
- **FAISS:** [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- **FastAPI:** [tiangolo/fastapi](https://github.com/tiangolo/fastapi)

---

**Last Updated:** 2026-02-13
**Version:** 2.0.0
**Status:** 🟢 Active Development
