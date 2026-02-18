# Fashion Search - 1주일 완성 플랜 (Cloud Run + CI/CD)

## 🎯 목표

**1주일 후:**
- ✅ Cloud Run 배포 완료 (24시간 운영)
- ✅ CI/CD 자동화 (git push → 3분 자동 배포)
- ✅ 로깅 시스템 (모든 검색 기록)
- ✅ 클릭 추적 (사용자 행동 분석)
- ✅ 발표 준비 완료

**비용: $0** (무료 티어 범위 내)

---

## 📅 Day 1: Cloud Run 배포 (2시간)

### 오전 (1시간)

#### Task 1.1: Google Cloud 설정 (20분)
```bash
□ Google Cloud Console 접속
□ 프로젝트 생성: fashion-search-demo
□ 결제 정보 등록 (무료 크레딧 $300)
```

#### Task 1.2: gcloud CLI 설치 (20분)
```bash
# Windows
□ gcloud SDK 다운로드
□ 설치 완료
□ gcloud init 실행
□ 계정 인증

# 확인
gcloud config list
```

#### Task 1.3: API 활성화 (5분)
```bash
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 오후 (1시간)

#### Task 1.4: Docker 이미지 빌드 및 푸시 (30분)
```bash
# 1. 프로젝트 루트
cd c:\Mainproject_DataHSH

# 2. 빌드
docker build -t gcr.io/fashion-search-demo/fashion-search:latest .

# 3. 인증
gcloud auth configure-docker

# 4. 푸시 (10분 소요)
docker push gcr.io/fashion-search-demo/fashion-search:latest
```

#### Task 1.5: Cloud Run 배포 (10분)
```bash
gcloud run deploy fashion-search \
  --image gcr.io/fashion-search-demo/fashion-search:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8001 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 3
```

#### Task 1.6: 동작 확인 (10분)
```bash
# URL 확인
# 예: https://fashion-search-abc123-uc.a.run.app

□ 웹 브라우저 접속
□ 이미지 업로드 테스트
□ 검색 결과 확인
□ API 문서 확인 (/docs)
□ 헬스 체크 (/health)
```

### ✅ Day 1 완료 체크리스트
- [ ] Cloud Run 배포 완료
- [ ] 공개 URL 접속 가능
- [ ] 검색 기능 정상 동작
- [ ] URL 저장: _______________________

---

## 📅 Day 2: CI/CD 자동화 (2시간)

### 오전 (1시간)

#### Task 2.1: Service Account 생성 (20분)
```bash
# 1. Service Account 생성
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

# 2. 권한 부여
gcloud projects add-iam-policy-binding fashion-search-demo \
  --member="serviceAccount:github-actions@fashion-search-demo.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding fashion-search-demo \
  --member="serviceAccount:github-actions@fashion-search-demo.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding fashion-search-demo \
  --member="serviceAccount:github-actions@fashion-search-demo.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# 3. 키 생성
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@fashion-search-demo.iam.gserviceaccount.com

# 4. key.json 내용 복사
cat key.json
```

#### Task 2.2: GitHub Secrets 설정 (10분)
```bash
□ GitHub 저장소 → Settings → Secrets
□ New repository secret 클릭

Secret 1:
Name: GCP_PROJECT_ID
Value: fashion-search-demo

Secret 2:
Name: GCP_SA_KEY
Value: <key.json 내용 붙여넣기>
```

### 오후 (1시간)

#### Task 2.3: GitHub Actions Workflow 작성 (30분)

파일 생성: `.github/workflows/deploy-cloudrun.yml`

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  SERVICE_NAME: fashion-search
  REGION: us-central1

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Configure Docker
        run: gcloud auth configure-docker

      - name: Build Docker image
        run: |
          docker build -t gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:${{ github.sha }} .
          docker tag gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:${{ github.sha }} \
            gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:latest

      - name: Push Docker image
        run: |
          docker push gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:${{ github.sha }}
          docker push gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:latest

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy ${{ env.SERVICE_NAME }} \
            --image gcr.io/${{ env.PROJECT_ID }}/${{ env.SERVICE_NAME }}:${{ github.sha }} \
            --platform managed \
            --region ${{ env.REGION }} \
            --allow-unauthenticated \
            --memory 4Gi \
            --cpu 2 \
            --max-instances 3

      - name: Health check
        run: |
          SERVICE_URL=$(gcloud run services describe ${{ env.SERVICE_NAME }} \
            --region ${{ env.REGION }} \
            --format 'value(status.url)')
          curl -f $SERVICE_URL/health || exit 1
```

#### Task 2.4: 자동 배포 테스트 (20분)
```bash
# 1. Workflow 커밋
git add .github/workflows/deploy-cloudrun.yml
git commit -m "Add CI/CD workflow"
git push

# 2. GitHub Actions 확인
□ https://github.com/your-repo/actions
□ Workflow 실행 확인
□ 3-5분 대기
□ 배포 성공 확인

# 3. 테스트 변경
# README.md 수정
git commit -m "Test auto deploy"
git push

# 4. 자동 배포 확인
□ GitHub Actions 다시 실행됨
□ 3분 후 배포 완료
```

### ✅ Day 2 완료 체크리스트
- [ ] GitHub Actions 설정 완료
- [ ] 자동 배포 테스트 성공
- [ ] git push → 3분 자동 배포 확인

---

## 📅 Day 3: 로깅 시스템 (3-4시간)

### 오전 (2시간)

#### Task 3.1: 로거 설정 파일 작성 (30분)

파일 생성: `utils/logger.py`

```python
"""
Structured Logging for Fashion Search
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import structlog
from structlog.stdlib import LoggerFactory


def setup_logger(log_dir: str = "logs", log_level: str = "INFO"):
    """
    구조화된 로거 설정
    """
    # 로그 디렉토리 생성
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 로그 파일 경로
    log_file = Path(log_dir) / f"search_{datetime.now().strftime('%Y%m%d')}.log"

    # structlog 설정
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 파일 핸들러
    file_handler = open(log_file, 'a', encoding='utf-8')

    return structlog.get_logger(), file_handler


def log_search_request(
    logger,
    query_id: str,
    query_type: str,
    category_filter: str = None,
    top_k: int = 10,
    **kwargs
):
    """검색 요청 로그"""
    logger.info(
        "search_request",
        query_id=query_id,
        query_type=query_type,
        category_filter=category_filter,
        top_k=top_k,
        **kwargs
    )


def log_search_response(
    logger,
    query_id: str,
    num_results: int,
    search_time_ms: int,
    avg_score: float,
    **kwargs
):
    """검색 응답 로그"""
    logger.info(
        "search_response",
        query_id=query_id,
        num_results=num_results,
        search_time_ms=search_time_ms,
        avg_score=avg_score,
        **kwargs
    )
```

#### Task 3.2: API에 로깅 추가 (1시간)

`api/search_api.py` 수정:

```python
# 상단에 추가
from utils.logger import setup_logger, log_search_request, log_search_response

# 앱 시작 시 로거 초기화
logger, log_file_handler = setup_logger()

# /search/upload 엔드포인트 수정
@app.post("/search/upload")
async def search_upload(
    file: UploadFile = File(...),
    category: Optional[str] = Query(None),
    top_k: int = Query(10, ge=1, le=100),
):
    query_id = str(uuid.uuid4())
    total_start = time.time()

    # 요청 로그
    log_search_request(
        logger,
        query_id=query_id,
        query_type="upload",
        category_filter=category,
        top_k=top_k,
        filename=file.filename,
        content_type=file.content_type
    )

    try:
        # ... 검색 로직 ...

        # 응답 로그
        log_search_response(
            logger,
            query_id=query_id,
            num_results=len(results),
            search_time_ms=int(search_time * 1000),
            avg_score=avg_score
        )

        return response

    except Exception as e:
        logger.error(
            "search_error",
            query_id=query_id,
            error=str(e)
        )
        raise
```

### 오후 (1-2시간)

#### Task 3.3: 로컬 테스트 (30분)
```bash
# 1. 로컬 실행
docker-compose restart

# 2. 검색 테스트
□ 웹 UI에서 이미지 업로드
□ 여러 번 검색 수행

# 3. 로그 확인
tail -f logs/search_20260213.log

# 예상 로그:
# {"event": "search_request", "query_id": "abc-123", "timestamp": "2026-02-13T10:30:00"}
# {"event": "search_response", "query_id": "abc-123", "num_results": 10, "search_time_ms": 512}
```

#### Task 3.4: Git Push (자동 배포) (10분)
```bash
git add utils/logger.py api/search_api.py requirements.txt
git commit -m "Add structured logging system"
git push

# GitHub Actions가 자동으로 배포
# 3분 후 Cloud Run에 반영
```

#### Task 3.5: Cloud Run 로그 확인 (20분)
```bash
# Cloud Run 로그 스트리밍
gcloud run services logs read fashion-search --follow

# 또는 Console에서
# https://console.cloud.google.com/run
# fashion-search 클릭 → LOGS 탭
```

### ✅ Day 3 완료 체크리스트
- [ ] 로깅 시스템 구현 완료
- [ ] 로컬 테스트 성공
- [ ] Cloud Run 배포 완료
- [ ] 로그 정상 기록 확인

---

## 📅 Day 4: 클릭 추적 (3-4시간)

### 오전 (2시간)

#### Task 4.1: 클릭 추적 API 구현 (1시간)

`api/search_api.py`에 추가:

```python
# Pydantic 모델
class InteractionRequest(BaseModel):
    query_id: str
    product_id: str
    rank: int
    action: str  # "click", "add_to_cart", "purchase"
    timestamp: Optional[str] = None


# 엔드포인트
@app.post("/interactions")
async def log_interaction(interaction: InteractionRequest):
    """사용자 상호작용 기록"""

    # 타임스탬프 추가
    if not interaction.timestamp:
        interaction.timestamp = datetime.utcnow().isoformat() + "Z"

    # 로그 기록
    logger.info(
        "user_interaction",
        query_id=interaction.query_id,
        product_id=interaction.product_id,
        rank=interaction.rank,
        action=interaction.action,
        timestamp=interaction.timestamp
    )

    # JSON 파일로도 저장 (분석용)
    interaction_file = Path("results/interactions") / f"interactions_{datetime.now().strftime('%Y%m%d')}.jsonl"
    interaction_file.parent.mkdir(parents=True, exist_ok=True)

    with open(interaction_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(interaction.dict()) + '\n')

    return {"status": "ok", "message": "Interaction logged"}
```

#### Task 4.2: 웹 UI에 클릭 추적 추가 (1시간)

`static/search.html` 수정:

```javascript
// 결과 표시 함수 수정
function displayResults(data) {
    const results = data.results || [];
    const queryId = data.query?.query_id;

    // ... 기존 코드 ...

    // 결과 카드에 클릭 이벤트 추가
    resultsGrid.innerHTML = results.map((item, idx) => `
        <div class="result-card" onclick="trackClick('${queryId}', '${item.product_id}', ${idx + 1})">
            <img src="${item.image_url}" alt="${item.title}" class="result-image">
            <div class="result-info">
                <span class="result-rank">#${idx + 1}</span>
                <span class="result-score">유사도: ${item.score.toFixed(3)}</span>
                <div class="result-title">${item.title}</div>
                <div class="result-price">₩${item.price.toLocaleString()}</div>
            </div>
        </div>
    `).join('');
}

// 클릭 추적 함수
async function trackClick(queryId, productId, rank) {
    try {
        await fetch('http://localhost:8001/interactions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query_id: queryId,
                product_id: productId,
                rank: rank,
                action: 'click'
            })
        });

        console.log(`Tracked click: ${productId} at rank ${rank}`);
    } catch (err) {
        console.error('Failed to track click:', err);
    }
}
```

### 오후 (1-2시간)

#### Task 4.3: 로컬 테스트 (30분)
```bash
# 1. 로컬 실행
docker-compose restart

# 2. 웹 UI 테스트
□ 이미지 업로드
□ 검색 결과 클릭
□ 여러 결과 클릭 테스트

# 3. 로그 확인
tail -f logs/search_20260214.log | grep "user_interaction"

# 4. Interaction 파일 확인
cat results/interactions/interactions_20260214.jsonl
```

#### Task 4.4: Git Push (자동 배포) (10분)
```bash
git add api/search_api.py static/search.html
git commit -m "Add click tracking system"
git push

# 자동 배포 확인
```

### ✅ Day 4 완료 체크리스트
- [ ] 클릭 추적 API 구현 완료
- [ ] 웹 UI 클릭 추적 추가
- [ ] 로컬 테스트 성공
- [ ] Cloud Run 배포 완료

---

## 📅 Day 5: 분석 도구 및 문서화 (3시간)

### 오전 (2시간)

#### Task 5.1: 로그 분석 스크립트 작성 (1시간)

파일 생성: `scripts/analysis/analyze_logs.py`

```python
"""
검색 로그 분석 도구
"""
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd


def parse_logs(log_file: str):
    """로그 파일 파싱"""
    events = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line)
                events.append(event)
            except:
                continue

    return events


def analyze_search_stats(events):
    """검색 통계 분석"""

    # 요청/응답 분리
    requests = [e for e in events if e.get('event') == 'search_request']
    responses = [e for e in events if e.get('event') == 'search_response']

    print(f"\n{'='*60}")
    print(f"검색 통계")
    print(f"{'='*60}")
    print(f"총 검색 수: {len(requests)}")
    print(f"성공: {len(responses)}")
    print(f"실패: {len(requests) - len(responses)}")

    if responses:
        # 응답 시간 분석
        times = [r['search_time_ms'] for r in responses]
        print(f"\n응답 시간:")
        print(f"  평균: {sum(times) / len(times):.0f}ms")
        print(f"  최소: {min(times)}ms")
        print(f"  최대: {max(times)}ms")

        # 유사도 분석
        scores = [r['avg_score'] for r in responses]
        print(f"\n평균 유사도:")
        print(f"  평균: {sum(scores) / len(scores):.3f}")
        print(f"  최소: {min(scores):.3f}")
        print(f"  최대: {max(scores):.3f}")

    print(f"{'='*60}\n")


def analyze_click_stats(events):
    """클릭 통계 분석"""

    interactions = [e for e in events if e.get('event') == 'user_interaction']

    if not interactions:
        print("클릭 데이터 없음")
        return

    print(f"\n{'='*60}")
    print(f"클릭 통계")
    print(f"{'='*60}")
    print(f"총 클릭 수: {len(interactions)}")

    # 순위별 클릭 분포
    rank_clicks = defaultdict(int)
    for interaction in interactions:
        rank_clicks[interaction['rank']] += 1

    print(f"\n순위별 클릭:")
    for rank in sorted(rank_clicks.keys())[:10]:
        print(f"  Rank {rank}: {rank_clicks[rank]}회")

    # CTR 계산 (간단 버전)
    # TODO: query_id로 매칭하여 정확한 CTR 계산

    print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", help="Log file to analyze")
    args = parser.parse_args()

    events = parse_logs(args.log_file)

    analyze_search_stats(events)
    analyze_click_stats(events)


if __name__ == "__main__":
    main()
```

#### Task 5.2: README 업데이트 (1시간)

`README.md` 작성:

```markdown
# Fashion Image Search System

패션 이미지 검색 시스템 - FashionCLIP + FAISS

## 🚀 Live Demo

**배포 URL:** https://fashion-search-abc123-uc.a.run.app

## 📊 성능

- **Top-5 Accuracy**: 78%
- **검색 속도**: 평균 0.5초
- **인덱싱**: 7,538개 상품
- **기술 스택**: FastAPI + FashionCLIP + FAISS + Cloud Run

## 🎯 주요 기능

1. **이미지 기반 검색**
   - 이미지 업로드 → 유사 상품 검색
   - Nine Oz (평면) → Naver Shopping (모델 착용)

2. **빠른 검색**
   - FAISS 벡터 인덱스
   - 2000배 속도 개선

3. **자동 배포**
   - GitHub Actions CI/CD
   - git push → 3분 자동 배포

## 🏗️ 아키텍처

```
[Frontend]
    │
    ├─ 이미지 업로드
    │
    ▼
[FastAPI]
    │
    ├─ FashionCLIP (임베딩)
    ├─ FAISS (벡터 검색)
    ├─ Re-ranking
    │
    ▼
[Cloud Run]
    │
    └─ 자동 스케일링
```

## 🛠️ 로컬 실행

### Docker 사용
```bash
docker-compose up -d
# http://localhost:8001/
```

### 직접 실행
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python api/search_api.py
```

## 📊 로그 분석

```bash
python scripts/analysis/analyze_logs.py logs/search_20260213.log
```

## 🔧 개발

### CI/CD
- GitHub Actions 자동 배포
- Cloud Run 무료 티어
- 3분 자동 배포

### 모니터링
- 구조화된 로깅
- 클릭 추적
- 성능 메트릭

## 📝 문서

- [배포 가이드](docs/CLOUDRUN_QUICKSTART.md)
- [무료 배포 옵션](docs/FREE_DEPLOYMENT.md)
- [1주일 플랜](docs/WEEK_PLAN.md)

## 📧 Contact

- GitHub: [@your-username](https://github.com/your-username)
- Email: your-email@example.com
```

### 오후 (1시간)

#### Task 5.3: 발표 자료 준비 (1시간)

파일 생성: `docs/PRESENTATION.md`

```markdown
# Fashion Search 발표 자료

## 슬라이드 1: 제목

**Fashion Image Search System**
- FashionCLIP + FAISS 기반
- 실시간 유사 상품 검색

## 슬라이드 2: 문제 정의

**문제:**
- Nine Oz (평면 제품 이미지)
- Naver Shopping (모델 착용 이미지)
- Domain Gap 문제

**목표:**
- 평면 이미지 → 착용 이미지 검색
- 1초 이내 응답

## 슬라이드 3: 기술 스택

- **모델**: FashionCLIP (Pretrained)
- **검색**: FAISS (2000x faster)
- **백엔드**: FastAPI
- **배포**: Google Cloud Run
- **CI/CD**: GitHub Actions

## 슬라이드 4: 시스템 아키텍처

[아키텍처 다이어그램]

## 슬라이드 5: 성능

- **Top-5 Accuracy**: 78%
- **검색 속도**: 0.5초
- **인덱싱**: 7,538개 상품

## 슬라이드 6: 라이브 데모

**배포 URL:** https://fashion-search-abc123-uc.a.run.app

[실제 검색 시연]

## 슬라이드 7: 기술적 도전

1. Domain Gap 해결
2. 검색 속도 최적화 (FAISS)
3. 무료 배포 (Cloud Run)

## 슬라이드 8: 향후 계획

1. Multi-Domain Training (Fine-tuning)
2. Re-ranking 개선
3. 실사용 데이터 수집

## 슬라이드 9: Q&A
```

### ✅ Day 5 완료 체크리스트
- [ ] 로그 분석 도구 작성
- [ ] README 업데이트
- [ ] 발표 자료 준비

---

## 📅 Day 6-7: 테스트 및 개선 (4시간)

### Day 6 오전 (2시간)

#### Task 6.1: 종합 테스트 (1시간)
```bash
□ 다양한 이미지로 검색 테스트
□ 카테고리 필터 테스트
□ Top-K 값 변경 테스트
□ 모바일 접속 테스트
□ 여러 브라우저 테스트
```

#### Task 6.2: 성능 테스트 (1시간)
```bash
# 동시 요청 테스트
# 예: Apache Bench
ab -n 100 -c 10 https://fashion-search-abc123-uc.a.run.app/health

□ 응답 시간 확인
□ 에러율 확인
□ Cloud Run 로그 확인
```

### Day 6 오후 (2시간)

#### Task 6.3: 에러 처리 개선 (1시간)

`api/search_api.py` 수정:

```python
# 에러 핸들러 추가
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc)
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "검색 중 오류가 발생했습니다. 다시 시도해주세요."
        }
    )
```

#### Task 6.4: UI 개선 (1시간)

`static/search.html` 수정:

```javascript
// 로딩 상태 개선
// 에러 메시지 개선
// 검색 히스토리 (localStorage)
```

### Day 7: 최종 점검 및 문서화

#### Task 7.1: 발표 리허설 (1시간)
```bash
□ 발표 자료 리뷰
□ 데모 시나리오 연습
□ 예상 질문 준비
□ 타이밍 체크
```

#### Task 7.2: 최종 배포 및 테스트 (1시간)
```bash
# 최종 변경사항 반영
git add .
git commit -m "Final improvements for presentation"
git push

# 배포 확인
□ CI/CD 성공
□ 모든 기능 동작
□ 로그 정상
```

#### Task 7.3: 발표 자료 마무리 (1시간)
```bash
□ 스크린샷 추가
□ 성능 지표 최신화
□ URL 확인
□ 백업 플랜 준비
```

### ✅ Week 1 완료 체크리스트
- [ ] Cloud Run 배포 완료
- [ ] CI/CD 자동화 완료
- [ ] 로깅 시스템 동작
- [ ] 클릭 추적 동작
- [ ] 분석 도구 완성
- [ ] 문서화 완료
- [ ] 발표 준비 완료
- [ ] 최종 테스트 통과

---

## 📊 최종 결과물

### 시스템
- ✅ **배포 URL**: https://fashion-search-abc123-uc.a.run.app
- ✅ **CI/CD**: GitHub Actions (자동 배포)
- ✅ **모니터링**: 구조화된 로깅
- ✅ **분석**: 클릭 추적 + 로그 분석

### 성능
- ✅ **Top-5**: 78%
- ✅ **응답 시간**: 0.5초
- ✅ **가용성**: 24/7

### 비용
- ✅ **배포**: $0
- ✅ **운영**: $0/월 (무료 티어)

---

## 🎯 발표 시나리오

### 1. 문제 소개 (2분)
- Domain Gap 문제
- 기존 솔루션의 한계

### 2. 기술 설명 (3분)
- FashionCLIP
- FAISS 검색
- Cloud Run 배포

### 3. 라이브 데모 (3분)
- URL 공유
- 실시간 검색 시연
- 결과 분석

### 4. 성능 및 결과 (2분)
- Top-K Accuracy
- 검색 속도
- 무료 배포

### 5. Q&A (5분)

---

## 🔧 트러블슈팅

### 문제 1: CI/CD 실패
```bash
# GitHub Actions 로그 확인
# Secrets 다시 확인
# Service Account 권한 확인
```

### 문제 2: Cold Start 느림
```bash
# 발표 5분 전 미리 접속
curl https://fashion-search-abc123-uc.a.run.app/health
```

### 문제 3: 이미지 로드 실패
```bash
# CORS 설정 확인
# 이미지 URL 유효성 확인
```

---

## 📝 다음 작업 (발표 후)

1. 실사용 데이터 수집 (1개월)
2. 카테고리별 성능 분석
3. Quick Wins 적용 (Re-ranking)
4. Multi-Domain Training 고려

---

**완료! 1주일 후 완벽한 발표 준비 완료!** 🎉
