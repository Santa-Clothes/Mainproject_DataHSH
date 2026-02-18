# Fashion Search Microservice - TODO

> 팀 프로젝트 통합용 TODO 리스트 (Cloud Run + CI/CD)

**Last Updated:** 2026-02-13
**Team:** Frontend (Next.js) + Backend (Spring Boot) + AI/Search (당신)

---

## ✅ 완료된 작업

### Core Features
- [x] FashionCLIP 임베딩 생성기 구현
- [x] FAISS 벡터 인덱스 구현 (2000배 속도 개선)
- [x] FastAPI REST API 구축
- [x] 이미지 업로드 검색 엔드포인트 (POST /search/upload)
- [x] Nine Oz 인덱스 검색 엔드포인트 (GET /search)
- [x] 웹 UI (Drag & Drop)

### Evaluation
- [x] Top-K Accuracy 평가 스크립트
- [x] 50개 샘플 평가 완료
- [x] 성능: Top-5 78% (ASOS/Taobao 수준)

### Documentation & Integration
- [x] Docker 설정 (Dockerfile, docker-compose.yml)
- [x] Spring Boot 통합 가이드 작성
- [x] Next.js 통합 가이드 작성
- [x] 배포 가이드 (Cloud Run)
- [x] 1주일 완성 플랜
- [x] 프로젝트 정리 (archive 폴더 분리)
- [x] README.md 팀 프로젝트용으로 재작성
- [x] .gitignore 업데이트

---

## 📅 Week 1: Cloud Run 배포 + CI/CD (7일)

### Day 1: Cloud Run 첫 배포 (2시간)
- [ ] Google Cloud 계정 생성 및 프로젝트 설정 (20분)
  - 프로젝트 이름: fashion-search-demo
  - 결제 정보 등록 (무료 크레딧 $300)

- [ ] gcloud CLI 설치 및 인증 (20분)
  ```bash
  # Windows: https://cloud.google.com/sdk/docs/install
  gcloud init
  ```

- [ ] API 활성화 (5분)
  ```bash
  gcloud services enable run.googleapis.com
  gcloud services enable containerregistry.googleapis.com
  ```

- [ ] Docker 이미지 빌드 및 푸시 (30분)
  ```bash
  docker build -t gcr.io/PROJECT_ID/fashion-search .
  gcloud auth configure-docker
  docker push gcr.io/PROJECT_ID/fashion-search
  ```

- [ ] Cloud Run 배포 (10분)
  ```bash
  gcloud run deploy fashion-search \
    --image gcr.io/PROJECT_ID/fashion-search \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2
  ```

- [ ] 동작 확인 (10분)
  - [ ] 웹 브라우저 접속
  - [ ] 이미지 업로드 테스트
  - [ ] API 문서 확인 (/docs)
  - [ ] 배포 URL 저장: ____________________

**가이드:** [docs/CLOUDRUN_QUICKSTART.md](docs/CLOUDRUN_QUICKSTART.md)

---

### Day 2: CI/CD 자동화 (2시간)

- [ ] Service Account 생성 (20분)
  ```bash
  gcloud iam service-accounts create github-actions
  # 권한 부여 스크립트 실행
  # key.json 생성
  ```

- [ ] GitHub Secrets 설정 (10분)
  - [ ] GCP_PROJECT_ID
  - [ ] GCP_SA_KEY (key.json 내용)

- [ ] GitHub Actions Workflow 작성 (30분)
  - [ ] 파일 생성: `.github/workflows/deploy-cloudrun.yml`
  - [ ] 설정 확인

- [ ] 자동 배포 테스트 (20분)
  ```bash
  git add .github/workflows/deploy-cloudrun.yml
  git commit -m "Add CI/CD workflow"
  git push
  # GitHub Actions 확인
  ```

- [ ] 테스트 커밋으로 재확인 (10분)
  - [ ] README 수정
  - [ ] git push
  - [ ] 3분 후 자동 배포 확인

**가이드:** [docs/WEEK_PLAN.md](docs/WEEK_PLAN.md) Day 2

---

### Day 3: 로깅 시스템 (3-4시간)

- [ ] 로거 설정 파일 작성 (30분)
  - [ ] `utils/logger.py` 생성
  - [ ] Structlog 설정
  - [ ] 로그 파일 rotation

- [ ] API에 로깅 추가 (1시간)
  - [ ] `api/search_api.py` 수정
  - [ ] 검색 요청 로그
  - [ ] 검색 응답 로그
  - [ ] 에러 로그

- [ ] 로컬 테스트 (30분)
  - [ ] Docker 재시작
  - [ ] 검색 수행
  - [ ] 로그 확인: `tail -f logs/search_*.log`

- [ ] Git Push (자동 배포) (10분)
  ```bash
  git add utils/logger.py api/search_api.py
  git commit -m "Add structured logging system"
  git push
  ```

- [ ] Cloud Run 로그 확인 (20분)
  ```bash
  gcloud run services logs read fashion-search --follow
  ```

**가이드:** [docs/WEEK_PLAN.md](docs/WEEK_PLAN.md) Day 3

---

### Day 4: 클릭 추적 (3-4시간)

- [ ] 클릭 추적 API 구현 (1시간)
  - [ ] `api/search_api.py`에 POST /interactions 추가
  - [ ] Pydantic 모델 정의
  - [ ] JSON 파일 저장

- [ ] 웹 UI에 클릭 추적 추가 (1시간)
  - [ ] `static/search.html` 수정
  - [ ] 클릭 이벤트 JavaScript 추가
  - [ ] fetch로 /interactions 호출

- [ ] 로컬 테스트 (30분)
  - [ ] 이미지 업로드
  - [ ] 결과 클릭
  - [ ] 로그 확인

- [ ] Git Push (자동 배포) (10분)
  ```bash
  git add api/search_api.py static/search.html
  git commit -m "Add click tracking system"
  git push
  ```

**가이드:** [docs/WEEK_PLAN.md](docs/WEEK_PLAN.md) Day 4

---

### Day 5: 분석 도구 및 문서화 (3시간)

- [ ] 로그 분석 스크립트 작성 (1시간)
  - [ ] `scripts/analysis/analyze_logs.py`
  - [ ] 검색 통계 분석
  - [ ] 클릭 통계 분석

- [ ] README 최종 업데이트 (1시간)
  - [ ] 배포 URL 추가
  - [ ] 팀원 정보 업데이트
  - [ ] 사용법 정리

- [ ] 발표 자료 준비 (1시간)
  - [ ] `docs/PRESENTATION.md` 작성
  - [ ] 스크린샷 캡처
  - [ ] 시연 시나리오

---

### Day 6-7: 테스트 및 개선 (4시간)

#### Day 6 오전 (2시간)
- [ ] 종합 테스트 (1시간)
  - [ ] 다양한 이미지 테스트
  - [ ] 카테고리 필터 테스트
  - [ ] 모바일 접속 테스트

- [ ] 성능 테스트 (1시간)
  - [ ] 동시 요청 테스트
  - [ ] 응답 시간 확인
  - [ ] Cloud Run 로그 확인

#### Day 6 오후 (2시간)
- [ ] 에러 처리 개선 (1시간)
  - [ ] 글로벌 exception handler
  - [ ] 에러 메시지 개선

- [ ] UI 개선 (1시간)
  - [ ] 로딩 상태 개선
  - [ ] 에러 메시지 개선

#### Day 7 (최종 점검)
- [ ] 발표 리허설 (1시간)
  - [ ] 발표 자료 리뷰
  - [ ] 데모 시나리오 연습
  - [ ] 예상 질문 준비

- [ ] 최종 배포 및 테스트 (1시간)
  - [ ] 최종 변경사항 반영
  - [ ] 모든 기능 동작 확인
  - [ ] 배포 URL 안정성 확인

- [ ] 발표 자료 마무리 (1시간)
  - [ ] 스크린샷 최신화
  - [ ] 성능 지표 업데이트
  - [ ] 백업 플랜 준비

---

## 📅 Week 2: 팀 통합 및 개선 (선택사항)

### Spring Boot팀 지원
- [ ] Spring Boot 연동 코드 리뷰
- [ ] API 엔드포인트 추가 요청 처리
- [ ] 에러 처리 가이드 제공
- [ ] 로컬 docker-compose 테스트 지원

### Next.js팀 지원
- [ ] Next.js 연동 코드 리뷰
- [ ] CORS 이슈 해결 지원
- [ ] UI 개선 제안
- [ ] 성능 최적화 팁 제공

### 성능 모니터링
- [ ] Cloud Run 메트릭 모니터링
- [ ] 에러율 추적
- [ ] 응답 시간 분석
- [ ] 비용 모니터링

### 추가 개선 (선택)
- [ ] Re-ranking 알고리즘 추가
- [ ] Query Expansion 구현
- [ ] 캐싱 시스템 추가
- [ ] Rate Limiting 추가

---

## 🔮 Future Work (발표 후)

### P1 (High Priority)
- [ ] 실사용 데이터 수집 (1개월)
- [ ] 카테고리별 성능 분석
- [ ] Quick Wins 적용 (Re-ranking)
- [ ] A/B 테스트 프레임워크

### P2 (Medium Priority)
- [ ] Last Layer Fine-Tuning
- [ ] 데이터베이스 연동 (PostgreSQL)
- [ ] 대시보드 구현 (Streamlit)
- [ ] 검색 히스토리 관리

### P3 (Low Priority)
- [ ] Multi-Domain Training (Full)
- [ ] Kubernetes 마이그레이션
- [ ] 모니터링 시스템 (Prometheus + Grafana)
- [ ] 로드 밸런싱 (Nginx)

---

## 📝 팀 커뮤니케이션

### 정기 체크인
- [ ] Week 1 Day 3: 중간 점검 (Spring Boot팀과)
- [ ] Week 1 Day 5: 통합 테스트 준비 (Next.js팀과)
- [ ] Week 1 Day 7: 최종 점검 (전체 팀)

### 공유할 정보
- [ ] 배포 URL (프로덕션)
- [ ] API 문서 링크
- [ ] 헬스 체크 URL
- [ ] 에러 로그 접근 방법
- [ ] 긴급 연락 방법

---

## 🎯 성공 기준

### Week 1 완료 시
- ✅ Cloud Run 배포 완료 (24시간 운영)
- ✅ CI/CD 자동화 (git push → 3분 자동 배포)
- ✅ 로깅 시스템 동작
- ✅ 클릭 추적 동작
- ✅ 발표 준비 완료
- ✅ 팀 통합 가이드 제공

### 비용
- ✅ 무료 (Cloud Run 무료 티어 범위 내)
- ✅ 월 200만 요청까지 무료

### 성능
- ✅ Top-5 Accuracy: 78%
- ✅ 검색 속도: 0.5초
- ✅ 가용성: 24/7

---

## 📚 참고 문서

### 통합 가이드
- [Spring Boot 통합](integration/SPRING_BOOT_INTEGRATION.md)
- [Next.js 통합](integration/NEXTJS_INTEGRATION.md)

### 배포 가이드
- [Cloud Run 30분 가이드](docs/CLOUDRUN_QUICKSTART.md)
- [1주일 완성 플랜](docs/WEEK_PLAN.md)
- [무료 배포 옵션](docs/FREE_DEPLOYMENT.md)

### 기술 문서
- [배포 비교](docs/DEPLOYMENT_COMPARISON.md)
- [상업적 요구사항](docs/commercial_requirements.md)
- [정리 계획](CLEANUP_PLAN.md)

---

## 🚀 다음 작업

**지금 당장:**
1. ✅ 프로젝트 정리 완료
2. ✅ 통합 가이드 작성 완료
3. ✅ README.md 업데이트 완료
4. ✅ TODO.md 업데이트 완료

**내일부터:**
5. ⬜ Day 1: Cloud Run 배포 시작
   - Google Cloud 계정 생성
   - gcloud CLI 설치
   - 첫 배포 완료

**1주일 후:**
6. ⬜ 발표 준비 완료
7. ⬜ 팀 통합 준비 완료

---

**화이팅! 🚀**
