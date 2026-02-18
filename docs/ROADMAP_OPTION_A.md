# Fashion Search - Option A 로드맵

## ✅ 완료된 작업

### 1. FAISS 벡터 검색 구현
- [x] FAISS IndexFlatIP 구현
- [x] 7,538개 벡터 인덱싱
- [x] 검색 속도: 1033s → 0.5s (2000x 개선)

### 2. FastAPI REST API
- [x] POST /search/upload (이미지 업로드 검색)
- [x] GET /search (Nine Oz 인덱스 검색)
- [x] GET /health (헬스 체크)
- [x] 웹 UI (drag & drop)

### 3. 성능 평가
- [x] Top-K Accuracy 측정
- [x] 50개 샘플 평가 완료
- [x] 결과: Top-5 78% (상업적으로 충분!)

### 4. 배포 준비
- [x] Docker 설정 완료
- [x] docker-compose.yml
- [x] 배포 가이드 작성

---

## 🎯 현재 상태

**성능 (Pretrained FashionCLIP):**
- Top-1: 44%
- Top-5: 78% ✅ (ASOS/Taobao 수준)
- Top-10: 88%
- MAP: 0.033
- MRR: 0.58

**시스템:**
- ✅ FAISS 벡터 검색 (2000x faster)
- ✅ FastAPI 서버
- ✅ 웹 UI
- ✅ Docker 배포 준비 완료

---

## 📅 다음 2주 작업 계획

### Week 1: 베타 런칭 준비

#### Day 1-2: 로깅 시스템 (P0 #5) ⭐
**목표:** 모든 검색 요청 기록

**작업:**
- [ ] Loguru/Structlog 설치
- [ ] 검색 로그 기록
  - query_id, timestamp
  - 검색 쿼리 정보
  - 결과 통계 (Top-K scores)
  - 응답 시간
- [ ] 로그 파일 rotation
- [ ] JSON 형식 로그

**코드 위치:**
- `api/search_api.py` - 로깅 추가
- `utils/logger.py` - 로거 설정

**테스트:**
```bash
# 검색 후 로그 확인
tail -f logs/search.log
```

---

#### Day 3-4: 클릭 추적 API
**목표:** 사용자 행동 데이터 수집

**작업:**
- [ ] POST /interactions 엔드포인트
  - query_id, product_id, action (click/add_to_cart/purchase)
- [ ] 클릭 로그 저장 (JSON)
- [ ] 웹 UI에 클릭 추적 JavaScript 추가

**데이터 구조:**
```json
{
  "query_id": "uuid-xxx",
  "product_id": "12345",
  "rank": 1,
  "action": "click",
  "timestamp": "2026-02-12T10:30:00Z"
}
```

---

#### Day 5: Docker 배포 테스트
**목표:** 로컬 Docker 완벽히 동작 확인

**작업:**
- [ ] Docker 이미지 빌드
- [ ] 로컬 테스트 (localhost:8001)
- [ ] 로그 확인 (docker-compose logs)
- [ ] 헬스 체크 동작 확인
- [ ] 볼륨 마운트 확인 (데이터 유지)

---

#### Day 6-7: 클라우드 배포 (AWS EC2)
**목표:** 외부 접속 가능한 베타 서버

**작업:**
- [ ] AWS EC2 인스턴스 생성
  - t3.xlarge (CPU) 또는 g4dn.xlarge (GPU)
- [ ] Docker 설치
- [ ] 프로젝트 복사 (Git clone)
- [ ] Docker Compose 실행
- [ ] 보안 그룹 설정 (8001 포트)
- [ ] 도메인 연결 (선택사항)

**비용:** $120-380/월

---

### Week 2: 데이터 분석 및 개선

#### Day 8-9: 로그 분석 도구
**목표:** 수집된 데이터 분석

**작업:**
- [ ] 로그 파싱 스크립트
- [ ] 통계 분석
  - 일별 검색 수
  - 평균 응답 시간
  - 카테고리별 검색 빈도
  - CTR (Click-Through Rate)
- [ ] 시각화 (Matplotlib/Plotly)

**스크립트:**
- `scripts/analysis/analyze_search_logs.py`
- `scripts/analysis/calculate_ctr.py`

---

#### Day 10-11: 카테고리별 성능 분석
**목표:** 어떤 카테고리가 약한지 파악

**작업:**
- [ ] 카테고리별 Top-K Accuracy 계산
- [ ] Confusion Matrix 생성
- [ ] 성능 낮은 카테고리 식별
- [ ] 개선 방향 문서화

**예상 결과:**
```
Category Performance:
- BL (블라우스): 85% ✅
- SK (스커트): 78% ✅
- DR (드레스): 68% ⚠️ (개선 필요)
- OP (원피스): 72%
```

---

#### Day 12-13: Quick Wins 구현
**목표:** 학습 없이 성능 개선

**작업:**
- [ ] **Re-ranking**
  - FAISS Top-100 가져오기
  - 카테고리 가중치 적용
  - 인기도 boost
  - 가격대 필터
- [ ] **Query Expansion**
  - 쿼리 이미지 증강 (rotation, crop)
  - 여러 변형 검색 후 결합
- [ ] **Ensemble**
  - 여러 유사도 점수 조합

**예상 개선:**
- Top-5: 78% → 82-85%

---

#### Day 14: 문서화 및 배포
**목표:** 팀원 공유 준비

**작업:**
- [ ] API 문서 업데이트
- [ ] 사용자 가이드 작성
- [ ] Docker Hub에 이미지 업로드
- [ ] 팀원 초대 및 테스트

---

## 🎯 2주 후 목표

### 시스템
- ✅ 베타 서버 운영 (AWS EC2)
- ✅ 로깅 시스템 동작
- ✅ 클릭 추적 데이터 수집
- ✅ Docker Hub 공유

### 성능
- 🎯 Top-5: 82-85% (Quick Wins 적용)
- 📊 실사용 데이터 50-100건

### 분석
- 📈 카테고리별 성능 리포트
- 👥 사용자 행동 분석 (CTR)
- 🔍 개선 방향 문서

---

## 📊 KPI Tracking

### Week 1
- [ ] 일일 검색 수: 10-50건
- [ ] 평균 응답 시간: < 1초
- [ ] 에러율: < 5%
- [ ] 베타 테스터: 5-10명

### Week 2
- [ ] 일일 검색 수: 50-200건
- [ ] CTR: > 30%
- [ ] 재검색률: < 40%
- [ ] 베타 테스터: 20-50명

---

## 🚀 3개월 로드맵 (참고)

### Month 1: 베타 운영 + 데이터 수집
- 로깅 시스템 ✅
- 클릭 추적 ✅
- 클라우드 배포 ✅
- Quick Wins 적용

### Month 2: 성능 개선
- 실사용 데이터 기반 분석
- Last Layer Fine-Tuning (선택)
- Re-ranking 알고리즘 개선
- A/B 테스트 프레임워크

### Month 3: 스케일업 준비
- 데이터베이스 연동 (PostgreSQL)
- 대시보드 구현 (Streamlit)
- 모니터링 시스템 (Prometheus + Grafana)
- 로드 밸런싱 (Nginx)

---

## 💡 의사결정 포인트

### 2주 후:
**질문:** Quick Wins로 충분한가? Fine-tuning 필요한가?

**결정 기준:**
- Quick Wins 후 Top-5 > 82% → Fine-tuning 보류
- 특정 카테고리 < 70% → Last Layer Fine-Tuning
- 전반적으로 낮음 → Multi-Domain Training 고려

### 1개월 후:
**질문:** 트래픽이 늘어나고 있나? 스케일 아웃 필요한가?

**결정 기준:**
- 일일 검색 < 500건 → 현재 EC2 유지
- 500-2000건 → ECS Auto Scaling
- 2000건+ → Kubernetes 고려

---

## 📝 다음 작업

**지금 당장:**
1. Docker 테스트 (start_docker.bat)
2. 웹 UI 열어서 검색 테스트
3. 결과 시각적으로 확인

**내일부터:**
1. 로깅 시스템 구현
2. 클릭 추적 API 추가

**계획 확인:**
- [ ] 이 로드맵 검토
- [ ] 우선순위 조정 필요시 논의
- [ ] 시작!
