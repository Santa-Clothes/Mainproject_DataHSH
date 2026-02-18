# 배포 옵션 상세 비교

## 🎯 발표/데모용 vs 상업용

### 발표/데모/포트폴리오용 (현재 상황)

| 항목 | 필요 수준 | 추천 솔루션 |
|------|-----------|-------------|
| **동시 접속자** | 10-50명 | Cloud Run, Oracle Cloud |
| **비용** | 무료 선호 | Cloud Run (무료 티어), Oracle (영구 무료) |
| **안정성** | 데모 때만 작동하면 OK | Cloud Run으로 충분 |
| **성능** | CPU로 충분 | GPU 불필요 |
| **관리** | 간단할수록 좋음 | Cloud Run (관리 불필요) |

**결론: EC2 불필요! Cloud Run 추천 ⭐**

---

### 상업용 (향후)

| 항목 | 필요 수준 | 추천 솔루션 |
|------|-----------|-------------|
| **동시 접속자** | 수백~수천명 | EC2, ECS, Kubernetes |
| **비용** | $100-500/월 OK | EC2 + Auto Scaling |
| **안정성** | 99.9% uptime | Load Balancer + 다중 인스턴스 |
| **성능** | GPU 필요 가능 | EC2 g4dn (GPU) |
| **관리** | DevOps 팀 | CI/CD 파이프라인 |

---

## 🐳 Docker 필요성 분석

### A. Google Cloud Run (Docker 필수)

```
Cloud Run = 컨테이너 실행 플랫폼
→ Docker 이미지만 실행 가능
→ Docker 없으면 배포 불가 ❌
```

**배포 흐름:**
```bash
1. Dockerfile 작성
2. 이미지 빌드
3. Google Container Registry에 푸시
4. Cloud Run이 컨테이너 실행

# Docker 없이는 불가능!
```

---

### B. Oracle Cloud Free Tier (Docker 선택사항)

```
Oracle Cloud = 일반 VM (가상 서버)
→ 원하는 대로 실행 가능
→ Docker도 가능, 직접 실행도 가능
```

#### Option 1: Docker 사용 (권장) ✅

```bash
# Oracle Cloud VM에서
git clone your-repo
cd fashion-search
docker-compose up -d

# 장점:
# - 환경 통일 (로컬과 동일)
# - 업데이트 쉬움 (docker pull)
# - 팀원과 동일 환경
```

#### Option 2: 직접 실행 (가능하지만 비추천)

```bash
# Oracle Cloud VM에서
git clone your-repo
cd fashion-search
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python api/search_api.py

# 단점:
# - 환경 차이 발생 가능 (Python 버전, 패키지 버전)
# - 의존성 관리 복잡
# - 팀원 환경과 다를 수 있음
```

---

### C. AWS EC2 (Docker 선택사항, 하지만 권장)

```
EC2도 일반 VM
→ Oracle Cloud와 동일
→ Docker 권장 (프로덕션 표준)
```

---

## 🤔 Docker를 왜 쓰는가?

### Docker의 진짜 목적

```
문제:
개발 환경 (내 컴퓨터):
- Python 3.11
- PyTorch 2.0
- Ubuntu WSL

배포 환경 (클라우드):
- Python 3.10 ❌
- PyTorch 1.9 ❌
- CentOS ❌

→ "내 컴퓨터에선 되는데요?" 문제 발생!
```

```
해결책: Docker
모든 환경을 "통째로 포장"
→ 어디서나 동일하게 실행
```

### Docker 없이 배포 시 문제들

1. **의존성 지옥**
```bash
# 로컬 (Windows)
pip install torch  # CUDA 11.8 버전 설치됨

# 클라우드 (Ubuntu)
pip install torch  # CUDA 12.1 버전 설치됨
→ 모델 로드 실패! ❌
```

2. **환경 변수**
```bash
# 로컬 .env
DEVICE=cuda

# 클라우드
DEVICE 설정 깜빡 → CPU 모드로 실행 (느림!)
```

3. **시스템 패키지**
```bash
# 로컬 (이미 설치됨)
libgomp1

# 클라우드 (설치 안 됨)
ImportError: libgomp.so.1 ❌
```

**Docker 사용 시:**
```bash
# 어디서나 동일
docker run your-image
→ 모든 것이 이미 포함됨 ✅
```

---

## 🔄 CI/CD 구조

### CI/CD란?

```
CI (Continuous Integration):
코드 변경 → 자동 테스트 → 자동 빌드

CD (Continuous Deployment):
빌드 완료 → 자동 배포 → 서비스 재시작
```

### 전통적 방법 (수동) ❌

```bash
1. 코드 수정
2. 로컬 테스트
3. Git push
4. SSH로 서버 접속
5. git pull
6. 서버 재시작
7. 에러 발생 시 4번부터 반복...

→ 시간 낭비, 실수 가능성 높음
```

### CI/CD 방법 (자동) ✅

```yaml
1. 코드 수정
2. Git push

# 이후 자동:
3. GitHub Actions 실행
4. 테스트 실행
5. Docker 이미지 빌드
6. Container Registry 푸시
7. 클라우드에 자동 배포
8. 헬스 체크
9. Slack/이메일 알림

→ 5분 안에 완료!
```

---

## 📦 FastAPI + Docker + CI/CD 예시

### 1. GitHub Actions (CI/CD 도구)

```yaml
# .github/workflows/deploy.yml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      # 1. 코드 체크아웃
      - uses: actions/checkout@v3

      # 2. Google Cloud 인증
      - uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      # 3. Docker 빌드
      - name: Build Docker image
        run: |
          docker build -t gcr.io/my-project/fashion-search:${{ github.sha }} .

      # 4. Container Registry 푸시
      - name: Push to GCR
        run: |
          docker push gcr.io/my-project/fashion-search:${{ github.sha }}

      # 5. Cloud Run 배포
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy fashion-search \
            --image gcr.io/my-project/fashion-search:${{ github.sha }} \
            --region us-central1

      # 6. 헬스 체크
      - name: Health check
        run: |
          curl -f https://fashion-search.run.app/health
```

### 2. 배포 흐름 시각화

```
[개발자]
    │
    ├─ git commit -m "Add new feature"
    ├─ git push
    │
    ▼
[GitHub]
    │
    ├─ GitHub Actions 트리거
    │
    ▼
[CI/CD Pipeline]
    │
    ├─ 1. 테스트 실행 (pytest)
    ├─ 2. Docker 이미지 빌드
    ├─ 3. Container Registry 푸시
    ├─ 4. Cloud Run 배포
    │
    ▼
[Cloud Run]
    │
    ├─ 새 버전 배포
    ├─ 헬스 체크 (10초)
    ├─ 트래픽 전환 (점진적)
    │
    ▼
[사용자]
    │
    └─ 자동으로 새 버전 사용!
```

**시간: 3-5분 (전부 자동!)**

---

## 🎯 발표용 배포 전략

### 추천: Google Cloud Run + GitHub Actions

#### 이유:
1. ✅ **완전 무료** (월 200만 요청)
2. ✅ **Docker 기반** (환경 통일)
3. ✅ **CI/CD 쉬움** (GitHub Actions 5분)
4. ✅ **관리 불필요** (자동 스케일링)
5. ✅ **HTTPS 자동**

#### 설정 시간:
- 초기 설정: 30분
- CI/CD 구축: 20분
- **총 50분이면 완성!**

#### 이후:
```bash
# 코드 수정 후
git commit -m "Update search algorithm"
git push

# 3분 후 자동 배포 완료!
# 아무것도 안 해도 됨
```

---

### 대안: Oracle Cloud (영구 무료)

#### 이유:
1. ✅ **완전 무료** (영구)
2. ✅ **4 vCPU + 24GB RAM**
3. ⭕ **Docker 선택사항**
4. ⭕ **수동 배포**

#### 설정 시간:
- 초기 설정: 1시간
- CI/CD 구축: 30분 (선택)

#### 배포:
```bash
# 방법 1: Docker (권장)
ssh oracle-vm
cd fashion-search
git pull
docker-compose restart

# 방법 2: 직접 실행
ssh oracle-vm
cd fashion-search
git pull
sudo systemctl restart fashion-search
```

---

## 💡 백엔드 배포 Q&A

### Q: 백 배포할 때 Docker 필요함?
**A: 필수는 아니지만 강력 권장!**

```
Docker 없이:
python api/search_api.py
→ 가능하지만 환경 관리 복잡

Docker 사용:
docker run fashion-search
→ 환경 통일, 관리 쉬움
```

### Q: FastAPI도 Docker 필요함?
**A: FastAPI = 백엔드 프레임워크**

```
FastAPI는 Python 라이브러리
→ Docker와는 별개

하지만:
FastAPI 앱을 Docker로 감싸면
→ 배포 쉬움, 환경 통일
```

### Q: CI/CD 없이 배포 가능?
**A: 가능하지만 비효율적**

```
CI/CD 없이 (수동):
1. 코드 수정
2. SSH 접속
3. git pull
4. 재시작
5. 에러 체크
→ 매번 5-10분

CI/CD 사용 (자동):
1. 코드 수정
2. git push
→ 끝! (자동 배포)
```

---

## 🚀 최종 추천 (발표용)

### Phase 1: 지금 (데모)
```bash
# ngrok (5분)
ngrok http 8001
→ 임시 URL 공유
```

### Phase 2: 발표 전 (1주일)
```bash
# Google Cloud Run (30분)
gcloud run deploy fashion-search
→ 영구 URL 생성
→ https://fashion-search.run.app
```

### Phase 3: 발표 후 (포트폴리오)
```bash
# GitHub Actions CI/CD (20분)
→ 자동 배포 설정
→ "git push"만 하면 자동 배포!
```

---

## 📋 체크리스트

### Docker가 필요한 경우:
- ✅ Google Cloud Run 사용
- ✅ 팀 프로젝트 (환경 통일)
- ✅ 프로덕션 배포
- ✅ CI/CD 구축

### Docker 없어도 되는 경우:
- ⭕ 개인 프로젝트 (혼자만 사용)
- ⭕ Oracle Cloud VM (직접 관리)
- ⭕ 일회성 데모
- ⭕ 로컬 개발

### CI/CD가 필요한 경우:
- ✅ 자주 업데이트
- ✅ 팀 협업
- ✅ 안정적 배포 필요
- ✅ 시간 절약

### CI/CD 없어도 되는 경우:
- ⭕ 일주일에 1번 미만 배포
- ⭕ 혼자 개발
- ⭕ 발표/데모용 (고정 버전)

---

## 💰 비용 비교 (월간)

| 방법 | 초기 비용 | 월 비용 | 트래픽 제한 |
|------|-----------|---------|-------------|
| **ngrok** | $0 | $0 | 무제한 (8시간 세션) |
| **Cloud Run** | $0 | $0 | 200만 요청 |
| **Oracle Cloud** | $0 | $0 | 무제한 |
| **AWS EC2 (t3.xlarge)** | $0 | $120 | 무제한 |
| **AWS EC2 (g4dn.xlarge)** | $0 | $380 | 무제한 |

**발표용 = Cloud Run 또는 Oracle Cloud 충분!**
