# Fashion Search - 무료 배포 옵션

## 🎯 배포 방법 비교

| 방법 | 비용 | 설정 난이도 | 접속 | 제한사항 |
|------|------|-------------|------|----------|
| **로컬 네트워크** | 무료 | ⭐ | 같은 WiFi만 | 외부 접속 불가 |
| **ngrok** | 무료 | ⭐⭐ | 인터넷 전체 | 8시간 세션 제한 |
| **Cloudflare Tunnel** | 무료 | ⭐⭐ | 인터넷 전체 | 무제한! ⭐ |
| **Google Cloud Run** | 무료 | ⭐⭐⭐ | 인터넷 전체 | 월 200만 요청 |
| **Oracle Cloud** | 무료 | ⭐⭐⭐⭐ | 인터넷 전체 | 영구 무료! |
| **AWS EC2** | 유료 | ⭐⭐⭐⭐ | 인터넷 전체 | $120-380/월 |

---

## 1️⃣ 로컬 네트워크 공유 (완전 무료)

### 사용 케이스
- 사무실/학교 같은 WiFi
- 내부 테스트
- 소수 인원 (5-10명)

### 방법

```bash
# 1. 본인 컴퓨터에서 Docker 실행
docker-compose up -d

# 2. 본인 IP 확인 (Windows)
ipconfig
# 예: 192.168.0.123

# 3. 같은 WiFi 사용자에게 알려주기
http://192.168.0.123:8001/
```

### 장점
- ✅ 완전 무료
- ✅ 설정 간단
- ✅ 보안 걱정 없음

### 단점
- ❌ 같은 네트워크만 접속 가능
- ❌ 본인 컴퓨터 켜져있어야 함
- ❌ 외부 인터넷 불가

---

## 2️⃣ ngrok (무료, 임시 URL) ⭐

### 사용 케이스
- 빠른 데모/시연
- 외부 베타 테스터 (일시적)
- 개발 중 테스트

### 설정 (5분이면 완료!)

```bash
# 1. ngrok 설치
# Windows: https://ngrok.com/download
# Mac: brew install ngrok
# Linux: snap install ngrok

# 2. 회원가입 (무료)
# https://dashboard.ngrok.com/signup

# 3. 인증 토큰 설정
ngrok config add-authtoken YOUR_TOKEN

# 4. 로컬 서버 실행
docker-compose up -d

# 5. ngrok 터널 생성
ngrok http 8001
```

**결과:**
```
ngrok

Session Status                online
Account                       your-email@example.com (Plan: Free)
Version                       3.1.0
Region                        United States (us)
Latency                       23ms
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://abc123.ngrok-free.app -> localhost:8001

Connections                   ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

**공유 URL:**
```
https://abc123.ngrok-free.app
→ 인터넷 어디서나 접속 가능!
```

### 장점
- ✅ 완전 무료 (Free tier)
- ✅ 설정 초간단 (5분)
- ✅ HTTPS 자동 제공
- ✅ 인터넷 어디서나 접속

### 단점
- ❌ 8시간마다 재시작 필요
- ❌ URL이 매번 바뀜 (유료는 고정)
- ❌ 본인 컴퓨터 켜져있어야 함

### 무료 플랜 제한
- ✅ 무제한 터널
- ✅ 40 connections/분
- ❌ URL 고정 불가 (유료: $8/월)

---

## 3️⃣ Cloudflare Tunnel (무료, 영구) ⭐⭐⭐

### 사용 케이스
- 장기 베타 테스트
- 팀 내부 공유
- ngrok보다 안정적

### 설정

```bash
# 1. Cloudflare 계정 생성 (무료)
# https://dash.cloudflare.com/sign-up

# 2. cloudflared 설치
# Windows: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation
# Mac: brew install cloudflare/cloudflare/cloudflared
# Linux: wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared-linux-amd64.deb

# 3. 로그인
cloudflared tunnel login

# 4. 터널 생성
cloudflared tunnel create fashion-search

# 5. 설정 파일 생성
# ~/.cloudflared/config.yml
```

**config.yml:**
```yaml
tunnel: <TUNNEL-ID>
credentials-file: /home/user/.cloudflared/<TUNNEL-ID>.json

ingress:
  - hostname: fashion-search.your-domain.com
    service: http://localhost:8001
  - service: http_status:404
```

```bash
# 6. DNS 설정
cloudflared tunnel route dns fashion-search fashion-search.your-domain.com

# 7. 터널 실행
cloudflared tunnel run fashion-search
```

### 장점
- ✅ **완전 무료**
- ✅ **무제한 사용**
- ✅ 고정 URL
- ✅ DDoS 방어 자동
- ✅ HTTPS 자동

### 단점
- ❌ 설정이 약간 복잡
- ❌ 도메인 필요 (무료 도메인도 가능)
- ❌ 본인 컴퓨터 켜져있어야 함

---

## 4️⃣ Google Cloud Run (무료 티어) ⭐⭐⭐

### 사용 케이스
- 실제 서비스 런칭
- 자동 스케일링
- 서버 관리 불필요

### 무료 티어
```
✅ 월 200만 요청 무료
✅ 360,000 GB-초 메모리 무료
✅ 180,000 vCPU-초 무료
✅ 1GB 아웃바운드 무료

→ 소규모 서비스는 영구 무료!
```

### 설정

```bash
# 1. gcloud CLI 설치
# https://cloud.google.com/sdk/docs/install

# 2. 로그인
gcloud auth login

# 3. 프로젝트 생성
gcloud projects create fashion-search-project
gcloud config set project fashion-search-project

# 4. Cloud Run API 활성화
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 5. Docker 이미지 빌드 및 푸시
docker build -t gcr.io/fashion-search-project/fashion-search:latest .
docker push gcr.io/fashion-search-project/fashion-search:latest

# 6. Cloud Run 배포
gcloud run deploy fashion-search \
  --image gcr.io/fashion-search-project/fashion-search:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8001 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 3
```

**결과:**
```
Service URL: https://fashion-search-abc123-uc.a.run.app
→ 자동으로 HTTPS, 글로벌 접속 가능!
```

### 장점
- ✅ **무료 티어 넉넉함**
- ✅ 24시간 돌아감 (컴퓨터 꺼도 됨!)
- ✅ 자동 스케일링
- ✅ HTTPS 자동
- ✅ 관리 불필요

### 단점
- ❌ GPU 지원 안 됨 (CPU만)
- ❌ Cold start (5초 대기)
- ❌ 설정이 약간 복잡

### 비용 예상
```
월 1,000건 검색 (각 1초 소요)
= 1,000초 vCPU 사용
= 무료 티어 범위 내 → $0

월 10,000건 검색
= 여전히 무료 티어 범위 → $0

월 100,000건 검색
= 약간 초과 → $5-10
```

---

## 5️⃣ Oracle Cloud (영구 무료!) ⭐⭐⭐⭐

### 사용 케이스
- 완전 무료로 24시간 서비스
- VM 필요 (EC2 대체)

### 무료 티어 (평생!)
```
✅ VM.Standard.E2.1.Micro (AMD)
   - 1 vCPU, 1GB RAM
   - 2개까지 무료!

✅ VM.Standard.A1.Flex (ARM)
   - 4 vCPU, 24GB RAM
   - 영구 무료! ⭐⭐⭐

✅ 200GB 스토리지
✅ 10TB 아웃바운드 월간
```

### 설정

```bash
# 1. Oracle Cloud 계정 생성
# https://www.oracle.com/cloud/free/

# 2. Compute Instance 생성
# - Shape: VM.Standard.A1.Flex (ARM)
# - OS: Ubuntu 22.04
# - 4 OCPUs, 24GB RAM

# 3. SSH 접속
ssh -i your-key ubuntu@<instance-ip>

# 4. Docker 설치
sudo apt update
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker $USER

# 5. 프로젝트 복사
git clone https://github.com/your-repo/fashion-search.git
cd fashion-search

# 6. 실행
docker-compose up -d

# 7. 방화벽 오픈
# Oracle Cloud Console에서:
# Networking → Security Lists → Ingress Rules
# - Source: 0.0.0.0/0
# - Protocol: TCP
# - Port: 8001
```

**접속:**
```
http://<instance-ip>:8001/
```

### 장점
- ✅ **영구 무료!**
- ✅ 24시간 돌아감
- ✅ 4 vCPU + 24GB RAM (ARM)
- ✅ 관리 권한 완전

### 단점
- ❌ ARM 아키텍처 (호환성 주의)
- ❌ GPU 없음
- ❌ 설정 복잡
- ❌ 가끔 리소스 부족으로 생성 실패

---

## 6️⃣ Fly.io (무료 티어)

### 무료 티어
```
✅ 3개 VM (shared-cpu-1x, 256MB)
✅ 160GB 아웃바운드/월
✅ 자동 스케일링

→ 작은 앱은 무료!
```

### 설정

```bash
# 1. flyctl 설치
# https://fly.io/docs/hands-on/install-flyctl/

# 2. 로그인
fly auth login

# 3. 앱 생성
fly launch
# → 자동으로 Dockerfile 감지

# 4. 배포
fly deploy

# 5. 접속
fly open
```

### 장점
- ✅ 무료 티어
- ✅ 설정 초간단
- ✅ 글로벌 CDN

### 단점
- ❌ 256MB RAM (작음)
- ❌ GPU 없음
- ❌ 무료 티어 제한

---

## 🎯 **추천 순서**

### 지금 당장 (데모/테스트):
```
1. ngrok (5분 설정) ⭐
→ 외부 테스터에게 바로 공유
```

### 1주일 베타 테스트:
```
2. Cloudflare Tunnel (무료, 고정 URL)
→ 안정적인 베타 테스트
```

### 정식 런칭:
```
3. Google Cloud Run (무료 티어)
→ 월 200만 요청 무료

또는

4. Oracle Cloud (영구 무료!)
→ VM 완전 제어, 24GB RAM
```

---

## 💡 **실전 전략**

### Phase 1: 내부 테스트 (지금)
```
로컬 Docker + ngrok
→ 비용: $0
→ 기간: 1-2주
```

### Phase 2: 베타 테스트 (다음 주)
```
Google Cloud Run 또는 Oracle Cloud
→ 비용: $0 (무료 티어)
→ 기간: 1-2개월
```

### Phase 3: 정식 서비스 (트래픽 증가 시)
```
AWS/GCP/Azure (유료)
→ 비용: $100-500/월
→ 자동 스케일링
```

---

## 🚀 **지금 바로 시작 (ngrok)**

```bash
# 1. ngrok 다운로드
# https://ngrok.com/download

# 2. 압축 풀기 및 실행
ngrok.exe http 8001

# 3. 공개 URL 복사
https://abc123.ngrok-free.app

# 4. 친구/테스터에게 공유!
```

**5분이면 끝!** ✨
