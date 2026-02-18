# Google Cloud Run - 30분 완성 가이드

## 🎯 발표/데모용 배포 (무료!)

**소요 시간: 30분**
**비용: $0** (월 200만 요청 무료)
**난이도: ⭐⭐⭐**

---

## 📋 준비물

- ✅ Google 계정 (Gmail)
- ✅ 신용카드 (무료지만 인증용 필요, 청구 안 됨)
- ✅ Dockerfile (이미 있음!)

---

## 🚀 Step-by-Step 가이드

### Step 1: Google Cloud 가입 (5분)

1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. "무료로 시작하기" 클릭
3. 결제 정보 입력 (무료 크레딧 $300 제공)
   - ⚠️ 무료 티어 범위 내면 청구 안 됨
   - 무료 티어 초과 시에만 청구

---

### Step 2: 프로젝트 생성 (2분)

```bash
# 프로젝트 이름 입력
fashion-search-demo
```

1. 콘솔 상단 프로젝트 선택
2. "새 프로젝트" 클릭
3. 프로젝트 이름: `fashion-search-demo`
4. "만들기" 클릭

---

### Step 3: gcloud CLI 설치 (5분)

#### Windows:
```bash
# 1. 다운로드
https://cloud.google.com/sdk/docs/install

# 2. 설치 완료 후 PowerShell에서
gcloud init

# 3. 로그인
# 브라우저가 열리면 Google 계정으로 로그인

# 4. 프로젝트 선택
# fashion-search-demo 선택
```

#### Mac:
```bash
brew install --cask google-cloud-sdk
gcloud init
```

#### Linux:
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

---

### Step 4: API 활성화 (1분)

```bash
# Cloud Run API 활성화
gcloud services enable run.googleapis.com

# Container Registry API 활성화
gcloud services enable containerregistry.googleapis.com

# Artifact Registry API 활성화
gcloud services enable artifactregistry.googleapis.com
```

---

### Step 5: Docker 이미지 빌드 및 푸시 (10분)

```bash
# 1. 프로젝트 루트로 이동
cd c:\Mainproject_DataHSH

# 2. 프로젝트 ID 확인
gcloud config get-value project
# 예: fashion-search-demo

# 3. Docker 이미지 태그
docker build -t gcr.io/fashion-search-demo/fashion-search:latest .

# 4. Docker 인증
gcloud auth configure-docker

# 5. 이미지 푸시 (5-7분 소요)
docker push gcr.io/fashion-search-demo/fashion-search:latest
```

**진행 중:**
```
The push refers to repository [gcr.io/fashion-search-demo/fashion-search]
abc123: Pushing [===========>                                       ]  50%
...
latest: digest: sha256:abc123... size: 1234
```

---

### Step 6: Cloud Run 배포 (2분)

```bash
# 배포 명령
gcloud run deploy fashion-search \
  --image gcr.io/fashion-search-demo/fashion-search:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8001 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 3 \
  --timeout 60s
```

**설명:**
- `--allow-unauthenticated`: 누구나 접속 가능 (발표용)
- `--memory 4Gi`: 메모리 4GB
- `--cpu 2`: CPU 2개
- `--max-instances 3`: 최대 3개 인스턴스 (자동 스케일링)
- `--timeout 60s`: 타임아웃 60초

**배포 완료:**
```
Deploying container to Cloud Run service [fashion-search]...
✓ Deploying new service... Done.
  ✓ Creating Revision...
  ✓ Routing traffic...
Done.
Service [fashion-search] revision [fashion-search-00001-xyz] has been deployed and is serving 100 percent of traffic.
Service URL: https://fashion-search-abc123-uc.a.run.app
```

---

### Step 7: 접속 테스트 (1분)

```bash
# 헬스 체크
curl https://fashion-search-abc123-uc.a.run.app/health

# 웹 브라우저에서 접속
https://fashion-search-abc123-uc.a.run.app/
```

**성공!** 🎉

---

## 🔄 코드 업데이트 (재배포)

```bash
# 1. 코드 수정 후

# 2. 새 이미지 빌드
docker build -t gcr.io/fashion-search-demo/fashion-search:v2 .

# 3. 푸시
docker push gcr.io/fashion-search-demo/fashion-search:v2

# 4. 재배포
gcloud run deploy fashion-search \
  --image gcr.io/fashion-search-demo/fashion-search:v2
```

---

## ⚡ CI/CD 추가 (GitHub Actions) - 선택사항

### 1. GitHub에 Secrets 추가

1. GitHub 저장소 → Settings → Secrets → New repository secret

2. 다음 Secrets 추가:

**GCP_PROJECT_ID:**
```
fashion-search-demo
```

**GCP_SA_KEY:**
```bash
# Service Account 키 생성
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

# 권한 부여
gcloud projects add-iam-policy-binding fashion-search-demo \
  --member="serviceAccount:github-actions@fashion-search-demo.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding fashion-search-demo \
  --member="serviceAccount:github-actions@fashion-search-demo.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding fashion-search-demo \
  --member="serviceAccount:github-actions@fashion-search-demo.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# 키 생성 (JSON 파일)
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@fashion-search-demo.iam.gserviceaccount.com

# key.json 내용을 복사하여 GitHub Secrets에 추가
cat key.json
```

---

### 2. GitHub Actions Workflow 생성

```yaml
# .github/workflows/deploy-cloudrun.yml
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

      - name: Notify deployment
        if: success()
        run: |
          echo "✅ Deployment successful!"
          echo "Service URL: $SERVICE_URL"
```

---

### 3. 자동 배포 테스트

```bash
# 1. Workflow 파일 추가
git add .github/workflows/deploy-cloudrun.yml
git commit -m "Add CI/CD workflow"
git push

# 2. GitHub Actions 탭에서 진행 상황 확인
# https://github.com/your-username/your-repo/actions

# 3. 3-5분 후 자동 배포 완료!
```

**이제부터:**
```bash
# 코드 수정 후
git commit -m "Update feature"
git push

# 자동으로:
# 1. 테스트 실행
# 2. Docker 빌드
# 3. Cloud Run 배포
# → 3분 안에 완료!
```

---

## 📊 모니터링

### Cloud Run 콘솔

```
https://console.cloud.google.com/run

확인 가능:
- 요청 수
- 응답 시간
- 에러율
- 메모리/CPU 사용량
- 로그
```

### 로그 확인

```bash
# 실시간 로그
gcloud run services logs read fashion-search --follow

# 최근 로그
gcloud run services logs read fashion-search --limit 50
```

---

## 💰 비용 모니터링

### 무료 티어 (월간)

```
✅ 2백만 요청
✅ 360,000 GB-초 메모리
✅ 180,000 vCPU-초
✅ 1GB 네트워크 송신

예상 사용량 (발표/데모):
- 요청: 1,000-5,000 (0.25%)
- 메모리: 10,000 GB-초 (2.7%)
- CPU: 5,000 vCPU-초 (2.7%)

→ 완전 무료! $0
```

### 비용 확인

```
https://console.cloud.google.com/billing

→ "예상 비용" 탭에서 실시간 확인
```

---

## 🔧 트러블슈팅

### 문제 1: Cold Start (첫 요청 느림)

**증상:**
```
첫 요청: 10-15초 소요
이후 요청: 1초 이내
```

**해결:**
```bash
# Minimum instances 설정 (유료)
gcloud run services update fashion-search \
  --min-instances 1
```

**무료 대안:**
```bash
# 발표 5분 전에 미리 요청
curl https://fashion-search-abc123-uc.a.run.app/health
```

---

### 문제 2: 메모리 부족

**증상:**
```
Container failed to start. Failed to start and then listen on the port defined by the PORT environment variable.
```

**해결:**
```bash
# 메모리 증가
gcloud run services update fashion-search \
  --memory 8Gi
```

---

### 문제 3: 타임아웃

**증상:**
```
Deadline exceeded
```

**해결:**
```bash
# 타임아웃 증가
gcloud run services update fashion-search \
  --timeout 300s
```

---

## 🎯 발표 전 체크리스트

- [ ] Cloud Run 배포 완료
- [ ] 공개 URL 접속 확인
- [ ] 웹 UI 동작 확인
- [ ] API 문서 확인 (/docs)
- [ ] 검색 기능 테스트 (이미지 업로드)
- [ ] 헬스 체크 통과 (/health)
- [ ] 모바일 접속 확인
- [ ] URL 저장 (발표 자료에 추가)

---

## 📝 발표 시나리오

```
1. "실제로 배포된 시스템을 보여드리겠습니다"
2. URL 공유: https://fashion-search-abc123-uc.a.run.app
3. 이미지 업로드 데모
4. 검색 결과 10개 표시
5. "Google Cloud Run으로 배포했으며,
    무료로 월 200만 요청까지 처리 가능합니다"
```

---

## 🚀 완료!

**배포 URL:**
```
https://fashion-search-abc123-uc.a.run.app
```

**소요 시간: 30분**
**비용: $0**
**유지비: $0/월** (무료 티어 범위 내)

**축하합니다! 이제 전 세계 어디서나 접속 가능한
Fashion Search 서비스가 완성되었습니다!** 🎉
