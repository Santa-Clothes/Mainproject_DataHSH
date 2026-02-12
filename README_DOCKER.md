# Fashion Search - Docker 사용 가이드

## 🚀 빠른 시작

### 1. Docker 설치

**Windows:**
- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) 다운로드 및 설치

**Mac:**
```bash
brew install --cask docker
```

**Linux:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

---

### 2. Docker 이미지 빌드

```bash
# 프로젝트 루트에서
cd c:\Mainproject_DataHSH

# Docker 이미지 빌드 (5-10분 소요)
docker build -t fashion-search:latest .
```

---

### 3. 실행

#### A. Docker Compose 사용 (추천)

```bash
# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

#### B. Docker 직접 실행

```bash
# CPU 모드
docker run -d \
  --name fashion-search \
  -p 8001:8001 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -e DEVICE=cpu \
  -e USE_FAISS=true \
  fashion-search:latest

# GPU 모드 (NVIDIA GPU 필요)
docker run -d \
  --name fashion-search \
  --gpus all \
  -p 8001:8001 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -e DEVICE=cuda \
  -e USE_FAISS=true \
  fashion-search:latest
```

---

### 4. 접속

```bash
# 웹 UI
http://localhost:8001/

# API 문서
http://localhost:8001/docs

# 헬스 체크
curl http://localhost:8001/health
```

---

## 📦 Docker Hub에 배포 (팀원 공유)

### 1. Docker Hub 계정 생성
[hub.docker.com](https://hub.docker.com/) 에서 가입

### 2. 로그인
```bash
docker login
```

### 3. 이미지 푸시
```bash
# 태그 지정 (your-username을 본인 계정으로 변경)
docker tag fashion-search:latest your-username/fashion-search:latest

# 푸시
docker push your-username/fashion-search:latest
```

### 4. 팀원이 사용
```bash
# 이미지 다운로드 및 실행
docker pull your-username/fashion-search:latest

docker run -d \
  --name fashion-search \
  -p 8001:8001 \
  -v ./data:/app/data \
  -v ./checkpoints:/app/checkpoints \
  your-username/fashion-search:latest
```

---

## 🔧 트러블슈팅

### 문제 1: 포트 충돌
```bash
# 에러: port is already allocated
# 해결: 다른 포트 사용
docker run -p 8002:8001 fashion-search:latest
```

### 문제 2: 데이터 파일 없음
```bash
# 에러: FileNotFoundError: data/csv/...
# 해결: 볼륨 마운트 확인
docker run -v "$(pwd)/data:/app/data" fashion-search:latest
```

### 문제 3: GPU 인식 안 됨
```bash
# NVIDIA Docker 런타임 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 📊 컨테이너 관리

### 상태 확인
```bash
docker ps                    # 실행 중인 컨테이너
docker ps -a                 # 모든 컨테이너
docker stats fashion-search  # 리소스 사용량
```

### 로그
```bash
docker logs fashion-search           # 전체 로그
docker logs -f fashion-search        # 실시간 로그
docker logs --tail 100 fashion-search # 마지막 100줄
```

### 재시작/중지
```bash
docker restart fashion-search  # 재시작
docker stop fashion-search     # 중지
docker start fashion-search    # 시작
docker rm fashion-search       # 삭제
```

### 컨테이너 내부 접속 (디버깅)
```bash
docker exec -it fashion-search bash

# 내부에서 Python 실행
python -c "from models.embedding_generator import FashionCLIPEmbeddingGenerator; print('OK')"
```

---

## 🌐 클라우드 배포

### AWS ECS (Elastic Container Service)
```bash
# 1. AWS CLI 설치 및 로그인
aws configure

# 2. ECR에 푸시
aws ecr create-repository --repository-name fashion-search
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker tag fashion-search:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fashion-search:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fashion-search:latest

# 3. ECS 클러스터 생성 (웹 콘솔에서)
```

### Google Cloud Run
```bash
# 1. gcloud CLI 설치
gcloud auth login

# 2. 배포 (1분이면 완료!)
gcloud run deploy fashion-search \
  --image fashion-search:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8001 \
  --memory 8Gi \
  --cpu 4
```

---

## 💰 비용 예상

### 클라우드 실행 비용 (월간)
- **AWS ECS**: $120-200 (t3.xlarge 기준)
- **Google Cloud Run**: $50-150 (사용량 기반)
- **Azure Container Instances**: $80-180

### 데이터 전송
- 10GB 업로드: ~$1
- 100GB 다운로드: ~$9

---

## 📝 다음 단계

1. ✅ Docker 이미지 빌드
2. ✅ 로컬 테스트
3. ⬜ Docker Hub 업로드
4. ⬜ 팀원에게 공유
5. ⬜ 클라우드 배포 (AWS/GCP)
