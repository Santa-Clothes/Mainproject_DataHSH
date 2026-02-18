# Fashion Search 배포 가이드

## 📋 배포 옵션 비교

| 방법 | 난이도 | 비용 | 확장성 | 사용 케이스 |
|------|--------|------|--------|-------------|
| **로컬 테스트** | ⭐ | 무료 | ❌ | 개발/테스트 |
| **클라우드 VM** | ⭐⭐ | $10-50/월 | ⭐ | 베타 테스트 |
| **Docker** | ⭐⭐⭐ | $20-100/월 | ⭐⭐⭐ | 팀 공유, 프로덕션 |
| **Kubernetes** | ⭐⭐⭐⭐⭐ | $100+/월 | ⭐⭐⭐⭐⭐ | 대규모 트래픽 |

---

## 🎯 **추천 순서**

```
1단계 (지금): 로컬 테스트 → 베타 사용자 5-10명
2단계 (1주): 클라우드 VM → 베타 사용자 50-100명
3단계 (1개월): Docker → 정식 오픈 (수백-수천명)
4단계 (향후): K8s → 스케일 아웃 (수만명+)
```

---

## 1️⃣ 로컬 테스트 배포 (현재 상태)

### 사용 케이스
- 내부 테스트
- 소수 베타 테스터 (같은 네트워크)
- 데모 시연

### 방법
```bash
# 1. 서버 실행
py api/search_api.py

# 2. 공유 (같은 Wi-Fi 내)
# Windows에서 내 IP 확인
ipconfig  # 예: 192.168.0.123

# 베타 테스터에게 알려주기
http://192.168.0.123:8001/
```

### 장점
- 설정 필요 없음
- 무료
- 빠른 테스트

### 단점
- 같은 네트워크에서만 접근 가능
- 서버 꺼지면 서비스 중단
- 보안 없음

---

## 2️⃣ 클라우드 VM 배포 (간단한 배포)

### 사용 케이스
- 외부 베타 테스터 (인터넷으로 접근)
- MVP 런칭
- 소규모 트래픽 (하루 100-1000 검색)

### A. AWS EC2

```bash
# 1. EC2 인스턴스 생성
# - Ubuntu 22.04 LTS
# - g4dn.xlarge (GPU 있음, $0.526/시간)
# - 또는 t3.xlarge (GPU 없음, CPU만, $0.166/시간)
# - 보안 그룹: 8001 포트 오픈

# 2. SSH 접속
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. 프로젝트 복사
git clone https://github.com/your-repo/fashion-search.git
cd fashion-search

# 4. 환경 설정
sudo apt update
sudo apt install python3-pip python3-venv -y

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. 모델 및 데이터 다운로드
# (S3나 다른 스토리지에서 가져오기)
aws s3 sync s3://your-bucket/checkpoints ./checkpoints/
aws s3 sync s3://your-bucket/data ./data/

# 6. 서버 실행 (백그라운드)
nohup python api/search_api.py > server.log 2>&1 &

# 또는 systemd 서비스로 실행 (추천)
sudo nano /etc/systemd/system/fashion-search.service
```

**systemd 서비스 파일:**
```ini
[Unit]
Description=Fashion Search API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/fashion-search
Environment="PATH=/home/ubuntu/fashion-search/venv/bin"
ExecStart=/home/ubuntu/fashion-search/venv/bin/python api/search_api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl enable fashion-search
sudo systemctl start fashion-search
sudo systemctl status fashion-search

# 로그 확인
sudo journalctl -u fashion-search -f
```

### B. Google Cloud (GCP)

```bash
# 1. Compute Engine VM 생성
# - n1-standard-4 (4 vCPU, 15GB RAM)
# - 또는 n1-highmem-4 (4 vCPU, 26GB RAM)
# - GPU 추가: NVIDIA T4

# 2. 위와 동일한 방식으로 설정
```

### C. Azure

```bash
# 1. Virtual Machine 생성
# - Standard_NC6 (GPU 포함)
# - 또는 Standard_D4s_v3 (CPU only)

# 2. 위와 동일한 방식으로 설정
```

### 비용 예상
- **GPU 있음 (g4dn.xlarge)**: $380/월 (24시간 가동)
- **CPU만 (t3.xlarge)**: $120/월
- **절약 방법**: Reserved Instance로 50% 할인

---

## 3️⃣ Docker 배포 (추천!) 🐳

### 사용 케이스
- 팀원들과 공유
- 환경 통일 (개발/프로덕션)
- 쉬운 배포 및 업데이트
- CI/CD 파이프라인

### 단계 1: Dockerfile 작성

```dockerfile
# Dockerfile
FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# 포트 노출
EXPOSE 8001

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# 서버 실행
CMD ["python", "api/search_api.py"]
```

### 단계 2: docker-compose.yml

```yaml
# docker-compose.yml
version: '3.8'

services:
  fashion-search:
    build: .
    container_name: fashion-search-api
    ports:
      - "8001:8001"
    volumes:
      # 데이터는 외부 볼륨으로 마운트
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./results:/app/results
    environment:
      - DEVICE=cuda  # 또는 cpu
      - USE_FAISS=true
      - LOG_LEVEL=INFO
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 단계 3: .dockerignore

```
# .dockerignore
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.env.local
.git/
.gitignore
*.md
results/
logs/
.vscode/
.idea/
```

### 단계 4: 빌드 및 실행

```bash
# 1. Docker 이미지 빌드
docker build -t fashion-search:latest .

# 2. 실행 (docker-compose 사용)
docker-compose up -d

# 3. 로그 확인
docker-compose logs -f

# 4. 중지
docker-compose down

# 5. 재시작
docker-compose restart
```

### Docker Hub에 배포 (팀원 공유)

```bash
# 1. Docker Hub 로그인
docker login

# 2. 태그 지정
docker tag fashion-search:latest your-username/fashion-search:latest

# 3. 푸시
docker push your-username/fashion-search:latest

# 팀원이 사용:
docker pull your-username/fashion-search:latest
docker run -d -p 8001:8001 your-username/fashion-search:latest
```

---

## 4️⃣ 프로덕션 배포 (Docker + 추가 설정)

### A. Nginx Reverse Proxy (SSL + 로드밸런싱)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - fashion-search
    restart: unless-stopped

  fashion-search:
    build: .
    container_name: fashion-search-api
    expose:
      - "8001"
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
    environment:
      - DEVICE=cuda
      - USE_FAISS=true
    restart: unless-stopped
    deploy:
      replicas: 3  # 3개 인스턴스 (로드밸런싱)
      resources:
        limits:
          cpus: '2'
          memory: 8G
```

**nginx.conf:**
```nginx
http {
    upstream fashion_search {
        server fashion-search-1:8001;
        server fashion-search-2:8001;
        server fashion-search-3:8001;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # HTTPS로 리다이렉트
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://fashion_search;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
```

### B. 모니터링 추가 (Prometheus + Grafana)

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
```

---

## 5️⃣ Kubernetes 배포 (대규모)

### 사용 케이스
- 하루 수만 건 이상 검색
- 자동 스케일링 필요
- 고가용성 (99.9% uptime)

### deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fashion-search
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fashion-search
  template:
    metadata:
      labels:
        app: fashion-search
    spec:
      containers:
      - name: fashion-search
        image: your-username/fashion-search:latest
        ports:
        - containerPort: 8001
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: DEVICE
          value: "cuda"
        - name: USE_FAISS
          value: "true"
---
apiVersion: v1
kind: Service
metadata:
  name: fashion-search-service
spec:
  selector:
    app: fashion-search
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8001
  type: LoadBalancer
```

---

## 📊 **비용 비교 (월간)**

| 배포 방식 | 인프라 비용 | 트래픽 처리 | 유지보수 |
|-----------|-------------|-------------|----------|
| 로컬 | $0 | 10명 | 쉬움 |
| AWS t3.xlarge (CPU) | $120 | 100-500명 | 보통 |
| AWS g4dn.xlarge (GPU) | $380 | 500-2000명 | 보통 |
| Docker + ECS | $200-500 | 1000-5000명 | 보통 |
| Kubernetes (EKS) | $500-2000 | 5000-50000명 | 어려움 |

---

## 🎯 **추천 배포 경로**

### 지금 (베타 테스트):
```
Docker Compose + 클라우드 VM (1대)
↓
비용: $120-380/월
트래픽: 100-500명 처리 가능
```

### 1개월 후 (정식 오픈):
```
Docker + AWS ECS (Auto Scaling)
↓
비용: $200-500/월
트래픽: 1000-5000명 처리 가능
```

### 3개월 후 (성장 단계):
```
Kubernetes + Load Balancer
↓
비용: $500-2000/월
트래픽: 수만 명 처리 가능
```

---

## 🔐 보안 체크리스트

- [ ] HTTPS 설정 (Let's Encrypt)
- [ ] API Rate Limiting
- [ ] 환경변수 암호화 (.env → AWS Secrets Manager)
- [ ] CORS 설정 (허용된 도메인만)
- [ ] 로그 모니터링
- [ ] 정기 백업 (체크포인트, DB)
- [ ] 방화벽 설정
- [ ] DDoS 방어 (CloudFlare)

---

## 📝 다음 단계

1. **지금 당장**: Dockerfile 작성
2. **오늘**: Docker 로컬 테스트
3. **내일**: AWS EC2 + Docker 배포
4. **1주일 후**: 로깅 시스템 추가
5. **2주일 후**: 모니터링 대시보드
