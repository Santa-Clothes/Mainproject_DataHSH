# Fashion Search API - Docker Image
FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgomp1 \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 의존성 설치 (캐싱 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사 (필요한 것만)
COPY api/ ./api/
COPY models/ ./models/
COPY utils/ ./utils/
COPY static/ ./static/

# FAISS 인덱스 복사 (검색에 필요)
COPY data/indexes/ ./data/indexes/

# 포트 노출
EXPOSE 8001

# 환경변수 기본값
# Note: Cloud Run has its own health check mechanism, so HEALTHCHECK is removed
ENV DEVICE=cpu \
    USE_FAISS=true \
    FAISS_INDEX_PATH=data/indexes/naver.index \
    LOG_LEVEL=INFO \
    DATA_SOURCE=supabase

# 서버 실행 (Cloud Run의 PORT 환경변수 사용, 기본값 8001)
CMD sh -c "uvicorn api.search_api:app --host 0.0.0.0 --port ${PORT:-8001}"
