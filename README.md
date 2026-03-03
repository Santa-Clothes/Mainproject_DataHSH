# Fashion Image Search API

K-Fashion 이미지 기반 유사 상품 검색 시스템.
FashionCLIP + FAISS + Supabase 기반의 FastAPI 서버.

---

## 성능

| 지표 | 수치 |
|------|------|
| Top-1 Accuracy | 44% |
| Top-5 Accuracy | 78% |
| Top-10 Accuracy | 88% |
| MRR | 0.58 |
| 스타일 분류 Top-1 | 45.3% |
| 스타일 분류 Top-3 | 69.6% |
| 검색 대상 상품 수 | 7,538개 (Naver Shopping) |

---

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

> 첫 실행 시 HuggingFace에서 FashionCLIP 모델 자동 다운로드 (약 1GB, 인터넷 필요)

### 2. 환경변수 설정

프로젝트 루트의 `.env` 파일에서 Supabase 연결 정보를 확인합니다.

```
DATA_SOURCE=supabase
SUPABASE_URL=<Supabase 프로젝트 URL>
SUPABASE_KEY=<Supabase API 키>
NINEOZ_TABLE=internal_products_512
NAVER_TABLE=naver_products
FAISS_INDEX_PATH=data/indexes/naver.index
DEVICE=cpu
```

### 3. 서버 실행

```bash
uvicorn api.search_api:app --host 0.0.0.0 --port 8001 --reload
```

### 4. 접속

- API 문서: http://localhost:8001/docs
- 웹 UI: http://localhost:8001/static/search.html
- 헬스 체크: http://localhost:8001/health

---

## Gradio 데모

FastAPI 서버 실행 후 별도 터미널에서:

```bash
python gradio_demo.py
```

- Gradio UI: http://localhost:7860

이미지를 업로드하면 유사 상품 갤러리와 스타일 분류 차트를 확인할 수 있습니다.

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 및 모델 로딩 확인 |
| POST | `/search/upload` | 이미지 파일 → 유사 상품 검색 |
| GET | `/search` | 이미지 URL → 유사 상품 검색 |
| POST | `/embed` | 이미지 → 768차원 임베딩 벡터 |
| POST | `/analyze` | 이미지 → 임베딩 + 스타일 분류 (Top-3) |

### 예시

```bash
# 이미지 검색
curl -X POST http://localhost:8001/search/upload \
  -F "file=@test_img.jpg" \
  -F "top_k=5"

# 임베딩 추출
curl -X POST http://localhost:8001/embed \
  -F "file=@test_img.jpg"

# 스타일 분류
curl -X POST http://localhost:8001/analyze \
  -F "file=@test_img.jpg"
```

---

## 프로젝트 구조

```
FinalProject_v2/
├── api/
│   ├── search_api.py        # 메인 FastAPI 서버 (포트 8001)
│   ├── search_pipeline.py   # FAISS + Supabase 검색 파이프라인
│   └── vector_index.py      # FAISS 인덱스 로드/검색
├── models/
│   └── embedding_generator.py   # FashionCLIP 임베딩 생성
├── utils/
│   ├── config.py            # 환경변수 기반 설정
│   └── supabase_loader.py   # Supabase 데이터 로더
├── checkpoints/
│   └── style_classifier.pt  # 학습된 MLP 스타일 분류기
├── data/
│   └── indexes/
│       ├── naver.index      # FAISS 벡터 인덱스 (23MB)
│       └── naver.ids.npy    # 상품 ID 매핑
├── static/
│   └── search.html          # 웹 검색 UI
├── gradio_demo.py           # Gradio 데모 UI (포트 7860)
├── .env                     # 환경변수
└── requirements.txt
```

---

## 데이터 흐름

```
이미지 업로드
    ↓
FashionCLIP (patrickjohncyh/fashion-clip)
    ↓
768차원 임베딩 벡터
    ↓
FAISS 인덱스 검색 (data/indexes/naver.index)
    ↓
Supabase에서 상품 메타데이터 조회 (naver_products 테이블)
    ↓
JSON 응답 반환
```

---

## 기술 스택

- **Framework**: FastAPI (Python 3.11)
- **AI Model**: FashionCLIP (`patrickjohncyh/fashion-clip`)
- **Vector Search**: FAISS (faiss-cpu)
- **Database**: Supabase (PostgreSQL)
- **Style Classifier**: MLP (PyTorch, 23개 K-Fashion 스타일 → 10개 대분류)
- **Demo UI**: Gradio

---

## 참고

- [FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com)
- [Gradio](https://www.gradio.app)
