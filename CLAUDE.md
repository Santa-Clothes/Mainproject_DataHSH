# Fashion Search API - CLAUDE.md

## Review Instructions

이 프로젝트를 검토할 때 아래 순서를 따라주세요.

1. **환경 설정 확인**: `.env` 파일에 Supabase 연결 정보가 올바르게 설정되어 있는지 확인
2. **서버 실행**: 아래 "서버 실행" 섹션의 명령어로 API 서버 시작
3. **헬스 체크**: `GET /health` 응답에서 `pipeline_loaded: true` 확인
4. **기능 테스트**: 프로젝트 루트의 `test_img.jpg`로 `/search/upload`, `/embed`, `/analyze` 호출 테스트
5. **코드 구조**: 핵심 파일은 `api/search_api.py` (메인), `api/search_pipeline.py` (검색 로직), `models/embedding_generator.py` (임베딩)

---

## 프로젝트 개요

K-Fashion 이미지 기반 유사 상품 검색 시스템입니다.
- FashionCLIP 모델로 이미지를 768차원 벡터로 변환
- FAISS 인덱스로 네이버 쇼핑 상품과 유사도 검색
- MLP 분류기로 23개 K-Fashion 스타일 분류
- Spring 백엔드와 REST API로 연동

## 환경 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정 (.env 파일)
프로젝트 루트의 `.env` 파일을 열어 Supabase 정보를 설정합니다.

```
DATA_SOURCE=supabase          # 데이터 소스 (supabase 또는 csv)
SUPABASE_URL=<Supabase 프로젝트 URL>
SUPABASE_KEY=<Supabase API 키>
NINEOZ_TABLE=internal_products_512
NAVER_TABLE=naver_products
FAISS_INDEX_PATH=data/indexes/naver.index
DEVICE=cpu                    # cpu 또는 cuda
```

## 서버 실행

프로젝트 루트에서 실행:
```bash
uvicorn api.search_api:app --host 0.0.0.0 --port 8001 --reload
```

서버 시작 후 접속:
- API 문서: http://localhost:8001/docs
- 웹 UI: http://localhost:8001/static/search.html

## 주요 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 및 모델 로딩 확인 |
| POST | `/search/upload` | 이미지 파일 업로드 → 유사 상품 검색 |
| GET | `/search` | 이미지 URL → 유사 상품 검색 |
| POST | `/embed` | 이미지 → 768차원 임베딩 벡터 반환 |
| POST | `/analyze` | 이미지 → 임베딩 + 스타일 분류 (Top-3) |

### /search/upload 예시
```bash
curl -X POST http://localhost:8001/search/upload \
  -F "file=@test_img.jpg" \
  -F "top_k=5"
```

### /analyze 예시
```bash
curl -X POST http://localhost:8001/analyze \
  -F "file=@test_img.jpg"
# 반환: {"embedding": [...], "dimension": 768, "styles": [{"style": "밀리터리", "score": 0.82}, ...]}
```

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
│   ├── config.py            # 환경변수 기반 설정 (SystemConfig)
│   └── supabase_loader.py   # Supabase 데이터 로더
├── checkpoints/
│   └── style_classifier.pt  # 학습된 MLP 스타일 분류기
├── data/
│   └── indexes/
│       ├── naver.index      # FAISS 벡터 인덱스 (23MB)
│       └── naver.ids.npy    # 상품 ID 매핑
├── static/
│   └── search.html          # 웹 검색 UI
├── integration/
│   ├── README.md            # Spring 연동 가이드
│   └── examples/            # Java 예제 코드
├── .env                     # 환경변수 설정 파일
└── requirements.txt
```

## 데이터 흐름

```
이미지 업로드
    ↓
FashionCLIP (patrickjohncyh/fashion-clip)
    ↓
768차원 벡터
    ↓
FAISS 인덱스 검색 (naver.index)
    ↓
Supabase에서 상품 메타데이터 조회 (naver_products 테이블)
    ↓
JSON 응답 반환
```

## 스타일 분류기

- 모델: MLP (Linear → ReLU → Dropout → Linear)
- 입력: FashionCLIP visual projection 출력 (512차원)
- 출력: 23개 K-Fashion 스타일 확률
- 성능: Top-1 45.3%, Top-3 69.6%
- 체크포인트: `checkpoints/style_classifier.pt`

## Spring 연동

Spring 백엔드 연동 방법은 `integration/README.md` 참조.

주요 호출 예시 (Java):
```java
// POST /embed
MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
body.add("file", new FileSystemResource(imageFile));
EmbeddingResponse response = restTemplate.postForObject(
    "http://localhost:8001/embed", body, EmbeddingResponse.class
);
```

## 트러블슈팅

- **포트 충돌**: `netstat -ano | findstr :8001` 으로 PID 확인 후 `taskkill /F /PID <PID>`
- **Supabase 연결 실패**: `.env`의 SUPABASE_URL, SUPABASE_KEY 확인
- **FAISS 인덱스 없음**: `data/indexes/naver.index` 파일 존재 여부 확인
- **모델 다운로드 오류**: 첫 실행 시 HuggingFace에서 FashionCLIP 자동 다운로드 (약 1GB, 인터넷 필요)
