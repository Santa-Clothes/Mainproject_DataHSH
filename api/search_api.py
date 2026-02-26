"""
Fashion Search API
===================

Nine Oz → K-Fashion → Naver Shopping 검색 시스템
FastAPI 기반 REST API
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import time
import uuid
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.search_pipeline import SearchPipeline
from utils.config import get_system_config

# 시스템 설정 로드
config = get_system_config()

# FastAPI 앱 생성
app = FastAPI(
    title="Fashion Search API",
    description="Nine Oz → K-Fashion → Naver Shopping 검색 시스템",
    version="2.0.0",  # Updated with real embedding support
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # credentials 비활성화 (allow_origins="*"와 함께 사용 불가)
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (static 폴더)
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 전역 파이프라인 인스턴스
pipeline: Optional[SearchPipeline] = None

# 스타일 분류기 전역 변수
STYLES = [
    "레트로", "로맨틱", "리조트", "매니시", "모던",
    "밀리터리", "섹시", "소피스트케이티드", "스트리트", "스포티",
    "아방가르드", "오리엔탈", "웨스턴", "젠더리스", "컨트리",
    "클래식", "키치", "톰보이", "펑크", "페미닌",
    "프레피", "히피", "힙합",
]

# 23개 세부 스타일 → 10개 대분류 매핑
STYLE_MAPPING = {
    "클래식":          "트래디셔널",
    "프레피":          "트래디셔널",
    "매니시":          "매니시",
    "톰보이":          "매니시",
    "페미닌":          "페미닌",
    "로맨틱":          "페미닌",
    "섹시":            "페미닌",
    "히피":            "에스닉",
    "웨스턴":          "에스닉",
    "오리엔탈":        "에스닉",
    "모던":            "컨템포러리",
    "소피스트케이티드": "컨템포러리",
    "아방가르드":      "컨템포러리",
    "컨트리":          "내추럴",
    "리조트":          "내추럴",
    "젠더리스":        "젠더플루이드",
    "스포티":          "스포티",
    "레트로":          "서브컬처",
    "키치":            "서브컬처",
    "힙합":            "서브컬처",
    "펑크":            "서브컬처",
    "밀리터리":        "캐주얼",
    "스트리트":        "캐주얼",
}

TOP10_STYLES = [
    "트래디셔널", "매니시", "페미닌", "에스닉", "컨템포러리",
    "내추럴", "젠더플루이드", "스포티", "서브컬처", "캐주얼",
]
CLASSIFIER_PATH = Path(__file__).parent.parent / "checkpoints" / "style_classifier.pt"
clip_model: CLIPModel = None
clip_processor: CLIPProcessor = None
style_classifier: nn.Sequential = None
style_labels: List[str] = STYLES


# ---------- Embed/Analyze Response 모델 ----------

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int


class StylePrediction(BaseModel):
    style: str
    score: float


class AnalyzeResponse(BaseModel):
    embedding: List[float]
    dimension: int
    styles: List[StylePrediction]


# ---------- Search Request/Response 모델 ----------
class SearchRequest(BaseModel):
    """검색 요청"""

    query_index: int = Field(..., description="나인오즈 CSV 인덱스", ge=0)
    initial_k: int = Field(100, description="초기 검색 결과 수", ge=1, le=500)
    final_k: int = Field(10, description="최종 반환 결과 수", ge=1, le=100)


class ProductInfo(BaseModel):
    """제품 정보"""

    product_id: str
    title: str
    price: float
    image_url: str
    category_id: str
    style_id: str
    score: float


class SearchResponse(BaseModel):
    """검색 응답"""

    query: Dict
    results: List[ProductInfo]
    stats: Dict


class HealthResponse(BaseModel):
    """헬스체크 응답"""

    status: str
    model_loaded: bool
    nineoz_count: int
    naver_count: int


# API 엔드포인트
@app.on_event("startup")
async def startup_event():
    """API 시작 시 파이프라인 초기화"""
    global pipeline

    print("\n" + "="*80)
    print("Initializing Fashion Search API")
    print("="*80)
    print(f"Data Source: {config.data_source}")
    if config.data_source == "supabase":
        print(f"Supabase URL: {config.supabase_url}")
        print(f"Nine Oz Table: {config.nineoz_table}")
        print(f"Naver Table: {config.naver_table}")
    else:
        print(f"Nine Oz CSV: {config.nineoz_csv_path}")
        print(f"Naver CSV: {config.naver_csv_path}")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Device: {config.device or 'auto'}")
    print(f"Use FAISS: {config.use_faiss}")
    if config.use_faiss:
        print(f"FAISS Index: {config.faiss_index_path}")
    print(f"Precompute embeddings: {config.precompute_embeddings}")
    print("="*80)

    try:
        pipeline = SearchPipeline(
            nineoz_csv_path=config.nineoz_csv_path,
            naver_csv_path=config.naver_csv_path,
            checkpoint_path=config.checkpoint_path,
            device=config.device,
            precompute_embeddings=config.precompute_embeddings,
            faiss_index_path=config.faiss_index_path if config.use_faiss else None,
            use_faiss=config.use_faiss,
            data_source=config.data_source,
            supabase_url=config.supabase_url,
            supabase_key=config.supabase_key,
            nineoz_table=config.nineoz_table,
            naver_table=config.naver_table,
        )
        print("\n[OK] Pipeline initialized successfully!")
    except Exception as e:
        print(f"\n[WARNING] Pipeline initialization failed: {e}")
        print("[INFO] API will start in limited mode (health check only)")
        pipeline = None

    # 스타일 분류기 로드
    global clip_model, clip_processor, style_classifier, style_labels
    try:
        clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        print("[OK] FashionCLIP (full) loaded for style classifier.")
    except Exception as e:
        print(f"[WARNING] FashionCLIP full load failed: {e}")

    if CLASSIFIER_PATH.exists() and clip_model is not None:
        ckpt = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
        emb_dim = ckpt["emb_dim"]
        num_classes = ckpt["num_classes"]
        style_classifier = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        style_classifier.load_state_dict(ckpt["classifier_state_dict"])
        style_classifier.eval()
        style_labels = ckpt.get("styles", STYLES)
        acc = ckpt.get("val_top1", 0) * 100
        print(f"[OK] Style classifier loaded. Val Top-1: {acc:.1f}%")
    else:
        print("[WARNING] Style classifier not found — /analyze unavailable.")

    print("="*80)


@app.get("/")
async def root():
    """루트 엔드포인트 - 웹 UI 제공"""
    html_path = Path(__file__).parent.parent / "static" / "search.html"
    if html_path.exists():
        return FileResponse(html_path)
    else:
        return {
            "message": "Fashion Search API",
            "version": "2.0.0",
            "endpoints": {
                "health": "/health",
                "search": "/search",
                "search_upload": "/search/upload",
                "query_item": "/query/{index}",
                "docs": "/docs",
            },
        }


@app.get("/api")
async def api_info():
    """API 정보"""
    return {
        "message": "Fashion Search API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "search_upload": "/search/upload",
            "query_item": "/query/{index}",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크"""
    if pipeline is None:
        return HealthResponse(
            status="limited",
            model_loaded=False,
            nineoz_count=0,
            naver_count=0,
        )

    return HealthResponse(
        status="healthy",
        model_loaded=pipeline.embedding_generator is not None,
        nineoz_count=len(pipeline.nineoz_df),
        naver_count=len(pipeline.naver_df),
    )


@app.get("/query/{index}", response_model=Dict)
async def get_query_item(index: int):
    """
    나인오즈 쿼리 아이템 가져오기

    Args:
        index: 나인오즈 CSV 인덱스

    Returns:
        쿼리 아이템 정보
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        query_item = pipeline.get_query_item(index)
        return query_item
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    검색 수행 (POST)

    Args:
        request: 검색 요청 (query_index, initial_k, final_k)

    Returns:
        검색 결과
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.search(
            query_index=request.query_index,
            initial_k=request.initial_k,
            final_k=request.final_k,
        )

        # ProductInfo 형식으로 변환
        products = [
            ProductInfo(
                product_id=item["product_id"],
                title=item["title"],
                price=item["price"],
                image_url=item["image_url"],
                category_id=item["category_id"],
                style_id=item.get("style_id", ""),
                score=item["score"],
            )
            for item in result["results"]
        ]

        return SearchResponse(
            query=result["query"], results=products, stats=result["stats"]
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/search", response_model=SearchResponse)
async def search_get(
    query_index: int = Query(..., description="나인오즈 CSV 인덱스", ge=0),
    initial_k: int = Query(100, description="초기 검색 결과 수", ge=1, le=500),
    final_k: int = Query(10, description="최종 반환 결과 수", ge=1, le=100),
):
    """
    검색 수행 (GET)

    Args:
        query_index: 나인오즈 CSV 인덱스
        initial_k: 초기 검색 결과 수
        final_k: 최종 반환 결과 수

    Returns:
        검색 결과
    """
    request = SearchRequest(
        query_index=query_index, initial_k=initial_k, final_k=final_k
    )
    return await search(request)


@app.post("/search/upload")
async def search_by_upload(
    file: UploadFile = File(..., description="이미지 파일 (JPG, PNG)"),
    category_filter: Optional[str] = Query(None, description="카테고리 필터 (optional)"),
    top_k: int = Query(10, description="반환할 결과 수", ge=1, le=100),
):
    """
    이미지 파일 업로드로 검색

    Args:
        file: 업로드된 이미지 파일
        category_filter: K-Fashion 카테고리 필터 (선택사항)
        top_k: 반환할 결과 수

    Returns:
        검색 결과 리스트
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # 이미지 파일 타입 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be an image."
        )

    try:
        # 전체 시간 측정 시작
        total_start = time.time()

        # 쿼리 ID 생성
        query_id = str(uuid.uuid4())

        # 이미지 파일 읽기
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # RGB로 변환 (RGBA, Grayscale 등 처리)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 검색 수행 (시간 측정)
        search_start = time.time()
        results = pipeline.search_by_image(
            image_source=image,
            category_filter=category_filter,
            initial_k=100,
            final_k=top_k
        )
        search_time = time.time() - search_start

        # 전체 시간
        total_time = time.time() - total_start

        # 결과 통계 계산
        scores = [r['score'] for r in results] if results else []
        avg_score = np.mean(scores) if scores else 0.0
        max_score = np.max(scores) if scores else 0.0
        min_score = np.min(scores) if scores else 0.0

        # 점수 분포 계산
        score_distribution = {
            "0.8-1.0": sum(1 for s in scores if s >= 0.8),
            "0.6-0.8": sum(1 for s in scores if 0.6 <= s < 0.8),
            "0.4-0.6": sum(1 for s in scores if 0.4 <= s < 0.6),
            "0.0-0.4": sum(1 for s in scores if s < 0.4),
        }

        # 랭크 추가
        for rank, result in enumerate(results, 1):
            result['rank'] = rank

        return {
            "query": {
                "query_id": query_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "image_info": {
                    "filename": file.filename,
                    "size": len(contents),
                    "dimensions": f"{image.size[0]}x{image.size[1]}",
                    "format": image.format or "Unknown"
                }
            },
            "results": results,
            "metrics": {
                "total_results": len(results),
                "search_time_ms": int(search_time * 1000),
                "total_time_ms": int(total_time * 1000),
                "category_filter": category_filter,
                "faiss_enabled": pipeline.use_faiss
            },
            "stats": {
                "avg_score": float(avg_score),
                "max_score": float(max_score),
                "min_score": float(min_score),
                "score_distribution": score_distribution
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_image(file: UploadFile = File(...)):
    """이미지 → 768차원 임베딩 벡터 반환 (Spring 검색용)"""
    if pipeline is None or pipeline.embedding_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"이미지 파일만 허용: {file.content_type}")

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    embedding = pipeline.embedding_generator.generate_embedding(image, normalize=True)

    return EmbeddingResponse(embedding=embedding.tolist(), dimension=len(embedding))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...), top_k: int = 3):
    """이미지 → 임베딩 + K-Fashion 스타일 분류 (Top-3)"""
    if pipeline is None or pipeline.embedding_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"이미지 파일만 허용: {file.content_type}")

    if style_classifier is None or clip_model is None:
        raise HTTPException(status_code=503, detail="Style classifier not loaded")

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # 1. 임베딩 생성
    embedding = pipeline.embedding_generator.generate_embedding(image, normalize=True)

    # 2. 스타일 분류
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vision_out = clip_model.vision_model(**inputs)
        img_feat = vision_out.pooler_output
        img_feat = clip_model.visual_projection(img_feat)
        img_feat = F.normalize(img_feat, p=2, dim=-1)
        logits = style_classifier(img_feat)[0]
        probs = torch.softmax(logits, dim=0)

    # 23개 확률 → 10개 대분류로 집계
    top10_probs: Dict[str, float] = {s: 0.0 for s in TOP10_STYLES}
    for i, label in enumerate(style_labels):
        parent = STYLE_MAPPING.get(label)
        if parent:
            top10_probs[parent] += probs[i].item()

    # 합계로 재정규화 (매핑 누락 항목 대비)
    total = sum(top10_probs.values())
    if total > 0:
        top10_probs = {k: v / total for k, v in top10_probs.items()}

    sorted_styles = sorted(top10_probs.items(), key=lambda x: x[1], reverse=True)
    top_k_capped = min(top_k, len(sorted_styles))
    styles = [
        StylePrediction(style=s, score=round(p, 4))
        for s, p in sorted_styles[:top_k_capped]
    ]

    return AnalyzeResponse(
        embedding=embedding.tolist(),
        dimension=len(embedding),
        styles=styles,
    )


@app.get("/categories", response_model=Dict)
async def get_categories():
    """
    사용 가능한 카테고리 목록

    Returns:
        카테고리 통계
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # 나인오즈 카테고리 통계
    nineoz_categories = (
        pipeline.nineoz_df["category_id"].value_counts().to_dict()
        if "category_id" in pipeline.nineoz_df.columns else {}
    )

    # 네이버 카테고리 통계
    naver_categories = (
        pipeline.naver_df["category_id"].value_counts().to_dict()
        if "category_id" in pipeline.naver_df.columns else {}
    )

    return {
        "nineoz_categories": nineoz_categories,
        "naver_categories": naver_categories,
    }


@app.get("/stats", response_model=Dict)
async def get_statistics():
    """
    데이터셋 통계

    Returns:
        통계 정보
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return {
        "nineoz": {
            "total_products": len(pipeline.nineoz_df),
            "unique_categories": pipeline.nineoz_df["category_id"].nunique()
            if "category_id" in pipeline.nineoz_df.columns else 0,
            "unique_styles": pipeline.nineoz_df["style_id"].nunique()
            if "style_id" in pipeline.nineoz_df.columns else 0,
            "unique_colors": pipeline.nineoz_df["color"].nunique()
            if "color" in pipeline.nineoz_df.columns else 0,
        },
        "naver": {
            "total_products": len(pipeline.naver_df),
            "unique_categories": pipeline.naver_df["category_id"].nunique()
            if "category_id" in pipeline.naver_df.columns else 0,
            "unique_styles": pipeline.naver_df["style_id"].nunique()
            if "style_id" in pipeline.naver_df.columns else 0,
        },
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Fashion Search API...")
    print("API Docs: http://localhost:8001/docs")
    print("API Redoc: http://localhost:8001/redoc")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
