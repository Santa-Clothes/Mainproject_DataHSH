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
from io import BytesIO
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

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


# Request/Response 모델
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
    kfashion_category: str
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
            query_embedding=None,  # 모델 없으면 랜덤 사용
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
                kfashion_category=item["kfashion_category"],
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
        pipeline.nineoz_df["kfashion_item_category"].value_counts().to_dict()
    )

    # 네이버 카테고리 통계
    naver_categories = (
        pipeline.naver_df["kfashion_item_category"].value_counts().to_dict()
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
            "unique_categories": pipeline.nineoz_df["kfashion_item_category"]
            .nunique(),
            "unique_colors": pipeline.nineoz_df["칼라명"].nunique()
            if "칼라명" in pipeline.nineoz_df.columns
            else 0,
        },
        "naver": {
            "total_products": len(pipeline.naver_df),
            "unique_categories": pipeline.naver_df["kfashion_item_category"]
            .nunique(),
        },
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Fashion Search API...")
    print("API Docs: http://localhost:8001/docs")
    print("API Redoc: http://localhost:8001/redoc")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
