"""
Fashion Search API
===================

Nine Oz → K-Fashion → Naver Shopping 검색 시스템
FastAPI 기반 REST API
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.search_pipeline import SearchPipeline

# FastAPI 앱 생성
app = FastAPI(
    title="Fashion Search API",
    description="Nine Oz → K-Fashion → Naver Shopping 검색 시스템",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    print("Initializing search pipeline...")
    pipeline = SearchPipeline(
        nineoz_csv_path="c:/Work/hwangseonghun/nineoz_with_kfashion_categories.csv",
        naver_csv_path="c:/Work/hwangseonghun/naver_with_kfashion_categories.csv",
        model=None,  # 나중에 모델 로드
    )
    print("Pipeline initialized successfully!")


@app.get("/", response_model=Dict)
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Fashion Search API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "query_item": "/query/{index}",
            "categories": "/categories",
            "stats": "/stats",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return HealthResponse(
        status="healthy",
        model_loaded=pipeline.model is not None,
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
