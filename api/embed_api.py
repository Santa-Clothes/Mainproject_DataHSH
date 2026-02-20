"""
Fashion Embedding API (마이크로서비스)
=====================================

이미지 → 768차원 벡터 변환만 담당.
DB 접근 없음. 검색 로직 없음.
Spring 백엔드에서 호출하는 AI 전용 서버.
"""

import sys
from pathlib import Path
from io import BytesIO
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.embedding_generator import FashionCLIPEmbeddingGenerator

app = FastAPI(
    title="Fashion Embedding API",
    description="FashionCLIP 이미지 임베딩 생성 서버",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델은 서버 시작 시 한 번만 로드
generator: FashionCLIPEmbeddingGenerator = None


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int


@app.on_event("startup")
async def startup():
    global generator
    print("\n[Embed API] Loading FashionCLIP model...")
    generator = FashionCLIPEmbeddingGenerator()
    print("[Embed API] Model ready.")


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": generator is not None}


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_image(file: UploadFile = File(...)):
    """
    이미지 파일을 받아 768차원 임베딩 벡터 반환.

    Spring에서 호출:
        POST /embed
        Content-Type: multipart/form-data
        Body: file=<이미지>

    Returns:
        { "embedding": [0.12, -0.34, ...], "dimension": 768 }
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"이미지 파일만 허용: {file.content_type}")

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    embedding = generator.generate_embedding(image, normalize=True)

    return EmbeddingResponse(
        embedding=embedding.tolist(),
        dimension=len(embedding),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
