"""
Fashion Embedding API (마이크로서비스)
=====================================

이미지 → 768차원 벡터 변환 + 스타일 분류 담당.
DB 접근 없음. 검색 로직 없음.
Spring 백엔드에서 호출하는 AI 전용 서버.

엔드포인트:
  POST /embed    — 이미지 → 임베딩 벡터 (기존, Spring 검색용)
  POST /analyze  — 이미지 → 임베딩 + 스타일 분류 (신규)
  GET  /health   — 서버 상태
"""

import sys
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.embedding_generator import FashionCLIPEmbeddingGenerator

app = FastAPI(
    title="Fashion Embedding API",
    description="FashionCLIP 이미지 임베딩 생성 + 스타일 분류 서버",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STYLES = [
    "레트로", "로맨틱", "리조트", "매니시", "모던",
    "밀리터리", "섹시", "소피스트케이티드", "스트리트", "스포티",
    "아방가르드", "오리엔탈", "웨스턴", "젠더리스", "컨트리",
    "클래식", "키치", "톰보이", "펑크", "페미닌",
    "프레피", "히피", "힙합",
]

CLASSIFIER_PATH = Path(__file__).parent.parent / "checkpoints" / "style_classifier.pt"

# 서버 시작 시 한 번만 로드
generator: FashionCLIPEmbeddingGenerator = None
clip_model: CLIPModel = None
clip_processor: CLIPProcessor = None
style_classifier: nn.Linear = None
style_labels: List[str] = STYLES


# ---------- Response Models ----------

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int


class StylePrediction(BaseModel):
    style: str
    score: float


class AnalyzeResponse(BaseModel):
    embedding: List[float]
    dimension: int
    styles: List[StylePrediction]  # top-3 스타일 + 점수


# ---------- Startup ----------

@app.on_event("startup")
async def startup():
    global generator, clip_model, clip_processor, style_classifier, style_labels

    print("\n[Embed API] Loading FashionCLIP model...")
    generator = FashionCLIPEmbeddingGenerator()

    # 스타일 분류기용 전체 CLIP 모델 로드
    try:
        clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        print("[Embed API] FashionCLIP (full) loaded for style classifier.")
    except Exception as e:
        print(f"[Embed API] FashionCLIP full load failed: {e}")

    # 학습된 MLP 분류기 로드
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
        print(f"[Embed API] Style classifier loaded. Val Top-1: {acc:.1f}%")
    else:
        print("[Embed API] Style classifier not found — /analyze unavailable.")

    print("[Embed API] Ready.")


# ---------- Endpoints ----------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "style_classifier": style_classifier is not None,
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_image(file: UploadFile = File(...)):
    """
    이미지 → 768차원 임베딩 벡터 반환 (기존 Spring 검색용, 변경 없음)
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


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...), top_k: int = 3):
    """
    이미지 → 임베딩 + 스타일 분류 동시 반환.

    Spring에서 호출:
        POST /analyze
        Content-Type: multipart/form-data
        Body: file=<이미지>

    Returns:
        {
          "embedding": [...],
          "dimension": 768,
          "styles": [
            {"style": "밀리터리", "score": 0.82},
            {"style": "스트리트", "score": 0.11},
            {"style": "스포티",   "score": 0.05}
          ]
        }
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"이미지 파일만 허용: {file.content_type}")

    if style_classifier is None or clip_model is None:
        raise HTTPException(status_code=503, detail="Style classifier not loaded")

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # 1. 임베딩 생성 (기존 방식, 검색용)
    embedding = generator.generate_embedding(image, normalize=True)

    # 2. 스타일 분류 (학습된 Linear 분류기)
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vision_out = clip_model.vision_model(**inputs)
        img_feat = vision_out.pooler_output
        img_feat = clip_model.visual_projection(img_feat)
        img_feat = F.normalize(img_feat, p=2, dim=-1)
        logits = style_classifier(img_feat)[0]         # [num_classes]
        probs = torch.softmax(logits, dim=0)

    top_k = min(top_k, len(style_labels))
    top_indices = probs.topk(top_k).indices.tolist()
    styles = [
        StylePrediction(style=style_labels[i], score=round(probs[i].item(), 4))
        for i in top_indices
    ]

    return AnalyzeResponse(
        embedding=embedding.tolist(),
        dimension=len(embedding),
        styles=styles,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
