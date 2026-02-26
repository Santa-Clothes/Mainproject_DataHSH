"""
스타일 라벨 CSV 생성 스크립트
==============================

nineounce_product_vectors.csv, naver_product_vectors_768_rows.csv 에 저장된
768차원 CLIPVisionModel pooler_output 벡터를 읽어,
CLIP visual_projection(768→512) → style_classifier(512→23) 파이프라인을 돌린 뒤
23개 세부 스타일을 10개 대분류로 집계하여 Top-3 결과를 CSV로 저장합니다.

출력:
  data/style_labels/nineounce_style_labels.csv
  data/style_labels/naver_style_labels.csv

컬럼:
  product_id, style_top1, style_score1, style_top2, style_score2, style_top3, style_score3
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── 상수 ─────────────────────────────────────────────────────────────────────
STYLES = [
    "레트로", "로맨틱", "리조트", "매니시", "모던",
    "밀리터리", "섹시", "소피스트케이티드", "스트리트", "스포티",
    "아방가르드", "오리엔탈", "웨스턴", "젠더리스", "컨트리",
    "클래식", "키치", "톰보이", "펑크", "페미닌",
    "프레피", "히피", "힙합",
]

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

CLASSIFIER_PATH = ROOT / "checkpoints" / "style_classifier.pt"
OUTPUT_DIR = ROOT / "data" / "style_labels"

NINEOUNCE_VECTORS = ROOT / "nineounce_product_vectors.csv"
NAVER_VECTORS     = ROOT / "naver_product_vectors_768_rows.csv"


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_models(device: str):
    print("[1/2] FashionCLIP (visual_projection 추출용) 로드 중...")
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model = clip_model.to(device)
    print("      [OK]")

    print("[2/2] Style classifier 로드 중...")
    ckpt = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)
    emb_dim    = ckpt["emb_dim"]
    num_classes = ckpt["num_classes"]
    style_labels = ckpt.get("styles", STYLES)

    classifier = nn.Sequential(
        nn.Linear(emb_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    classifier.eval()
    classifier = classifier.to(device)
    print(f"      [OK] emb_dim={emb_dim}, num_classes={num_classes}")

    return clip_model, classifier, style_labels


# ── 임베딩 파싱 ───────────────────────────────────────────────────────────────
def parse_embedding(raw) -> np.ndarray:
    """CSV에 저장된 문자열 임베딩을 numpy 배열로 변환"""
    if isinstance(raw, str):
        return np.array(ast.literal_eval(raw), dtype=np.float32)
    return np.array(raw, dtype=np.float32)


# ── 10개 대분류 집계 ──────────────────────────────────────────────────────────
def aggregate_to_top10(probs: np.ndarray, style_labels: list) -> dict:
    top10 = {s: 0.0 for s in TOP10_STYLES}
    for i, label in enumerate(style_labels):
        parent = STYLE_MAPPING.get(label)
        if parent:
            top10[parent] += float(probs[i])
    total = sum(top10.values())
    if total > 0:
        top10 = {k: v / total for k, v in top10.items()}
    return top10


# ── 단일 CSV 처리 ─────────────────────────────────────────────────────────────
def process_csv(
    csv_path: Path,
    output_path: Path,
    clip_model: CLIPModel,
    classifier: nn.Module,
    style_labels: list,
    device: str,
    batch_size: int = 256,
):
    print(f"\n처리 중: {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"  총 {len(df)}개 상품")

    records = []
    total = len(df)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df.iloc[start:end]

        # 768차원 pooler_output 벡터 파싱
        vecs = np.stack([parse_embedding(row["embedding"]) for _, row in batch.iterrows()])
        tensor = torch.tensor(vecs, dtype=torch.float32).to(device)  # [B, 768]

        with torch.no_grad():
            # 768 → 512 (visual_projection)
            feat = clip_model.visual_projection(tensor)           # [B, 512]
            feat = F.normalize(feat, p=2, dim=-1)

            # 512 → 23 logits → softmax
            logits = classifier(feat)                             # [B, 23]
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()  # [B, 23]

        for idx, (_, row) in enumerate(batch.iterrows()):
            top10 = aggregate_to_top10(probs[idx], style_labels)
            sorted_top10 = sorted(top10.items(), key=lambda x: x[1], reverse=True)

            records.append({
                "product_id":   row["product_id"],
                "style_top1":   sorted_top10[0][0],
                "style_score1": round(sorted_top10[0][1], 4),
                "style_top2":   sorted_top10[1][0],
                "style_score2": round(sorted_top10[1][1], 4),
                "style_top3":   sorted_top10[2][0],
                "style_score3": round(sorted_top10[2][1], 4),
            })

        print(f"  [{end}/{total}] 완료", end="\r")

    print()
    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {output_path}")
    print(f"  스타일 분포 (top1):")
    print(out_df["style_top1"].value_counts().to_string())


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 60)

    clip_model, classifier, style_labels = load_models(device)

    process_csv(
        csv_path    = NINEOUNCE_VECTORS,
        output_path = OUTPUT_DIR / "nineounce_style_labels.csv",
        clip_model  = clip_model,
        classifier  = classifier,
        style_labels= style_labels,
        device      = device,
    )

    process_csv(
        csv_path    = NAVER_VECTORS,
        output_path = OUTPUT_DIR / "naver_style_labels.csv",
        clip_model  = clip_model,
        classifier  = classifier,
        style_labels= style_labels,
        device      = device,
    )

    print("\n" + "=" * 60)
    print("완료! 출력 파일:")
    print(f"  {OUTPUT_DIR / 'nineounce_style_labels.csv'}")
    print(f"  {OUTPUT_DIR / 'naver_style_labels.csv'}")


if __name__ == "__main__":
    main()
