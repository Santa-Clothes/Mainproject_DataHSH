"""
FashionCLIP Zero-Shot 스타일 분류 검증
========================================

K-Fashion 데이터셋으로 FashionCLIP zero-shot 분류 정확도 측정.
스타일당 N장만 샘플링해서 빠르게 검증.

실행:
    python scripts/evaluation/validate_style_classification.py
    python scripts/evaluation/validate_style_classification.py --samples 50
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# K-Fashion 23개 스타일 (한국어 → 영어 프롬프트)
STYLE_PROMPTS = {
    "레트로":        "retro vintage fashion style clothing",
    "로맨틱":        "romantic feminine lace floral fashion style",
    "리조트":        "resort vacation casual beach fashion style",
    "매니시":        "mannish tailored suit blazer fashion style",
    "모던":          "modern minimalist contemporary fashion style",
    "밀리터리":      "military utility cargo jacket fashion style",
    "섹시":          "sexy glamorous bodycon fashion style",
    "소피스트케이티드": "sophisticated elegant chic fashion style",
    "스트리트":      "street urban hip casual fashion style",
    "스포티":        "sporty athletic activewear fashion style",
    "아방가르드":    "avant-garde experimental artistic fashion style",
    "오리엔탈":      "oriental asian inspired traditional fashion style",
    "웨스턴":        "western cowboy denim boots fashion style",
    "젠더리스":      "genderless gender neutral unisex fashion style",
    "컨트리":        "country rustic folk fashion style",
    "클래식":        "classic timeless traditional fashion style",
    "키치":          "kitsch quirky colorful fun fashion style",
    "톰보이":        "tomboy androgynous casual fashion style",
    "펑크":          "punk rock edgy leather fashion style",
    "페미닌":        "feminine soft pastel dress fashion style",
    "프레피":        "preppy collegiate school fashion style",
    "히피":          "hippie bohemian free-spirited fashion style",
    "힙합":          "hip hop baggy streetwear urban fashion style",
}


def load_model(device: str):
    """FashionCLIP 전체 모델 로드 (vision + text encoder)"""
    print("[모델 로드] FashionCLIP 로딩 중...")
    try:
        model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        print("[OK] FashionCLIP 로드 완료")
    except Exception as e:
        print(f"[경고] FashionCLIP 실패 ({e}), 기본 CLIP 사용")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("[OK] 기본 CLIP 로드 완료")

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, processor


def compute_text_embeddings(model, processor, styles: list, device: str) -> torch.Tensor:
    """23개 스타일 텍스트 임베딩 사전 계산"""
    prompts = [STYLE_PROMPTS[s] for s in styles]

    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        text_out = model.text_model(**inputs)
        text_features = text_out.pooler_output
        text_features = model.text_projection(text_features)
        text_features = F.normalize(text_features, p=2, dim=-1)

    return text_features  # [num_styles, dim]


def sample_images(data_root: Path, styles: list, n_samples: int) -> list:
    """스타일별 N장 샘플링"""
    samples = []
    for style in styles:
        style_dir = data_root / style
        if not style_dir.exists():
            print(f"  [경고] 폴더 없음: {style_dir}")
            continue

        images = sorted(style_dir.glob("*.jpg"))
        if len(images) == 0:
            images = sorted(style_dir.glob("*.png"))

        selected = random.sample(images, min(n_samples, len(images)))
        for img_path in selected:
            samples.append((img_path, style))

    random.shuffle(samples)
    print(f"[샘플링] 총 {len(samples)}장 선택 ({len(styles)}개 스타일 × ~{n_samples}장)")
    return samples


def evaluate(model, processor, samples: list, styles: list,
             text_embeddings: torch.Tensor, device: str):
    """이미지별 스타일 분류 후 정확도 계산"""

    style_to_idx = {s: i for i, s in enumerate(styles)}
    results = {s: {"correct": 0, "total": 0} for s in styles}
    top1_correct = 0
    top3_correct = 0
    total = 0

    print(f"\n[평가 시작] {len(samples)}장 처리 중...")

    for i, (img_path, true_style) in enumerate(samples):
        if i % 50 == 0:
            print(f"  {i}/{len(samples)}...")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # 이미지 임베딩
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = model.vision_model(**inputs)
            image_features = vision_out.pooler_output
            image_features = model.visual_projection(image_features)
            image_features = F.normalize(image_features, p=2, dim=-1)

        # 유사도 계산
        similarities = (image_features @ text_embeddings.T)[0]  # [num_styles]
        top_indices = similarities.argsort(descending=True)

        pred_style = styles[top_indices[0].item()]
        top3_styles = [styles[top_indices[k].item()] for k in range(min(3, len(styles)))]

        # 정확도 집계
        results[true_style]["total"] += 1
        if pred_style == true_style:
            results[true_style]["correct"] += 1
            top1_correct += 1
        if true_style in top3_styles:
            top3_correct += 1
        total += 1

    return results, top1_correct, top3_correct, total


def print_results(results: dict, top1_correct: int, top3_correct: int, total: int):
    """결과 출력"""
    print("\n" + "=" * 60)
    print("FashionCLIP Zero-Shot 스타일 분류 결과")
    print("=" * 60)

    print(f"\n{'스타일':<16} {'정답/전체':>10} {'정확도':>8}")
    print("-" * 40)

    style_accs = []
    for style, r in sorted(results.items()):
        if r["total"] == 0:
            continue
        acc = r["correct"] / r["total"] * 100
        style_accs.append(acc)
        bar = "█" * int(acc / 5)
        print(f"{style:<16} {r['correct']:>4}/{r['total']:<4}  {acc:>5.1f}%  {bar}")

    print("-" * 40)
    print(f"\n[전체 결과]")
    print(f"  Top-1 Accuracy: {top1_correct}/{total} = {top1_correct/total*100:.1f}%")
    print(f"  Top-3 Accuracy: {top3_correct}/{total} = {top3_correct/total*100:.1f}%")
    print(f"  스타일 평균 Accuracy: {np.mean(style_accs):.1f}%")
    print("=" * 60)

    print("\n[발표용 요약]")
    print(f"  FashionCLIP zero-shot 분류 정확도")
    print(f"  - Top-1: {top1_correct/total*100:.1f}%")
    print(f"  - Top-3: {top3_correct/total*100:.1f}%")
    print(f"  - 평가 샘플: {total}장 (K-Fashion {len(results)}개 스타일)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default=r"C:\K-fashion\Training\원천데이터",
                        help="K-Fashion 원천데이터 경로")
    parser.add_argument("--samples", type=int, default=30,
                        help="스타일당 샘플 수 (기본: 30)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[설정] device={device}, 스타일당 샘플={args.samples}장")

    data_root = Path(args.data_root)
    styles = [s for s in STYLE_PROMPTS.keys() if (data_root / s).exists()]
    print(f"[데이터] 발견된 스타일: {len(styles)}개")

    # 모델 로드
    model, processor = load_model(device)

    # 텍스트 임베딩 사전 계산
    print("[텍스트 임베딩] 23개 스타일 프롬프트 인코딩 중...")
    text_embeddings = compute_text_embeddings(model, processor, styles, device)
    print(f"[OK] 텍스트 임베딩 완료: {text_embeddings.shape}")

    # 이미지 샘플링
    samples = sample_images(data_root, styles, args.samples)

    # 평가
    results, top1, top3, total = evaluate(
        model, processor, samples, styles, text_embeddings, device
    )

    # 결과 출력
    print_results(results, top1, top3, total)


if __name__ == "__main__":
    main()
