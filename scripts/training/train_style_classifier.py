"""
FashionCLIP 스타일 분류기 학습
================================

FashionCLIP Vision Encoder (고정) + Linear(512 → 23) 학습.
K-Fashion 데이터로 fine-tuning.

실행:
    python scripts/train_style_classifier.py
    python scripts/train_style_classifier.py --samples 300 --epochs 50

결과:
    checkpoints/style_classifier.pt  ← 학습된 분류기
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

STYLES = [
    "레트로", "로맨틱", "리조트", "매니시", "모던",
    "밀리터리", "섹시", "소피스트케이티드", "스트리트", "스포티",
    "아방가르드", "오리엔탈", "웨스턴", "젠더리스", "컨트리",
    "클래식", "키치", "톰보이", "펑크", "페미닌",
    "프레피", "히피", "힙합",
]
STYLE_TO_IDX = {s: i for i, s in enumerate(STYLES)}


def load_fashionclip(device: str):
    """FashionCLIP 로드 (파라미터 전체 고정)"""
    print("[1/4] FashionCLIP 로딩...", flush=True)
    try:
        model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        print("  [OK] FashionCLIP 로드 완료")
    except Exception as e:
        print(f"  [경고] FashionCLIP 실패 ({e}), 기본 CLIP 사용")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, processor


def extract_embeddings(model, processor, data_root: Path, styles: list,
                       n_samples: int, device: str, batch_size: int = 32):
    """
    K-Fashion 이미지 → FashionCLIP 임베딩 추출 (배치 처리)
    반환: embeddings [N, 512], labels [N]
    """
    all_embeddings = []
    all_labels = []

    for style in styles:
        style_dir = data_root / style
        if not style_dir.exists():
            print(f"  [경고] 폴더 없음: {style}")
            continue

        imgs = sorted(style_dir.glob("*.jpg"))
        if len(imgs) == 0:
            imgs = sorted(style_dir.glob("*.png"))

        selected = random.sample(imgs, min(n_samples, len(imgs)))
        label_idx = STYLE_TO_IDX[style]

        # 배치 단위 처리
        for i in range(0, len(selected), batch_size):
            batch_paths = selected[i:i + batch_size]
            batch_images = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                except Exception:
                    continue

            if not batch_images:
                continue

            inputs = processor(images=batch_images, return_tensors="pt",
                               padding=True).to(device)
            with torch.no_grad():
                vision_out = model.vision_model(**inputs)
                emb = vision_out.pooler_output
                emb = model.visual_projection(emb)
                emb = F.normalize(emb, p=2, dim=-1)

            all_embeddings.append(emb.cpu())
            all_labels.extend([label_idx] * len(batch_images))

        count = min(n_samples, len(imgs))
        print(f"  {style:<16} {count}장 완료", flush=True)

    embeddings = torch.cat(all_embeddings, dim=0)       # [N, 512]
    labels = torch.tensor(all_labels, dtype=torch.long) # [N]
    return embeddings, labels


def train_classifier(train_embeddings, train_labels, val_embeddings, val_labels,
                     num_classes: int, epochs: int, lr: float, device: str):
    """MLP 분류기 학습 (Linear → ReLU → Dropout → Linear)"""
    emb_dim = train_embeddings.shape[1]
    classifier = nn.Sequential(
        nn.Linear(emb_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    ).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_embeddings.to(device),
                                  train_labels.to(device))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    best_val_acc = 0.0
    best_state = None

    print(f"\n[3/4] MLP 분류기 학습 ({epochs} epochs)...", flush=True)
    for epoch in range(epochs):
        # Train
        classifier.train()
        total_loss = 0.0
        for emb_batch, label_batch in train_loader:
            optimizer.zero_grad()
            logits = classifier(emb_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validate
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_embeddings.to(device))
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels.to(device)).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1:>3}/{epochs}  loss={avg_loss:.4f}  "
                  f"val_acc={val_acc*100:.1f}%  best={best_val_acc*100:.1f}%",
                  flush=True)

    # 최고 가중치 복원
    classifier.load_state_dict(best_state)
    return classifier, best_val_acc


def evaluate_per_style(classifier, embeddings, labels, styles, device):
    """스타일별 정확도 계산"""
    classifier.eval()
    with torch.no_grad():
        logits = classifier(embeddings.to(device))
        preds = logits.argmax(dim=1).cpu()
        top3_preds = logits.topk(3, dim=1).indices.cpu()

    labels = labels.cpu()
    results = {}
    for i, style in enumerate(styles):
        mask = labels == i
        if mask.sum() == 0:
            continue
        style_preds = preds[mask]
        style_top3 = top3_preds[mask]
        top1_acc = (style_preds == i).float().mean().item()
        top3_acc = (style_top3 == i).any(dim=1).float().mean().item()
        results[style] = {"top1": top1_acc, "top3": top3_acc, "n": mask.sum().item()}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", default=r"C:\K-fashion\Training\원천데이터")
    parser.add_argument("--val_root",   default=r"C:\K-fashion\Validation\원천데이터")
    parser.add_argument("--samples",    type=int, default=500,
                        help="스타일당 학습 샘플 수 (기본: 500)")
    parser.add_argument("--val_samples", type=int, default=100,
                        help="스타일당 검증 샘플 수 (기본: 100)")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("FashionCLIP 스타일 분류기 학습")
    print("=" * 60)
    print(f"device     : {device}")
    print(f"학습 샘플  : 스타일당 {args.samples}장")
    print(f"검증 샘플  : 스타일당 {args.val_samples}장")
    print(f"epochs     : {args.epochs}")
    print("=" * 60)

    # 1. 모델 로드
    model, processor = load_fashionclip(device)

    # 2. 임베딩 추출
    train_root = Path(args.train_root)
    val_root   = Path(args.val_root)
    styles = [s for s in STYLES if (train_root / s).exists()]

    print(f"\n[2/4] 학습 임베딩 추출 ({len(styles)}개 스타일 × ~{args.samples}장)...")
    t0 = time.time()
    train_emb, train_labels = extract_embeddings(
        model, processor, train_root, styles, args.samples, device
    )
    print(f"  → 총 {len(train_labels)}장 완료 ({time.time()-t0:.0f}초)")

    print(f"\n  검증 임베딩 추출 ({len(styles)}개 스타일 × ~{args.val_samples}장)...")
    val_emb, val_labels = extract_embeddings(
        model, processor, val_root, styles, args.val_samples, device
    )
    print(f"  → 총 {len(val_labels)}장 완료")

    # 3. 학습
    classifier, best_val_acc = train_classifier(
        train_emb, train_labels, val_emb, val_labels,
        num_classes=len(styles), epochs=args.epochs, lr=args.lr, device=device
    )

    # 4. 최종 평가
    print("\n[4/4] 최종 평가...")
    per_style = evaluate_per_style(classifier, val_emb, val_labels, styles, device)

    print("\n" + "=" * 60)
    print("스타일별 Top-1 / Top-3 정확도")
    print("=" * 60)
    top1_list, top3_list = [], []
    for style, r in per_style.items():
        bar = "#" * int(r["top1"] * 20)
        print(f"  {style:<16} Top1={r['top1']*100:5.1f}%  Top3={r['top3']*100:5.1f}%  {bar}")
        top1_list.append(r["top1"])
        top3_list.append(r["top3"])

    overall_top1 = sum(top1_list) / len(top1_list)
    overall_top3 = sum(top3_list) / len(top3_list)

    print("=" * 60)
    print(f"  전체 Top-1: {overall_top1*100:.1f}%")
    print(f"  전체 Top-3: {overall_top3*100:.1f}%")
    print(f"  (best val during training: {best_val_acc*100:.1f}%)")
    print("=" * 60)

    # 저장
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "style_classifier.pt"
    torch.save({
        "classifier_state_dict": classifier.state_dict(),
        "styles": styles,
        "style_to_idx": STYLE_TO_IDX,
        "emb_dim": train_emb.shape[1],
        "num_classes": len(styles),
        "val_top1": overall_top1,
        "val_top3": overall_top3,
        "args": vars(args),
    }, save_path)
    print(f"\n[저장] {save_path}")
    print("\n[발표용 요약]")
    print(f"  FashionCLIP + Linear 분류기 (K-Fashion {len(styles)}개 스타일)")
    print(f"  학습: 스타일당 {args.samples}장 × {len(styles)} = {len(train_labels)}장")
    print(f"  Top-1 Accuracy: {overall_top1*100:.1f}%")
    print(f"  Top-3 Accuracy: {overall_top3*100:.1f}%")


if __name__ == "__main__":
    main()
