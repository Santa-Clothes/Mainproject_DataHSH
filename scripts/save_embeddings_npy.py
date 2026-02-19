"""
Naver 제품 임베딩 생성 → 로컬 .npz 저장
==========================================

1단계: Supabase에서 임베딩이 없는 제품 목록만 가져와서
        FashionCLIP으로 임베딩 생성 후 로컬 .npz에 저장.
        중간에 죽어도 checkpoint.json으로 이어서 재개 가능.

다음 단계: python scripts/upload_embeddings_pg.py
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.embedding_generator import FashionCLIPEmbeddingGenerator
from utils.supabase_loader import SupabaseLoader
from utils.config import get_system_config

DEFAULT_NPZ_PATH = "data/embeddings/naver_embeddings.npz"
DEFAULT_CHECKPOINT_PATH = "data/embeddings/checkpoint.json"


def load_checkpoint(checkpoint_path: str) -> set:
    """이미 처리 완료된 product_id 집합 반환"""
    p = Path(checkpoint_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = set(data.get("completed_ids", []))
        print(f"  [Checkpoint] 이전 진행 상황 로드: {len(ids)}개 완료됨")
        return ids
    return set()


def save_checkpoint(checkpoint_path: str, completed_ids: list):
    """진행 상황 저장"""
    p = Path(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(
            {"completed_ids": completed_ids, "count": len(completed_ids)},
            f,
            ensure_ascii=False,
        )


def merge_and_save_npz(npz_path: str, new_ids: list, new_embeddings: np.ndarray):
    """기존 .npz에 새 데이터를 병합하여 저장"""
    p = Path(npz_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        existing = np.load(p, allow_pickle=True)
        all_ids = list(existing["product_ids"]) + new_ids
        all_embeddings = np.vstack([existing["embeddings"], new_embeddings])
    else:
        all_ids = new_ids
        all_embeddings = new_embeddings

    np.savez_compressed(
        npz_path,
        embeddings=all_embeddings,
        product_ids=np.array(all_ids),
    )
    return len(all_ids)


def generate_and_save(
    npz_path: str = DEFAULT_NPZ_PATH,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    batch_size: int = 32,
    save_every: int = 200,
    limit: int = None,
    checkpoint_path_arg: str = None,
    device: str = None,
):
    print("\n" + "=" * 80)
    print("Step 1: 임베딩 생성 → 로컬 .npz 저장")
    print("=" * 80)

    # 1. 이미 처리된 ID 로드
    done_ids = load_checkpoint(checkpoint_path)

    # 2. Supabase에서 임베딩 없는 제품 조회
    print("\n[1] Supabase 연결 및 제품 로드...")
    loader = SupabaseLoader()

    all_rows = []
    batch = 500
    offset = 0
    while True:
        resp = (
            loader.client.table("naver_products")
            .select("product_id, image_url")
            .is_("embedding", "null")
            .range(offset, offset + batch - 1)
            .execute()
        )
        if not resp.data:
            break
        all_rows.extend(resp.data)
        if len(resp.data) < batch:
            break
        offset += batch

    print(f"  임베딩 없는 제품: {len(all_rows)}개")

    # 체크포인트로 이미 처리된 것 제외
    todo = [r for r in all_rows if str(r["product_id"]) not in done_ids]
    if limit:
        todo = todo[:limit]
    print(f"  이번에 처리할 제품: {len(todo)}개")

    if not todo:
        print("\n[OK] 처리할 제품이 없습니다!")
        return

    # 3. FashionCLIP 모델 로드
    print("\n[2] FashionCLIP 모델 로드...")
    config = get_system_config()
    generator = FashionCLIPEmbeddingGenerator(
        checkpoint_path=checkpoint_path_arg or config.checkpoint_path,
        device=device or config.device,
    )
    print("  [OK] 모델 로드 완료")

    # 4. 배치 단위 임베딩 생성 + 주기적 저장
    print(f"\n[3] 임베딩 생성 (batch_size={batch_size}, save_every={save_every})...")
    print(f"  총 {len(todo)}개 / checkpoint마다 {save_every}개씩 저장")

    image_urls = [r["image_url"] for r in todo]
    product_ids = [str(r["product_id"]) for r in todo]

    buffer_ids = []
    buffer_embeddings = []
    total_saved = 0
    total_batches = (len(image_urls) + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_idx in tqdm(range(total_batches), desc="Generating"):
        s = batch_idx * batch_size
        e = min(s + batch_size, len(image_urls))
        batch_urls = image_urls[s:e]
        batch_ids = product_ids[s:e]

        try:
            embeddings = generator.generate_embeddings_batch(
                image_sources=batch_urls,
                batch_size=len(batch_urls),
                show_progress=False,
            )
            buffer_embeddings.append(embeddings)
            buffer_ids.extend(batch_ids)

        except Exception as exc:
            # 배치 실패 시 개별 처리
            for url, pid in zip(batch_urls, batch_ids):
                try:
                    emb = generator.generate_embedding(url).reshape(1, -1)
                    buffer_embeddings.append(emb)
                    buffer_ids.append(pid)
                except Exception:
                    # 완전히 실패한 이미지는 건너뜀
                    pass

        # save_every 간격으로 .npz에 flush
        if len(buffer_ids) >= save_every:
            merged_emb = np.vstack(buffer_embeddings)
            total_count = merge_and_save_npz(npz_path, buffer_ids, merged_emb)
            done_ids.update(buffer_ids)
            save_checkpoint(checkpoint_path, list(done_ids))
            total_saved += len(buffer_ids)
            tqdm.write(
                f"  [Saved] {total_saved}개 누적 저장 (npz 총 {total_count}개)"
            )
            buffer_ids = []
            buffer_embeddings = []

    # 남은 버퍼 저장
    if buffer_ids:
        merged_emb = np.vstack(buffer_embeddings)
        total_count = merge_and_save_npz(npz_path, buffer_ids, merged_emb)
        done_ids.update(buffer_ids)
        save_checkpoint(checkpoint_path, list(done_ids))
        total_saved += len(buffer_ids)

    elapsed = time.time() - start_time
    print(f"\n[OK] 완료!")
    print(f"  저장된 임베딩: {total_saved}개")
    print(f"  파일: {npz_path}")
    print(f"  소요 시간: {elapsed/60:.1f}분")
    print(f"\n다음 단계:")
    print(f"  python scripts/upload_embeddings_pg.py --npz {npz_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="임베딩 생성 → 로컬 .npz 저장")
    parser.add_argument("--npz", default=DEFAULT_NPZ_PATH, help=".npz 저장 경로")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="체크포인트 경로")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=200, help="몇 개마다 .npz에 저장할지")
    parser.add_argument("--limit", type=int, default=None, help="테스트용 처리 개수 제한")
    parser.add_argument("--checkpoint_model", type=str, default=None, help="모델 체크포인트 경로")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    generate_and_save(
        npz_path=args.npz,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        save_every=args.save_every,
        limit=args.limit,
        checkpoint_path_arg=args.checkpoint_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
