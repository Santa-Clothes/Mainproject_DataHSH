"""
Naver 제품 임베딩 생성 및 Supabase 업데이트
============================================

Supabase에서 Naver 제품 데이터를 로드하고,
FashionCLIP으로 임베딩을 생성한 후 다시 Supabase에 저장
"""

import sys
from pathlib import Path
import argparse
import json
import time
import numpy as np
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.embedding_generator import FashionCLIPEmbeddingGenerator
from utils.supabase_loader import SupabaseLoader
from utils.config import get_system_config


def generate_and_upload_embeddings(
    batch_size: int = 32,
    limit: int = None,
    checkpoint_path: str = None,
    device: str = None,
    dry_run: bool = False,
):
    """
    Naver 제품 임베딩 생성 및 업로드

    Args:
        batch_size: 배치 크기
        limit: 처리할 최대 제품 수 (None이면 전체)
        checkpoint_path: FashionCLIP 체크포인트 경로
        device: 디바이스 (cuda/cpu)
        dry_run: True이면 실제 업로드 안 함 (테스트용)
    """
    print("\n" + "="*80)
    print("Naver Product Embedding Generation")
    print("="*80)
    print(f"Batch size: {batch_size}")
    print(f"Limit: {limit or 'ALL'}")
    print(f"Device: {device or 'auto'}")
    print(f"Dry run: {dry_run}")
    print("="*80)

    # Supabase 로더 초기화
    print("\n[1] Connecting to Supabase...")
    loader = SupabaseLoader()

    # Naver 제품 로드
    print("\n[2] Loading Naver products...")
    df = loader.load_table('naver_products', limit=limit)
    print(f"  [OK] Loaded: {len(df)} products")

    # 임베딩이 없는 제품만 필터
    null_mask = df['embedding'].isna()
    null_count = null_mask.sum()
    print(f"  Missing embeddings: {null_count}")
    print(f"  Has embeddings: {len(df) - null_count}")

    if null_count == 0:
        print("\n[OK] All products already have embeddings!")
        return

    df_todo = df[null_mask].copy()
    print(f"\n  Products to process: {len(df_todo)}")

    # FashionCLIP 모델 로드
    print("\n[3] Loading FashionCLIP model...")
    generator = FashionCLIPEmbeddingGenerator(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    print("  [OK] Model loaded")

    # 임베딩 생성
    print(f"\n[4] Generating embeddings (batch_size={batch_size})...")
    print(f"    Estimated time: ~{len(df_todo) * 0.5 / 60:.1f} minutes")

    image_urls = df_todo['image_url'].tolist()
    product_ids = df_todo['product_id'].tolist()

    all_embeddings = []
    failed_indices = []
    failed_products = []

    # 배치 단위 처리
    total_batches = (len(image_urls) + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_idx in tqdm(range(total_batches), desc="Generating embeddings"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_urls))

        batch_urls = image_urls[start_idx:end_idx]
        batch_ids = product_ids[start_idx:end_idx]

        try:
            # 배치 임베딩 생성
            embeddings = generator.generate_embeddings_batch(
                image_sources=batch_urls,
                batch_size=len(batch_urls),
                show_progress=False
            )

            all_embeddings.append(embeddings)

        except Exception as e:
            print(f"\n  [ERROR] Batch {batch_idx} failed: {e}")
            # 실패한 배치는 개별 처리
            for i, (url, pid) in enumerate(zip(batch_urls, batch_ids)):
                try:
                    emb = generator.generate_embedding(url)
                    all_embeddings.append(emb.reshape(1, -1))
                except Exception as e2:
                    idx = start_idx + i
                    failed_indices.append(idx)
                    failed_products.append(pid)
                    all_embeddings.append(np.zeros((1, 768)))  # 실패 시 0 벡터
                    if len(failed_indices) <= 10:
                        print(f"  [ERROR] {pid}: {str(e2)[:100]}")

    elapsed_time = time.time() - start_time

    # 전체 임베딩 합치기
    all_embeddings = np.vstack(all_embeddings)
    print(f"\n  [OK] Embedding generation complete: {all_embeddings.shape}")
    print(f"  Time: {elapsed_time/60:.1f} minutes")

    if failed_indices:
        print(f"  [WARNING] {len(failed_indices)} images failed")
        if len(failed_products) <= 20:
            print(f"  Failed IDs: {failed_products}")

    # Supabase 업데이트
    if dry_run:
        print("\n[5] Dry run mode - skipping upload")
        print(f"  Generated embeddings: {len(all_embeddings)}")
        return all_embeddings, product_ids

    print(f"\n[5] Uploading to Supabase (BULK INSERT)...")
    print(f"    Total records: {len(product_ids)}")
    print(f"    This may take a while...")

    success_count = 0
    error_count = 0
    upload_start = time.time()

    # BULK 업로드 (한 번에 100개씩)
    upload_batch_size = 100
    total_upload_batches = (len(product_ids) + upload_batch_size - 1) // upload_batch_size

    for batch_idx in tqdm(range(total_upload_batches), desc="Bulk uploading"):
        start_idx = batch_idx * upload_batch_size
        end_idx = min(start_idx + upload_batch_size, len(product_ids))

        try:
            # 개별 update (embedding 컬럼만 업데이트)
            for i in range(start_idx, end_idx):
                product_id = product_ids[i]
                embedding_list = all_embeddings[i].tolist()

                loader.client.table('naver_products').update(
                    {'embedding': embedding_list}
                ).eq('product_id', product_id).execute()

                success_count += 1

            # Rate limit 방지
            if batch_idx < total_upload_batches - 1:
                time.sleep(0.1)

        except Exception as e:
            error_count += (end_idx - start_idx)
            print(f"\n  [ERROR] Batch {batch_idx} ({start_idx}-{end_idx}): {e}")

    upload_time = time.time() - upload_start

    print(f"\n  [OK] Upload complete")
    print(f"    Success: {success_count}")
    print(f"    Failed: {error_count}")
    print(f"    Time: {upload_time/60:.1f} minutes")

    # 다음 단계 안내
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Build FAISS index:")
    print("     python scripts/build_supabase_faiss_index.py")
    print("  2. Set USE_FAISS=true in .env")
    print("  3. Restart API")
    print("="*80)

    return all_embeddings, product_ids


def main():
    parser = argparse.ArgumentParser(description="Generate Naver embeddings and upload to Supabase")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of products to process (for testing)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (default: from .env)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device (default: auto-detect)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode - don't upload to Supabase"
    )

    args = parser.parse_args()

    # 설정 로드
    config = get_system_config()
    checkpoint_path = args.checkpoint or config.checkpoint_path
    device = args.device or config.device

    generate_and_upload_embeddings(
        batch_size=args.batch_size,
        limit=args.limit,
        checkpoint_path=checkpoint_path,
        device=device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
