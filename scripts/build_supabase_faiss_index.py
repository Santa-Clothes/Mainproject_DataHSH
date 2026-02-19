"""
Supabase 데이터로 FAISS 인덱스 빌드
===================================

Supabase에서 Naver 제품 임베딩을 로드하여 FAISS 인덱스 생성
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.vector_index import FaissVectorIndex
from utils.supabase_loader import SupabaseLoader
from utils.config import get_system_config


def build_faiss_index_from_supabase(
    output_path: str = "data/indexes/naver.index",
):
    """
    Supabase에서 임베딩을 로드하여 FAISS 인덱스 빌드

    Args:
        output_path: 인덱스 저장 경로
    """
    print("\n" + "="*80)
    print("FAISS Index Builder (Supabase)")
    print("="*80)
    print(f"Output: {output_path}")
    print("="*80)

    # Supabase 로더 초기화
    print("\n[1] Connecting to Supabase...")
    loader = SupabaseLoader()

    # Naver 제품 로드 (임베딩이 있는 것만)
    print("\n[2] Loading Naver products from Supabase...")
    print("  Loading products with embeddings only...")

    # 배치로 로드 (타임아웃 방지)
    import pandas as pd
    all_data = []
    batch_size = 500
    offset = 0

    while True:
        try:
            query = loader.client.table('naver_products').select("*").not_.is_('embedding', 'null').range(offset, offset + batch_size - 1)
            response = query.execute()

            if not response.data:
                break

            all_data.extend(response.data)
            print(f"    Loaded: {len(all_data)} products (batch: {len(response.data)})")

            if len(response.data) < batch_size:
                break

            offset += batch_size

        except Exception as e:
            print(f"    Warning: Error at offset {offset}: {e}")
            # 에러 발생 시 다음 배치로
            offset += batch_size
            if offset > 10000:  # 최대 10000개까지만
                break
            continue

    df = pd.DataFrame(all_data)
    print(f"  [OK] Loaded: {len(df)} products with embeddings")

    # 임베딩 파싱
    print("\n[3] Parsing embeddings...")
    embeddings_list = []
    product_ids = []
    valid_count = 0
    null_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing embeddings"):
        product_id = row['product_id']
        embedding = row['embedding']

        # Null 체크
        if embedding is None or (isinstance(embedding, float) and np.isnan(embedding)):
            null_count += 1
            continue

        try:
            # 타입 변환
            if isinstance(embedding, str):
                embedding = json.loads(embedding)

            embedding_array = np.array(embedding, dtype=np.float32)

            # 차원 체크
            if embedding_array.shape[0] != 768:
                error_count += 1
                continue

            # NaN/Inf 체크
            if np.isnan(embedding_array).any() or np.isinf(embedding_array).any():
                error_count += 1
                continue

            embeddings_list.append(embedding_array)
            product_ids.append(str(product_id))
            valid_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\n  [ERROR] {product_id}: {e}")

    print(f"\n  [OK] Parsing complete")
    print(f"    Valid: {valid_count}")
    print(f"    NULL: {null_count}")
    print(f"    Errors: {error_count}")

    if valid_count == 0:
        print("\n[ERROR] No valid embeddings found!")
        print("Please run: python scripts/generate_naver_embeddings.py")
        return None

    # numpy 배열로 변환
    embeddings = np.vstack(embeddings_list)
    print(f"\n  Embeddings shape: {embeddings.shape}")

    # FAISS 인덱스 빌드
    print(f"\n[4] Building FAISS index...")
    index = FaissVectorIndex(embedding_dim=768)
    index.build_index(
        embeddings=embeddings,
        product_ids=product_ids,
        normalize=False  # 이미 정규화되어 있음
    )

    # 저장
    print(f"\n[5] Saving index...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(output_path))

    print(f"\n" + "="*80)
    print("Index built successfully!")
    print("="*80)
    print(f"\n[Index Stats]")
    stats = index.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\n[Next Steps]")
    print(f"  1. Set USE_FAISS=true in .env")
    print(f"  2. Set FAISS_INDEX_PATH={output_path}")
    print(f"  3. Restart API server")
    print("="*80)

    return index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from Supabase")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output index path (default: from .env)"
    )

    args = parser.parse_args()

    # 설정 로드
    config = get_system_config()
    output_path = args.output or config.faiss_index_path

    build_faiss_index_from_supabase(output_path=output_path)


if __name__ == "__main__":
    main()
