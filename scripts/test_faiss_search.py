"""
FAISS 검색 테스트
================

FAISS 인덱스가 제대로 작동하는지 테스트
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.search_pipeline import SearchPipeline
from utils.config import get_system_config


def main():
    print("\n" + "="*80)
    print("FAISS Search Performance Test")
    print("="*80)

    # 설정 로드
    config = get_system_config()

    print(f"\n[Configuration]")
    print(f"  Use FAISS: {config.use_faiss}")
    print(f"  FAISS Index: {config.faiss_index_path}")
    print(f"  Naver CSV: {config.naver_csv_path}")
    print(f"  Nine Oz CSV: {config.nineoz_csv_path}")
    print("="*80)

    # 파이프라인 초기화
    print(f"\n[1] Initializing Search Pipeline...")
    start = time.time()

    pipeline = SearchPipeline(
        nineoz_csv_path=config.nineoz_csv_path,
        naver_csv_path=config.naver_csv_path,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
        faiss_index_path=config.faiss_index_path if config.use_faiss else None,
        use_faiss=config.use_faiss,
        precompute_embeddings=False,  # FAISS 사용 시 불필요
    )

    init_time = time.time() - start
    print(f"  [OK] Pipeline initialized in {init_time:.2f}s")

    # 쿼리 아이템 가져오기
    print(f"\n[2] Getting query item...")
    query_item = pipeline.get_query_item(0)
    print(f"  Query: {query_item['product_name']}")
    print(f"  Category: {query_item['kfashion_category']}")

    # 검색 수행
    print(f"\n[3] Performing search...")

    # 이미지 URL 가져오기
    query_image_url = pipeline.nineoz_df.iloc[0]['image_url']

    start = time.time()
    results = pipeline.search_by_image(
        image_source=query_image_url,
        category_filter=query_item['kfashion_category'],
        initial_k=100,
        final_k=10
    )
    search_time = time.time() - start

    print(f"  [OK] Search completed in {search_time:.3f}s")

    # 결과 출력
    print(f"\n[4] Search Results (Top 10):")
    print("-"*80)
    for i, item in enumerate(results, 1):
        print(f"{i:2d}. {item['title'][:50]:50s} | Score: {item['score']:.4f}")
    print("-"*80)

    # 성능 요약
    print(f"\n{'='*80}")
    print(f"Performance Summary")
    print(f"{'='*80}")
    print(f"  Initialization: {init_time:.2f}s")
    print(f"  Search Time: {search_time:.3f}s")
    print(f"  Results: {len(results)} products")
    print(f"  Using: {'FAISS' if config.use_faiss else 'Numpy'}")
    print(f"{'='*80}")

    # 추가 검색 테스트 (FAISS 캐싱 효과 확인)
    if config.use_faiss:
        print(f"\n[5] Additional search test (cache warm)...")
        times = []
        for i in range(3):
            start = time.time()
            _ = pipeline.search_by_image(
                image_source=pipeline.nineoz_df.iloc[i]['image_url'],
                initial_k=100,
                final_k=10
            )
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        print(f"  Average search time (3 queries): {avg_time:.3f}s")
        print(f"  Min: {min(times):.3f}s | Max: {max(times):.3f}s")


if __name__ == "__main__":
    main()
