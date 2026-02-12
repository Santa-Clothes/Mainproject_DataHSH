"""
테스트 데이터셋 생성 도구

나인오즈 제품에 대한 검색 결과를 보여주고,
사용자가 relevant/not relevant를 라벨링하여 ground truth를 만듭니다.
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List
import argparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from api.search_pipeline import SearchPipeline


def select_sample_queries(
    nineoz_df: pd.DataFrame,
    num_samples: int = 10,
    random_seed: int = 42
) -> List[Dict]:
    """
    나인오즈 제품에서 샘플 쿼리 선택

    Args:
        nineoz_df: 나인오즈 제품 DataFrame
        num_samples: 샘플 개수
        random_seed: 랜덤 시드

    Returns:
        샘플 쿼리 리스트
    """
    random.seed(random_seed)

    # 이미지가 있는 제품만 선택
    valid_products = nineoz_df[nineoz_df['image_url'].notna()].copy()

    # 랜덤 샘플링
    if len(valid_products) < num_samples:
        print(f"[WARNING] Only {len(valid_products)} products available")
        num_samples = len(valid_products)

    sampled = valid_products.sample(n=num_samples, random_state=random_seed)

    queries = []
    for idx, row in sampled.iterrows():
        queries.append({
            'query_id': f"9oz_{str(row['product_id']).zfill(4)}",
            'product_id': row['product_id'],
            'product_name': row['product_name'],
            'image_url': row['image_url'],
            'kfashion_item_category': row.get('kfashion_item_category', 'unknown')
        })

    return queries


def generate_search_results_for_labeling(
    pipeline: SearchPipeline,
    queries: List[Dict],
    top_k: int = 20
) -> Dict:
    """
    각 쿼리에 대한 검색 결과 생성 (라벨링용)

    Args:
        pipeline: 검색 파이프라인
        queries: 쿼리 리스트
        top_k: 상위 K개 결과

    Returns:
        라벨링용 데이터
    """
    labeling_data = {
        'queries': []
    }

    print("\n" + "="*80)
    print("Generating search results for labeling")
    print("="*80)

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query: {query['query_id']}")
        print(f"  Product: {query['product_name']}")
        print(f"  Category: {query['kfashion_item_category']}")

        try:
            # 검색 수행
            result = pipeline.search_by_url(
                image_url=query['image_url'],
                top_k=top_k,
                filter_params={}
            )

            # 결과 정리
            search_results = []
            for rank, item in enumerate(result['results'], 1):
                search_results.append({
                    'rank': rank,
                    'product_id': item['product_id'],
                    'title': item['title'],
                    'image_url': item['image_url'],
                    'price': item.get('price', 'N/A'),
                    'score': item['score'],
                    'is_relevant': None  # 라벨링할 항목
                })

            labeling_data['queries'].append({
                'query_id': query['query_id'],
                'product_id': query['product_id'],
                'product_name': query['product_name'],
                'query_image_url': query['image_url'],
                'kfashion_item_category': query['kfashion_item_category'],
                'search_results': search_results
            })

            print(f"  ✓ Generated {len(search_results)} results")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    return labeling_data


def interactive_labeling(labeling_data: Dict) -> Dict:
    """
    대화형 라벨링 (터미널 기반)

    Args:
        labeling_data: 라벨링할 데이터

    Returns:
        라벨링된 데이터
    """
    print("\n" + "="*80)
    print("INTERACTIVE LABELING")
    print("="*80)
    print("\nInstructions:")
    print("  - 각 검색 결과를 보고 relevant(1) / not relevant(0) 입력")
    print("  - 'skip'을 입력하면 현재 쿼리 건너뛰기")
    print("  - 'quit'를 입력하면 라벨링 종료")
    print("="*80)

    labeled_data = {'queries': []}

    for query_data in labeling_data['queries']:
        print("\n" + "="*80)
        print(f"Query: {query_data['query_id']}")
        print(f"Product: {query_data['product_name']}")
        print(f"Category: {query_data['kfashion_item_category']}")
        print(f"Query Image: {query_data['query_image_url']}")
        print("="*80)

        relevant_product_ids = []

        for result in query_data['search_results']:
            print(f"\n[Rank {result['rank']}] Score: {result['score']:.4f}")
            print(f"  Title: {result['title']}")
            print(f"  Price: {result['price']}")
            print(f"  Image: {result['image_url']}")

            # 라벨 입력
            while True:
                label_input = input(f"  Relevant? (1=yes, 0=no, skip, quit): ").strip().lower()

                if label_input == 'quit':
                    print("\n[INFO] Labeling stopped by user")
                    return labeled_data

                if label_input == 'skip':
                    print("[INFO] Skipping this query")
                    break

                if label_input in ['0', '1']:
                    is_relevant = label_input == '1'
                    result['is_relevant'] = is_relevant

                    if is_relevant:
                        relevant_product_ids.append(result['product_id'])

                    break
                else:
                    print("  [ERROR] Invalid input. Please enter 1, 0, skip, or quit")

            if label_input == 'skip':
                break

        # 쿼리 데이터 저장
        if relevant_product_ids:  # relevant가 하나라도 있으면 저장
            labeled_data['queries'].append({
                'query_id': query_data['query_id'],
                'product_id': query_data['product_id'],
                'relevant_product_ids': relevant_product_ids
            })
            print(f"\n✓ Labeled: {len(relevant_product_ids)} relevant products")

    return labeled_data


def save_labeling_data(data: Dict, output_file: str):
    """라벨링 데이터 저장"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Labeling data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create Test Dataset for Search Evaluation")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sample queries"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of search results per query"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/test_data/search_labeling_data.json",
        help="Output file for labeling data"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive labeling (terminal)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/multi_domain/best_model.pt",
        help="Model checkpoint path"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Create Test Dataset for Search Evaluation")
    print("="*80)
    print(f"Samples: {args.num_samples}")
    print(f"Top-K results: {args.top_k}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    # 파이프라인 초기화
    print("Initializing search pipeline...")
    pipeline = SearchPipeline(
        nineoz_csv_path="data/csv/internal_products_rows.csv",
        naver_csv_path="data/csv/naver_products_rows.csv",
        checkpoint_path=args.checkpoint,
        faiss_index_path="data/indexes/naver.index",
        use_faiss=True,
        device="cpu"
    )
    print("[OK] Pipeline initialized\n")

    # 샘플 쿼리 선택
    print("Selecting sample queries...")
    queries = select_sample_queries(pipeline.nineoz_df, num_samples=args.num_samples)
    print(f"[OK] Selected {len(queries)} queries\n")

    # 검색 결과 생성
    labeling_data = generate_search_results_for_labeling(pipeline, queries, top_k=args.top_k)

    # 라벨링 데이터 저장 (검색 결과만)
    prelabel_output = args.output.replace('.json', '_prelabel.json')
    save_labeling_data(labeling_data, prelabel_output)

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print(f"1. 라벨링 데이터가 저장되었습니다: {prelabel_output}")
    print(f"2. 이 파일을 열어서 각 검색 결과의 'is_relevant' 필드를 수동으로 편집하세요")
    print(f"   - true: 관련 있는 제품")
    print(f"   - false: 관련 없는 제품")
    print(f"3. 편집 완료 후, convert_labeling_to_test.py로 변환하세요")

    if args.interactive:
        print("\n대화형 라벨링을 시작합니다...")
        labeled_data = interactive_labeling(labeling_data)
        test_output = args.output.replace('_prelabel', '')
        save_labeling_data(labeled_data, test_output)

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
