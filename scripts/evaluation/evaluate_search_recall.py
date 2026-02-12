"""
Search Recall 성능 측정 스크립트

나인오즈 쿼리 이미지 → 네이버 유사 제품 검색의 Recall@K를 측정합니다.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from api.search_pipeline import SearchPipeline


def load_test_queries(test_file: str) -> List[Dict]:
    """
    테스트 쿼리 데이터 로드

    Format:
    {
        "queries": [
            {
                "query_id": "9oz_001",
                "product_id": "1234",  # 나인오즈 제품 ID
                "relevant_product_ids": ["naver_123", "naver_456", ...]  # Ground truth
            }
        ]
    }
    """
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['queries']


def evaluate_recall_at_k(
    pipeline: SearchPipeline,
    test_queries: List[Dict],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[int, float]:
    """
    Recall@K 계산

    Args:
        pipeline: 검색 파이프라인
        test_queries: 테스트 쿼리 리스트
        k_values: 측정할 K 값들

    Returns:
        {K: Recall@K} 딕셔너리
    """
    recall_scores = {k: [] for k in k_values}

    print("\n" + "="*80)
    print("Evaluating Recall@K")
    print("="*80)
    print(f"Total queries: {len(test_queries)}")
    print(f"K values: {k_values}")
    print("="*80 + "\n")

    for query in tqdm(test_queries, desc="Evaluating queries"):
        query_id = query['query_id']
        product_id = query['product_id']
        relevant_ids = set(query['relevant_product_ids'])

        # 나인오즈 제품 찾기
        nineoz_product = pipeline.nineoz_df[
            pipeline.nineoz_df['product_id'] == product_id
        ]

        if len(nineoz_product) == 0:
            print(f"[WARNING] Query {query_id}: Product {product_id} not found")
            continue

        # 이미지 URL 가져오기
        image_url = nineoz_product.iloc[0]['image_url']

        # 검색 수행
        try:
            result = pipeline.search_by_url(
                image_url=image_url,
                top_k=max(k_values),  # 최대 K만큼 검색
                filter_params={}
            )

            # 검색 결과 ID 추출
            result_ids = [r['product_id'] for r in result['results']]

            # 각 K에 대해 Recall 계산
            for k in k_values:
                top_k_ids = set(result_ids[:k])
                hits = len(top_k_ids & relevant_ids)
                recall = hits / len(relevant_ids) if len(relevant_ids) > 0 else 0
                recall_scores[k].append(recall)

        except Exception as e:
            print(f"[ERROR] Query {query_id}: {e}")
            # 실패 시 0점 처리
            for k in k_values:
                recall_scores[k].append(0.0)

    # 평균 Recall 계산
    avg_recall = {k: sum(scores) / len(scores) if scores else 0
                  for k, scores in recall_scores.items()}

    return avg_recall


def evaluate_precision_at_k(
    pipeline: SearchPipeline,
    test_queries: List[Dict],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[int, float]:
    """
    Precision@K 계산
    """
    precision_scores = {k: [] for k in k_values}

    for query in tqdm(test_queries, desc="Evaluating precision"):
        query_id = query['query_id']
        product_id = query['product_id']
        relevant_ids = set(query['relevant_product_ids'])

        # 나인오즈 제품 찾기
        nineoz_product = pipeline.nineoz_df[
            pipeline.nineoz_df['product_id'] == product_id
        ]

        if len(nineoz_product) == 0:
            continue

        image_url = nineoz_product.iloc[0]['image_url']

        try:
            result = pipeline.search_by_url(
                image_url=image_url,
                top_k=max(k_values),
                filter_params={}
            )

            result_ids = [r['product_id'] for r in result['results']]

            for k in k_values:
                top_k_ids = set(result_ids[:k])
                hits = len(top_k_ids & relevant_ids)
                precision = hits / k if k > 0 else 0
                precision_scores[k].append(precision)

        except Exception as e:
            for k in k_values:
                precision_scores[k].append(0.0)

    avg_precision = {k: sum(scores) / len(scores) if scores else 0
                     for k, scores in precision_scores.items()}

    return avg_precision


def print_evaluation_results(
    recall: Dict[int, float],
    precision: Dict[int, float],
    num_queries: int
):
    """평가 결과 출력"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total queries evaluated: {num_queries}")
    print("\n" + "-"*80)
    print(f"{'Metric':<20} {'@1':<10} {'@5':<10} {'@10':<10} {'@20':<10}")
    print("-"*80)

    # Recall
    recall_line = f"{'Recall':<20}"
    for k in [1, 5, 10, 20]:
        if k in recall:
            recall_line += f"{recall[k]*100:>9.2f}% "
        else:
            recall_line += f"{'N/A':<10}"
    print(recall_line)

    # Precision
    precision_line = f"{'Precision':<20}"
    for k in [1, 5, 10, 20]:
        if k in precision:
            precision_line += f"{precision[k]*100:>9.2f}% "
        else:
            precision_line += f"{'N/A':<10}"
    print(precision_line)

    print("="*80 + "\n")

    # 상업용 비교
    print("📊 Commercial Benchmark Comparison:")
    print("-"*80)
    if 10 in recall:
        recall_10 = recall[10] * 100
        print(f"Your Model (Recall@10):     {recall_10:.1f}%")
        print(f"ASOS (estimated):            ~75-78%")
        print(f"Musinsa (estimated):         ~75-82%")
        print(f"Naver Shopping (estimated):  ~62-68%")
        print()

        if recall_10 >= 75:
            print("✨ EXCELLENT! 상업용 수준 달성! (무신사급)")
        elif recall_10 >= 70:
            print("🎯 GOOD! 상업 서비스 가능 수준")
        elif recall_10 >= 65:
            print("👍 Not bad. 네이버 쇼핑 수준")
        else:
            print("⚠️ 개선 필요. Fine-tuning 강화 권장")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Search Recall Evaluation")
    parser.add_argument(
        "--test-file",
        type=str,
        default="tests/test_data/search_test_queries.json",
        help="Test queries JSON file"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values for Recall@K and Precision@K"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/multi_domain/best_model.pt",
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--faiss-index",
        type=str,
        default="data/indexes/naver.index",
        help="FAISS index path"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Fashion Search - Recall Evaluation")
    print("="*80)
    print(f"Test file: {args.test_file}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"FAISS index: {args.faiss_index}")
    print("="*80 + "\n")

    # 테스트 쿼리 로드
    test_queries = load_test_queries(args.test_file)
    print(f"[OK] Loaded {len(test_queries)} test queries\n")

    # 파이프라인 초기화
    print("Initializing search pipeline...")
    pipeline = SearchPipeline(
        nineoz_csv_path="data/csv/internal_products_rows.csv",
        naver_csv_path="data/csv/naver_products_rows.csv",
        checkpoint_path=args.checkpoint,
        faiss_index_path=args.faiss_index,
        use_faiss=True,
        device="cpu"  # GPU 사용 시 "cuda"
    )
    print("[OK] Pipeline initialized\n")

    # Recall 평가
    recall = evaluate_recall_at_k(pipeline, test_queries, args.k_values)

    # Precision 평가
    precision = evaluate_precision_at_k(pipeline, test_queries, args.k_values)

    # 결과 출력
    print_evaluation_results(recall, precision, len(test_queries))

    # 결과 저장
    output_file = "results/evaluation/search_recall_results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    results = {
        "num_queries": len(test_queries),
        "recall": recall,
        "precision": precision,
        "k_values": args.k_values,
        "checkpoint": args.checkpoint
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
