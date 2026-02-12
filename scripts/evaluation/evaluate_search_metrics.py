"""
검색 성능 평가 지표
===================

Top-K Accuracy, Precision@K, Recall@K, MRR, MAP 등 계산
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

import pandas as pd
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.search_pipeline import SearchPipeline
from utils.config import get_system_config


class SearchMetricsEvaluator:
    """검색 성능 평가"""

    def __init__(self, pipeline: SearchPipeline):
        self.pipeline = pipeline

    def evaluate_single_query(
        self,
        query_idx: int,
        results: List[Dict],
        top_k_list: List[int] = [1, 5, 10, 20]
    ) -> Dict:
        """
        단일 쿼리 평가

        Args:
            query_idx: 나인오즈 쿼리 인덱스
            results: 검색 결과
            top_k_list: 평가할 K 값들

        Returns:
            평가 지표
        """
        # 쿼리 아이템 정보
        query_item = self.pipeline.get_query_item(query_idx)
        # Use category_id instead of kfashion_category (which doesn't exist in CSV)
        query_category = query_item.get('category_code', query_item.get('category_id', ''))

        # 결과 카테고리
        result_categories = [r.get('category_id', '') for r in results]

        metrics = {}

        # Top-K Accuracy (카테고리 일치 여부)
        for k in top_k_list:
            top_k_categories = result_categories[:k]
            # 카테고리가 하나라도 일치하면 성공
            hit = any(query_category in cat for cat in top_k_categories if cat)
            metrics[f'top_{k}_accuracy'] = 1.0 if hit else 0.0

        # Precision@K (관련 문서 비율)
        for k in top_k_list:
            top_k_categories = result_categories[:k]
            relevant_count = sum(1 for cat in top_k_categories if query_category in cat and cat)
            metrics[f'precision@{k}'] = relevant_count / k if k > 0 else 0.0

        # Recall@K (전체 관련 문서 중 찾은 비율)
        # 전체 네이버 DB에서 해당 카테고리 개수
        total_relevant = len(self.pipeline.naver_df[
            self.pipeline.naver_df['category_id'] == query_category
        ]) if query_category else 0

        for k in top_k_list:
            top_k_categories = result_categories[:k]
            relevant_count = sum(1 for cat in top_k_categories if query_category in cat and cat)
            metrics[f'recall@{k}'] = (
                relevant_count / total_relevant if total_relevant > 0 else 0.0
            )

        # MRR (Mean Reciprocal Rank) - 첫 번째 관련 문서 위치
        first_relevant_rank = None
        for rank, cat in enumerate(result_categories, 1):
            if query_category in cat and cat:
                first_relevant_rank = rank
                break

        metrics['mrr'] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

        # Average Precision (AP)
        relevant_count = 0
        precision_sum = 0.0
        for rank, cat in enumerate(result_categories, 1):
            if query_category in cat and cat:
                relevant_count += 1
                precision_sum += relevant_count / rank

        metrics['ap'] = (
            precision_sum / total_relevant if total_relevant > 0 else 0.0
        )

        return metrics

    def evaluate_dataset(
        self,
        n_samples: int = 100,
        initial_k: int = 100,
        top_k_list: List[int] = [1, 5, 10, 20],
        save_results: bool = True,
    ) -> Dict:
        """
        데이터셋 전체 평가

        Args:
            n_samples: 평가할 샘플 수
            initial_k: 초기 검색 결과 수
            top_k_list: 평가할 K 값들
            save_results: 결과 저장 여부

        Returns:
            집계된 평가 지표
        """
        print(f"\n{'='*80}")
        print(f"Search Performance Evaluation")
        print(f"{'='*80}")
        print(f"Samples: {n_samples}")
        print(f"Initial K: {initial_k}")
        print(f"Evaluating Top-K: {top_k_list}")
        print(f"{'='*80}\n")

        all_metrics = []
        n_queries = min(n_samples, len(self.pipeline.nineoz_df))

        for idx in tqdm(range(n_queries), desc="Evaluating queries"):
            # 쿼리 이미지 가져오기
            query_image_url = self.pipeline.nineoz_df.iloc[idx]['image_url']

            # 검색 수행 (카테고리 필터 없이)
            try:
                results = self.pipeline.search_by_image(
                    image_source=query_image_url,
                    category_filter=None,  # 필터 없이 전체 검색
                    initial_k=initial_k,
                    final_k=initial_k,
                )

                # 평가
                metrics = self.evaluate_single_query(idx, results, top_k_list)
                metrics['query_idx'] = idx
                all_metrics.append(metrics)

            except Exception as e:
                print(f"\n[Warning] Query {idx} failed: {e}")
                continue

        # 집계
        df_metrics = pd.DataFrame(all_metrics)

        # 평균 계산
        aggregated = {
            'n_samples': len(df_metrics),
        }

        for col in df_metrics.columns:
            if col != 'query_idx':
                aggregated[f'mean_{col}'] = df_metrics[col].mean()
                aggregated[f'std_{col}'] = df_metrics[col].std()

        # MAP (Mean Average Precision)
        aggregated['MAP'] = df_metrics['ap'].mean()

        # MRR (Mean Reciprocal Rank)
        aggregated['MRR'] = df_metrics['mrr'].mean()

        # 결과 출력
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"\nSamples Evaluated: {aggregated['n_samples']}\n")

        print("Top-K Accuracy:")
        for k in top_k_list:
            acc = aggregated[f'mean_top_{k}_accuracy']
            std = aggregated[f'std_top_{k}_accuracy']
            print(f"  Top-{k:2d}: {acc:.3f} ± {std:.3f}")

        print(f"\nPrecision@K:")
        for k in top_k_list:
            prec = aggregated[f'mean_precision@{k}']
            std = aggregated[f'std_precision@{k}']
            print(f"  P@{k:2d}: {prec:.3f} ± {std:.3f}")

        print(f"\nRecall@K:")
        for k in top_k_list:
            rec = aggregated[f'mean_recall@{k}']
            std = aggregated[f'std_recall@{k}']
            print(f"  R@{k:2d}: {rec:.3f} ± {std:.3f}")

        print(f"\nOverall Metrics:")
        print(f"  MAP:  {aggregated['MAP']:.3f}")
        print(f"  MRR:  {aggregated['MRR']:.3f}")
        print(f"{'='*80}\n")

        # 저장
        if save_results:
            output_dir = Path("results/evaluation")
            output_dir.mkdir(parents=True, exist_ok=True)

            # 상세 결과
            df_metrics.to_csv(output_dir / "detailed_metrics.csv", index=False)
            print(f"Saved detailed metrics: {output_dir / 'detailed_metrics.csv'}")

            # 집계 결과
            with open(output_dir / "aggregated_metrics.json", 'w') as f:
                json.dump(aggregated, f, indent=2)
            print(f"Saved aggregated metrics: {output_dir / 'aggregated_metrics.json'}")

        return aggregated


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate search performance")
    parser.add_argument("--n_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--initial_k", type=int, default=100,
                       help="Initial search results to retrieve")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save results")

    args = parser.parse_args()

    # 설정 로드
    config = get_system_config()

    print("Initializing search pipeline...")
    pipeline = SearchPipeline(
        nineoz_csv_path=config.nineoz_csv_path,
        naver_csv_path=config.naver_csv_path,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
        faiss_index_path=config.faiss_index_path if config.use_faiss else None,
        use_faiss=config.use_faiss,
        precompute_embeddings=False,
    )

    # 평가
    evaluator = SearchMetricsEvaluator(pipeline)
    results = evaluator.evaluate_dataset(
        n_samples=args.n_samples,
        initial_k=args.initial_k,
        top_k_list=[1, 5, 10, 20, 50],
        save_results=not args.no_save,
    )


if __name__ == "__main__":
    main()
