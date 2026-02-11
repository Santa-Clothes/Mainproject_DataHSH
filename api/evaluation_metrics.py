"""
검색 품질 평가 메트릭
===========================

검색 결과의 품질을 측정하는 다양한 메트릭 제공
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class SearchMetrics:
    """검색 결과 평가 메트릭"""

    @staticmethod
    def precision_at_k(
        retrieved: List[str], relevant: List[str], k: int = 10
    ) -> float:
        """
        Precision@K: 상위 K개 결과 중 관련 항목 비율

        Args:
            retrieved: 검색 결과 ID 리스트
            relevant: 관련 항목 ID 리스트
            k: 평가할 상위 결과 수

        Returns:
            Precision@K 값 (0.0 ~ 1.0)
        """
        if not retrieved or not relevant:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in retrieved_k if item in relevant)
        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Recall@K: 전체 관련 항목 중 상위 K개에서 찾은 비율

        Args:
            retrieved: 검색 결과 ID 리스트
            relevant: 관련 항목 ID 리스트
            k: 평가할 상위 결과 수

        Returns:
            Recall@K 값 (0.0 ~ 1.0)
        """
        if not retrieved or not relevant:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in retrieved_k if item in relevant)
        return relevant_retrieved / len(relevant)

    @staticmethod
    def average_precision(retrieved: List[str], relevant: List[str]) -> float:
        """
        Average Precision: 모든 관련 항목의 precision 평균

        Args:
            retrieved: 검색 결과 ID 리스트
            relevant: 관련 항목 ID 리스트

        Returns:
            Average Precision 값 (0.0 ~ 1.0)
        """
        if not retrieved or not relevant:
            return 0.0

        precisions = []
        relevant_count = 0

        for i, item in enumerate(retrieved, 1):
            if item in relevant:
                relevant_count += 1
                precisions.append(relevant_count / i)

        return sum(precisions) / len(relevant) if precisions else 0.0

    @staticmethod
    def mean_average_precision(
        all_retrieved: List[List[str]], all_relevant: List[List[str]]
    ) -> float:
        """
        Mean Average Precision (MAP): 여러 쿼리의 AP 평균

        Args:
            all_retrieved: 각 쿼리의 검색 결과 리스트
            all_relevant: 각 쿼리의 관련 항목 리스트

        Returns:
            MAP 값 (0.0 ~ 1.0)
        """
        if not all_retrieved or not all_relevant:
            return 0.0

        aps = []
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            ap = SearchMetrics.average_precision(retrieved, relevant)
            aps.append(ap)

        return np.mean(aps) if aps else 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevant: List[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k: int = 10,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain (NDCG@K)

        Args:
            retrieved: 검색 결과 ID 리스트
            relevant: 관련 항목 ID 리스트
            relevance_scores: 항목별 관련도 점수 (optional)
            k: 평가할 상위 결과 수

        Returns:
            NDCG@K 값 (0.0 ~ 1.0)
        """
        if not retrieved or not relevant:
            return 0.0

        # 관련도 점수가 없으면 binary relevance 사용
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant}

        # DCG 계산
        dcg = 0.0
        for i, item in enumerate(retrieved[:k], 1):
            rel = relevance_scores.get(item, 0.0)
            dcg += rel / np.log2(i + 1)

        # IDCG 계산 (이상적인 순서)
        ideal_scores = sorted(
            [relevance_scores.get(item, 0.0) for item in relevant], reverse=True
        )
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores[:k]))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def category_accuracy(
        retrieved: List[Dict], query_category: str, k: int = 10
    ) -> float:
        """
        카테고리 일치율: 상위 K개 결과 중 올바른 카테고리 비율

        Args:
            retrieved: 검색 결과 리스트 (각 항목은 dict with 'kfashion_category')
            query_category: 쿼리 카테고리
            k: 평가할 상위 결과 수

        Returns:
            카테고리 일치율 (0.0 ~ 1.0)
        """
        if not retrieved:
            return 0.0

        retrieved_k = retrieved[:k]
        matches = sum(
            1
            for item in retrieved_k
            if query_category in item.get("kfashion_category", "")
        )
        return matches / k

    @staticmethod
    def diversity_score(retrieved: List[Dict], category_key: str = "kfashion_category") -> float:
        """
        다양성 점수: 검색 결과의 카테고리 다양성 측정

        Args:
            retrieved: 검색 결과 리스트
            category_key: 카테고리 키

        Returns:
            다양성 점수 (0.0 ~ 1.0, 높을수록 다양함)
        """
        if not retrieved:
            return 0.0

        categories = [item.get(category_key, "") for item in retrieved]
        unique_categories = len(set(categories))
        return unique_categories / len(categories) if categories else 0.0


class SearchEvaluator:
    """검색 시스템 종합 평가"""

    def __init__(self, pipeline, test_queries: List[int]):
        """
        Args:
            pipeline: SearchPipeline 인스턴스
            test_queries: 테스트 쿼리 인덱스 리스트
        """
        self.pipeline = pipeline
        self.test_queries = test_queries
        self.results = []

    def evaluate_single_query(
        self, query_index: int, initial_k: int = 100, final_k: int = 10
    ) -> Dict:
        """
        단일 쿼리 평가

        Args:
            query_index: 쿼리 인덱스
            initial_k: 초기 검색 결과 수
            final_k: 최종 결과 수

        Returns:
            평가 결과 딕셔너리
        """
        # 검색 실행
        result = self.pipeline.search(
            query_index=query_index, initial_k=initial_k, final_k=final_k
        )

        query_item = result["query"]
        search_results = result["results"]
        stats = result["stats"]

        # 메트릭 계산
        metrics = {
            "query_index": query_index,
            "query_product": query_item["product_name"],
            "query_category": query_item["kfashion_category"],
            "category_accuracy": SearchMetrics.category_accuracy(
                search_results, query_item["kfashion_category"], k=final_k
            ),
            "diversity": SearchMetrics.diversity_score(search_results),
            "initial_count": stats["initial_count"],
            "filtered_count": stats["filtered_count"],
            "final_count": stats["final_count"],
            "filter_rate": stats["filtered_count"] / stats["initial_count"]
            if stats["initial_count"] > 0
            else 0.0,
        }

        return metrics

    def evaluate_all(self, initial_k: int = 100, final_k: int = 10) -> pd.DataFrame:
        """
        전체 쿼리 평가

        Args:
            initial_k: 초기 검색 결과 수
            final_k: 최종 결과 수

        Returns:
            평가 결과 DataFrame
        """
        print(f"\nEvaluating {len(self.test_queries)} queries...")

        results = []
        for i, query_idx in enumerate(self.test_queries, 1):
            print(f"  [{i}/{len(self.test_queries)}] Query {query_idx}...", end=" ")

            try:
                metrics = self.evaluate_single_query(
                    query_idx, initial_k=initial_k, final_k=final_k
                )
                results.append(metrics)
                print(f"Category Acc: {metrics['category_accuracy']:.2%}")
            except Exception as e:
                print(f"Error: {e}")
                continue

        df = pd.DataFrame(results)
        self.results = df
        return df

    def print_summary(self):
        """평가 결과 요약 출력"""
        if self.results is None or len(self.results) == 0:
            print("No evaluation results available.")
            return

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\nTotal Queries: {len(self.results)}")
        print(f"\nAverage Metrics:")
        print(f"  Category Accuracy: {self.results['category_accuracy'].mean():.2%}")
        print(f"  Diversity Score:   {self.results['diversity'].mean():.2%}")
        print(f"  Filter Rate:       {self.results['filter_rate'].mean():.2%}")

        print(f"\nFiltering Stats:")
        print(f"  Avg Initial Results:  {self.results['initial_count'].mean():.1f}")
        print(f"  Avg Filtered Results: {self.results['filtered_count'].mean():.1f}")
        print(f"  Avg Final Results:    {self.results['final_count'].mean():.1f}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    # 테스트
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from search_pipeline import SearchPipeline

    # 파이프라인 생성
    pipeline = SearchPipeline(
        nineoz_csv_path="c:/Work/hwangseonghun/nineoz_with_kfashion_categories.csv",
        naver_csv_path="c:/Work/hwangseonghun/naver_with_kfashion_categories.csv",
    )

    # 평가기 생성 (테스트 쿼리 10개)
    test_queries = list(range(0, 10))
    evaluator = SearchEvaluator(pipeline, test_queries)

    # 평가 실행
    results_df = evaluator.evaluate_all(initial_k=100, final_k=10)

    # 요약 출력
    evaluator.print_summary()

    # 상세 결과 저장
    results_df.to_csv("c:/Work/hwangseonghun/FinalProject_v2/results/search_evaluation.csv", index=False)
    print("\nDetailed results saved to results/search_evaluation.csv")
