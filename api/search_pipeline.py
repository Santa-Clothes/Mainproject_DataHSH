"""
나인오즈 → 네이버 검색 파이프라인
====================================

모델과 독립적으로 작동하는 검색 로직
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.category_mapper import filter_by_item_type


class SearchPipeline:
    """검색 파이프라인"""

    def __init__(
        self,
        nineoz_csv_path: str,
        naver_csv_path: str,
        model=None,  # 나중에 로드
    ):
        """
        Args:
            nineoz_csv_path: 나인오즈 CSV 경로
            naver_csv_path: 네이버 CSV 경로
            model: K-Fashion 모델 (optional)
        """
        self.nineoz_csv_path = nineoz_csv_path
        self.naver_csv_path = naver_csv_path
        self.model = model

        # 데이터 로드
        self.nineoz_df = None
        self.naver_df = None
        self._load_data()

    def _load_data(self):
        """CSV 데이터 로드"""
        print(f"Loading Nine Oz data from {self.nineoz_csv_path}...")
        self.nineoz_df = pd.read_csv(self.nineoz_csv_path)
        print(f"  Loaded: {len(self.nineoz_df)} products")

        print(f"Loading Naver data from {self.naver_csv_path}...")
        self.naver_df = pd.read_csv(self.naver_csv_path)
        print(f"  Loaded: {len(self.naver_df)} products")

    def get_query_item(self, index: int = 0) -> Dict:
        """
        나인오즈 쿼리 아이템 가져오기

        Args:
            index: 나인오즈 CSV 인덱스

        Returns:
            쿼리 아이템 정보
        """
        if index >= len(self.nineoz_df):
            raise ValueError(f"Index {index} out of range")

        row = self.nineoz_df.iloc[index]
        return {
            "index": index,
            "product_code": row.get("제품코드", ""),
            "product_name": row.get("제품명", ""),
            "category_code": row.get("품목코드", ""),
            "category_name": row.get("품목명", ""),
            "color": row.get("칼라명", ""),
            "kfashion_category": row.get("kfashion_item_category", ""),
        }

    def search_by_embedding(
        self, query_embedding: np.ndarray, top_k: int = 100
    ) -> List[Dict]:
        """
        임베딩 기반 검색 (모델 필요)

        Args:
            query_embedding: 쿼리 이미지 임베딩
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        if self.model is None:
            # 모델 없으면 랜덤 결과 반환 (개발/테스트용)
            print("Warning: No model loaded. Returning random results.")
            indices = np.random.choice(len(self.naver_df), top_k, replace=False)
            results = []
            for idx in indices:
                row = self.naver_df.iloc[idx]
                results.append(
                    {
                        "product_id": row["product_id"],
                        "title": row["title"],
                        "price": row["price"],
                        "image_url": row["image_url"],
                        "category_id": row["category_id"],
                        "kfashion_category": row["kfashion_item_category"],
                        "score": np.random.random(),  # 랜덤 점수
                    }
                )
            return results

        # 실제 모델 검색 (나중에 구현)
        # TODO: model.search(query_embedding, top_k)
        raise NotImplementedError("Model-based search not implemented yet")

    def filter_by_category(
        self, results: List[Dict], target_category: str
    ) -> List[Dict]:
        """
        카테고리 필터링 (색상은 무시)

        Args:
            results: 검색 결과
            target_category: 타겟 K-Fashion 카테고리

        Returns:
            필터링된 결과
        """
        filtered = []
        for result in results:
            result_category = result.get("kfashion_category", "")

            # 타겟 카테고리 포함 여부 확인
            if target_category in result_category:
                filtered.append(result)

        return filtered

    def rank_results(self, results: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        결과 랭킹

        Args:
            results: 필터링된 결과
            top_k: 최종 반환 개수

        Returns:
            상위 top_k 결과
        """
        # 점수로 정렬
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_results[:top_k]

    def search(
        self,
        query_index: int = 0,
        query_embedding: Optional[np.ndarray] = None,
        initial_k: int = 100,
        final_k: int = 10,
    ) -> Dict:
        """
        전체 검색 파이프라인

        Args:
            query_index: 나인오즈 CSV 인덱스
            query_embedding: 쿼리 임베딩 (optional)
            initial_k: 초기 검색 결과 수
            final_k: 최종 반환 결과 수

        Returns:
            검색 결과 딕셔너리
        """
        # 1. 쿼리 아이템 가져오기
        query_item = self.get_query_item(query_index)
        print(f"\nQuery: {query_item['product_name']} ({query_item['color']})")
        print(f"Category: {query_item['kfashion_category']}")

        # 2. 임베딩 검색 (initial_k개)
        if query_embedding is None:
            # 임베딩 없으면 랜덤 생성 (테스트용)
            query_embedding = np.random.randn(512)

        search_results = self.search_by_embedding(query_embedding, top_k=initial_k)
        print(f"\n1. Initial search: {len(search_results)} results")

        # 3. 카테고리 필터링
        filtered_results = self.filter_by_category(
            search_results, query_item["kfashion_category"]
        )
        print(f"2. After category filter: {len(filtered_results)} results")

        # 4. 랭킹
        final_results = self.rank_results(filtered_results, top_k=final_k)
        print(f"3. Final ranking: {len(final_results)} results")

        return {
            "query": query_item,
            "results": final_results,
            "stats": {
                "initial_count": len(search_results),
                "filtered_count": len(filtered_results),
                "final_count": len(final_results),
            },
        }


if __name__ == "__main__":
    # 테스트
    pipeline = SearchPipeline(
        nineoz_csv_path="c:/Work/hwangseonghun/nineoz_with_kfashion_categories.csv",
        naver_csv_path="c:/Work/hwangseonghun/naver_with_kfashion_categories.csv",
    )

    # 검색 테스트 (랜덤 임베딩)
    result = pipeline.search(query_index=0, initial_k=100, final_k=10)

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    print(f"\nQuery: {result['query']['product_name']}")
    print(f"Category: {result['query']['kfashion_category']}")
    print(f"\nTop 10 Recommendations:")
    for i, item in enumerate(result["results"], 1):
        print(f"{i:2d}. {item['title'][:50]:50s} (Score: {item['score']:.3f})")
