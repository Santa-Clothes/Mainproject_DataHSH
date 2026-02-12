"""
나인오즈 → 네이버 검색 파이프라인
====================================

FashionCLIP 기반 임베딩 검색 로직
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.embedding_generator import FashionCLIPEmbeddingGenerator
from api.vector_index import FaissVectorIndex
from utils.supabase_loader import SupabaseLoader


class SearchPipeline:
    """검색 파이프라인"""

    def __init__(
        self,
        nineoz_csv_path: Optional[str] = None,
        naver_csv_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        precompute_embeddings: bool = False,
        faiss_index_path: Optional[str] = None,
        use_faiss: bool = True,
        data_source: str = "csv",
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        nineoz_table: str = "internal_products",
        naver_table: str = "naver_products",
    ):
        """
        Args:
            nineoz_csv_path: 나인오즈 CSV 경로 (data_source=csv인 경우)
            naver_csv_path: 네이버 CSV 경로 (data_source=csv인 경우)
            checkpoint_path: FashionCLIP 체크포인트 경로
            device: 'cuda' or 'cpu' (None이면 자동 감지)
            precompute_embeddings: 네이버 제품 임베딩 사전 계산 여부 (FAISS 미사용 시)
            faiss_index_path: FAISS 인덱스 파일 경로 (.index)
            use_faiss: FAISS 사용 여부 (True 권장)
            data_source: 데이터 소스 ('csv' or 'supabase')
            supabase_url: Supabase project URL (data_source=supabase인 경우)
            supabase_key: Supabase API key (data_source=supabase인 경우)
            nineoz_table: Nine Oz 테이블 이름 (data_source=supabase인 경우)
            naver_table: Naver 테이블 이름 (data_source=supabase인 경우)
        """
        self.nineoz_csv_path = nineoz_csv_path
        self.naver_csv_path = naver_csv_path
        self.use_faiss = use_faiss
        self.data_source = data_source
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.nineoz_table = nineoz_table
        self.naver_table = naver_table

        # 데이터 로드
        self.nineoz_df = None
        self.naver_df = None
        self._load_data()

        # 임베딩 생성기 로드
        print("\n[Search Pipeline] Initializing embedding generator...")
        self.embedding_generator = FashionCLIPEmbeddingGenerator(
            checkpoint_path=checkpoint_path,
            device=device
        )

        # FAISS 인덱스 또는 numpy 임베딩
        self.faiss_index = None
        self.naver_embeddings = None

        if use_faiss and faiss_index_path:
            # FAISS 인덱스 로드
            self._load_faiss_index(faiss_index_path)
        elif precompute_embeddings:
            # Fallback: numpy 임베딩 사전 계산
            self._precompute_naver_embeddings()

    def _load_data(self):
        """데이터 로드 (CSV or Supabase)"""
        if self.data_source == "supabase":
            # Supabase에서 로드
            print(f"\n[Search Pipeline] Loading data from Supabase...")
            loader = SupabaseLoader(url=self.supabase_url, key=self.supabase_key)

            print(f"[Search Pipeline] Loading Nine Oz data from table: {self.nineoz_table}")
            self.nineoz_df = loader.load_nineoz_products(table_name=self.nineoz_table)
            print(f"  [OK] Loaded: {len(self.nineoz_df)} products")

            print(f"[Search Pipeline] Loading Naver data from table: {self.naver_table}")
            self.naver_df = loader.load_naver_products(table_name=self.naver_table)
            print(f"  [OK] Loaded: {len(self.naver_df)} products")

        else:
            # CSV에서 로드
            print(f"\n[Search Pipeline] Loading Nine Oz data from {self.nineoz_csv_path}...")
            self.nineoz_df = pd.read_csv(self.nineoz_csv_path)
            print(f"  [OK] Loaded: {len(self.nineoz_df)} products")

            print(f"[Search Pipeline] Loading Naver data from {self.naver_csv_path}...")
            self.naver_df = pd.read_csv(self.naver_csv_path)
            print(f"  [OK] Loaded: {len(self.naver_df)} products")

    def _load_faiss_index(self, index_path: str):
        """FAISS 인덱스 로드"""
        try:
            print(f"\n[Search Pipeline] Loading FAISS index: {index_path}")
            self.faiss_index = FaissVectorIndex(index_path=index_path)
            print(f"  [OK] FAISS index loaded: {self.faiss_index.index.ntotal} vectors")
        except Exception as e:
            print(f"  [WARNING] Failed to load FAISS index: {e}")
            print(f"  [INFO] Falling back to on-the-fly embedding generation")
            self.faiss_index = None

    def _precompute_naver_embeddings(self):
        """네이버 제품 임베딩 사전 계산 (선택적)"""
        print(f"\n[Search Pipeline] Precomputing Naver product embeddings...")
        print(f"  This may take a while for {len(self.naver_df)} products...")

        # 이미지 URL 리스트
        image_urls = self.naver_df['image_url'].tolist()

        # 배치로 임베딩 생성
        self.naver_embeddings = self.embedding_generator.generate_embeddings_batch(
            image_sources=image_urls,
            batch_size=32,
            normalize=True,
            show_progress=True
        )

        print(f"  [OK] Precomputed embeddings shape: {self.naver_embeddings.shape}")

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
            "product_code": row.get("product_id", ""),
            "product_name": row.get("product_name", ""),
            "category_code": row.get("category_id", ""),
            "category_name": row.get("category_name", ""),  # May not exist
            "color": row.get("color", ""),
            "category_id": row.get("category_id", ""),  # For evaluation script
        }

    def search_by_embedding(
        self, query_embedding: np.ndarray, top_k: int = 100
    ) -> List[Dict]:
        """
        임베딩 기반 코사인 유사도 검색

        Args:
            query_embedding: 쿼리 이미지 임베딩 [embedding_dim]
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트 (유사도 점수 포함)
        """
        # FAISS 사용 가능 시
        if self.use_faiss and self.faiss_index is not None:
            return self._search_with_faiss(query_embedding, top_k)
        else:
            return self._search_with_numpy(query_embedding, top_k)

    def _search_with_faiss(
        self, query_embedding: np.ndarray, top_k: int = 100
    ) -> List[Dict]:
        """FAISS 기반 검색"""
        # FAISS 검색
        distances, indices = self.faiss_index.search(
            query_embedding, top_k=top_k, normalize=True
        )

        # Product IDs 가져오기
        product_ids = self.faiss_index.get_product_ids(indices)

        # 결과 생성
        results = []
        for product_id, score in zip(product_ids, distances):
            # DataFrame에서 제품 정보 찾기
            row = self.naver_df[self.naver_df['product_id'].astype(str) == str(product_id)]
            if not row.empty:
                row = row.iloc[0]
                results.append({
                    "product_id": str(row["product_id"]),  # Convert to native Python type
                    "title": str(row["title"]),
                    "price": int(row["price"]) if pd.notna(row["price"]) else 0,
                    "image_url": str(row["image_url"]),
                    "category_id": str(row["category_id"]),
                    "kfashion_category": str(row.get("kfashion_item_category", "")),
                    "score": float(score),
                })

        return results

    def _search_with_numpy(
        self, query_embedding: np.ndarray, top_k: int = 100
    ) -> List[Dict]:
        """Numpy 기반 검색 (Fallback)"""
        # 네이버 제품 임베딩 확보
        if self.naver_embeddings is None:
            # 사전 계산 안 된 경우 on-the-fly 생성
            print("[Search Pipeline] Generating Naver embeddings on-the-fly...")
            image_urls = self.naver_df['image_url'].tolist()
            self.naver_embeddings = self.embedding_generator.generate_embeddings_batch(
                image_sources=image_urls,
                batch_size=32,
                normalize=True,
                show_progress=True
            )

        # 코사인 유사도 계산 (이미 L2 normalized되어 있으면 dot product)
        # query_embedding: [embedding_dim]
        # naver_embeddings: [num_products, embedding_dim]
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        similarities = np.dot(self.naver_embeddings, query_norm)  # [num_products]

        # Top-k 인덱스 추출
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # 결과 생성
        results = []
        for idx in top_indices:
            row = self.naver_df.iloc[idx]
            results.append({
                "product_id": str(row["product_id"]),  # Convert to native Python type
                "title": str(row["title"]),
                "price": int(row["price"]) if pd.notna(row["price"]) else 0,
                "image_url": str(row["image_url"]),
                "category_id": str(row["category_id"]),
                "kfashion_category": str(row.get("kfashion_item_category", "")),
                "score": float(similarities[idx]),  # 코사인 유사도 점수
            })

        return results

    def filter_by_category(
        self, results: List[Dict], target_category: str
    ) -> List[Dict]:
        """
        카테고리 필터링

        Args:
            results: 검색 결과
            target_category: 타겟 카테고리 ID (e.g., 'BL', 'OP', 'SK')

        Returns:
            필터링된 결과
        """
        filtered = []
        for result in results:
            result_category = result.get("category_id", "")

            # 타겟 카테고리 일치 여부 확인 (exact match)
            if target_category == result_category:
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
        query_image: Optional[Union[str, Image.Image]] = None,
        query_embedding: Optional[np.ndarray] = None,
        initial_k: int = 100,
        final_k: int = 10,
    ) -> Dict:
        """
        전체 검색 파이프라인

        Args:
            query_index: 나인오즈 CSV 인덱스
            query_image: 쿼리 이미지 (경로, URL, 또는 PIL Image)
            query_embedding: 쿼리 임베딩 (optional, 이미 있으면 사용)
            initial_k: 초기 검색 결과 수
            final_k: 최종 반환 결과 수

        Returns:
            검색 결과 딕셔너리
        """
        # 1. 쿼리 아이템 가져오기
        query_item = self.get_query_item(query_index)
        print(f"\n[Search] Query: {query_item['product_name']} ({query_item['color']})")
        print(f"[Search] Category: {query_item.get('category_id', '')}")

        # 2. 쿼리 임베딩 생성
        if query_embedding is None:
            if query_image is None:
                raise ValueError("Either query_image or query_embedding must be provided")

            print(f"[Search] Generating query embedding...")
            query_embedding = self.embedding_generator.generate_embedding(
                query_image,
                normalize=True
            )
            print(f"[Search] Query embedding shape: {query_embedding.shape}")

        # 3. 임베딩 검색 (initial_k개)
        search_results = self.search_by_embedding(query_embedding, top_k=initial_k)
        print(f"[Search] 1. Initial search: {len(search_results)} results")

        # 4. 카테고리 필터링
        filtered_results = self.filter_by_category(
            search_results, query_item.get("category_id", "")
        )
        print(f"[Search] 2. After category filter: {len(filtered_results)} results")

        # 5. 랭킹
        final_results = self.rank_results(filtered_results, top_k=final_k)
        print(f"[Search] 3. Final ranking: {len(final_results)} results")

        return {
            "query": query_item,
            "results": final_results,
            "stats": {
                "initial_count": len(search_results),
                "filtered_count": len(filtered_results),
                "final_count": len(final_results),
            },
        }

    def search_by_image(
        self,
        image_source: Union[str, Image.Image],
        category_filter: Optional[str] = None,
        initial_k: int = 100,
        final_k: int = 10,
    ) -> List[Dict]:
        """
        이미지로 직접 검색 (나인오즈 CSV 없이)

        Args:
            image_source: 쿼리 이미지 (경로, URL, PIL Image)
            category_filter: 카테고리 필터 (optional)
            initial_k: 초기 검색 결과 수
            final_k: 최종 반환 결과 수

        Returns:
            검색 결과 리스트
        """
        print(f"\n[Search] Direct image search...")

        # 1. 임베딩 생성
        query_embedding = self.embedding_generator.generate_embedding(
            image_source,
            normalize=True
        )

        # 2. 임베딩 검색
        search_results = self.search_by_embedding(query_embedding, top_k=initial_k)
        print(f"[Search] Initial search: {len(search_results)} results")

        # 3. 카테고리 필터링 (optional)
        if category_filter:
            search_results = self.filter_by_category(search_results, category_filter)
            print(f"[Search] After category filter: {len(search_results)} results")

        # 4. Top-K
        final_results = search_results[:final_k]
        print(f"[Search] Final results: {len(final_results)}")

        return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search Pipeline Test")
    parser.add_argument("--nineoz_csv", type=str, required=True, help="Nine Oz CSV path")
    parser.add_argument("--naver_csv", type=str, required=True, help="Naver CSV path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--query_index", type=int, default=0, help="Query index")
    parser.add_argument("--query_image", type=str, default=None, help="Query image path/URL")
    parser.add_argument("--precompute", action="store_true", help="Precompute Naver embeddings")

    args = parser.parse_args()

    # 검색 파이프라인 초기화
    print("\n" + "=" * 80)
    print("Initializing Search Pipeline")
    print("=" * 80)

    pipeline = SearchPipeline(
        nineoz_csv_path=args.nineoz_csv,
        naver_csv_path=args.naver_csv,
        checkpoint_path=args.checkpoint,
        precompute_embeddings=args.precompute,
    )

    print("\n" + "=" * 80)
    print("Running Search")
    print("=" * 80)

    # 검색 실행
    result = pipeline.search(
        query_index=args.query_index,
        query_image=args.query_image,
        initial_k=100,
        final_k=10
    )

    print("\n" + "=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)
    print(f"\nQuery: {result['query']['product_name']}")
    print(f"Category: {result['query']['kfashion_category']}")
    print(f"\nTop 10 Recommendations:")
    for i, item in enumerate(result["results"], 1):
        print(f"{i:2d}. {item['title'][:50]:50s} (Score: {item['score']:.3f})")

    print("\n" + "=" * 80)
    print(f"Stats:")
    print(f"  Initial: {result['stats']['initial_count']}")
    print(f"  Filtered: {result['stats']['filtered_count']}")
    print(f"  Final: {result['stats']['final_count']}")
    print("=" * 80)
