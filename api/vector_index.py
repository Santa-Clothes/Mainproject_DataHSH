"""
FAISS 벡터 인덱스 관리
====================

네이버 제품 임베딩을 FAISS 인덱스로 관리하여 고속 검색 제공
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


class FaissVectorIndex:
    """FAISS 기반 벡터 인덱스"""

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_dim: int = 768,  # FashionCLIP 기본 차원
        use_gpu: bool = False,
    ):
        """
        Args:
            index_path: 인덱스 파일 경로 (.index)
            embedding_dim: 임베딩 차원
            use_gpu: GPU 사용 여부 (faiss-gpu 설치 필요)
        """
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index = None
        self.product_ids = []  # FAISS 인덱스와 product_id 매핑

        # 인덱스 로드 또는 생성
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
        else:
            self._create_empty_index()

    def _create_empty_index(self):
        """빈 FAISS 인덱스 생성"""
        # IndexFlatIP: Inner Product (코사인 유사도 - L2 normalized 벡터 필요)
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            print(f"[FAISS] Using GPU acceleration")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            print(f"[FAISS] Using CPU")

        print(f"[FAISS] Created empty index (dim={self.embedding_dim})")

    def build_index(
        self,
        embeddings: np.ndarray,
        product_ids: List[str],
        normalize: bool = True,
    ):
        """
        임베딩으로 인덱스 구축

        Args:
            embeddings: [num_products, embedding_dim]
            product_ids: 제품 ID 리스트
            normalize: L2 정규화 여부 (코사인 유사도 위해 필요)
        """
        if len(embeddings) != len(product_ids):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) != Product IDs count ({len(product_ids)})"
            )

        print(f"\n[FAISS] Building index...")
        print(f"  Products: {len(product_ids)}")
        print(f"  Embedding dim: {embeddings.shape[1]}")

        # L2 정규화 (코사인 유사도를 위해)
        if normalize:
            print(f"  Normalizing embeddings...")
            faiss.normalize_L2(embeddings)

        # 인덱스에 추가
        self._create_empty_index()
        self.index.add(embeddings.astype(np.float32))
        self.product_ids = product_ids

        print(f"  [OK] Index built: {self.index.ntotal} vectors")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 100, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        벡터 검색

        Args:
            query_embedding: 쿼리 임베딩 [embedding_dim] or [1, embedding_dim]
            top_k: 반환할 결과 수
            normalize: 쿼리 정규화 여부

        Returns:
            (distances, indices): 거리 점수와 인덱스
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty. Build index first.")

        # 쿼리 정규화
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        if normalize:
            faiss.normalize_L2(query_embedding)

        # 검색
        distances, indices = self.index.search(query_embedding, top_k)

        return distances[0], indices[0]  # [top_k], [top_k]

    def get_product_ids(self, indices: np.ndarray) -> List[str]:
        """
        인덱스를 product_id로 변환

        Args:
            indices: FAISS 인덱스 배열

        Returns:
            product_id 리스트
        """
        return [self.product_ids[idx] for idx in indices if idx < len(self.product_ids)]

    def save_index(self, save_path: str):
        """인덱스를 디스크에 저장"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # GPU 인덱스면 CPU로 변환 후 저장
        index_to_save = self.index
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(index_to_save, str(save_path))
        print(f"[FAISS] Index saved: {save_path}")

        # Product IDs도 함께 저장
        ids_path = save_path.with_suffix(".ids.npy")
        np.save(ids_path, np.array(self.product_ids, dtype=object))
        print(f"[FAISS] Product IDs saved: {ids_path}")

    def load_index(self, index_path: str):
        """디스크에서 인덱스 로드"""
        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        print(f"[FAISS] Loading index: {index_path}")
        self.index = faiss.read_index(str(index_path))

        # GPU 변환
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print(f"[FAISS] Moved index to GPU")

        # Product IDs 로드
        ids_path = index_path.with_suffix(".ids.npy")
        if ids_path.exists():
            self.product_ids = np.load(ids_path, allow_pickle=True).tolist()
            print(f"[FAISS] Product IDs loaded: {len(self.product_ids)}")
        else:
            print(f"[WARNING] Product IDs file not found: {ids_path}")
            self.product_ids = []

        print(f"[FAISS] Index loaded: {self.index.ntotal} vectors")

    def get_stats(self) -> dict:
        """인덱스 통계"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "use_gpu": self.use_gpu,
            "index_path": str(self.index_path) if self.index_path else None,
            "num_products": len(self.product_ids),
        }


def build_naver_index(
    naver_csv_path: str,
    embedding_generator,
    output_path: str = "data/indexes/naver.index",
    batch_size: int = 32,
) -> FaissVectorIndex:
    """
    네이버 CSV로부터 FAISS 인덱스 빌드

    Args:
        naver_csv_path: 네이버 CSV 경로
        embedding_generator: FashionCLIPEmbeddingGenerator 인스턴스
        output_path: 인덱스 저장 경로
        batch_size: 임베딩 생성 배치 크기

    Returns:
        구축된 FaissVectorIndex
    """
    print(f"\n{'='*80}")
    print(f"Building Naver FAISS Index")
    print(f"{'='*80}")

    # CSV 로드
    print(f"\n[1] Loading CSV: {naver_csv_path}")
    df = pd.read_csv(naver_csv_path)
    print(f"  [OK] {len(df)} products loaded")

    # 임베딩 생성
    print(f"\n[2] Generating embeddings (batch_size={batch_size})")
    image_urls = df['image_url'].tolist()
    product_ids = df['product_id'].astype(str).tolist()

    embeddings = embedding_generator.generate_embeddings_batch(
        image_sources=image_urls,
        batch_size=batch_size,
        normalize=True,
        show_progress=True,
    )

    # 인덱스 빌드
    print(f"\n[3] Building FAISS index")
    index = FaissVectorIndex(embedding_dim=embeddings.shape[1])
    index.build_index(embeddings, product_ids, normalize=False)  # 이미 정규화됨

    # 저장
    print(f"\n[4] Saving index: {output_path}")
    index.save_index(output_path)

    print(f"\n{'='*80}")
    print(f"Index built successfully!")
    print(f"  Vectors: {index.index.ntotal}")
    print(f"  Dimension: {embeddings.shape[1]}")
    print(f"  File: {output_path}")
    print(f"{'='*80}\n")

    return index


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.embedding_generator import FashionCLIPEmbeddingGenerator
    from utils.config import get_system_config

    # 설정 로드
    config = get_system_config()

    # 임베딩 생성기
    generator = FashionCLIPEmbeddingGenerator(
        checkpoint_path=config.checkpoint_path,
        device=config.device,
    )

    # 인덱스 빌드
    index = build_naver_index(
        naver_csv_path=config.naver_csv_path,
        embedding_generator=generator,
        output_path="data/indexes/naver.index",
        batch_size=config.embedding_batch_size,
    )

    # 테스트 검색
    print("\n" + "="*80)
    print("Testing search...")
    print("="*80)

    # 첫 번째 제품으로 테스트
    test_embedding = np.random.randn(1, index.embedding_dim).astype(np.float32)
    distances, indices = index.search(test_embedding, top_k=5)

    print("\nTop 5 results:")
    for i, (dist, idx) in enumerate(zip(distances, indices), 1):
        product_id = index.product_ids[idx]
        print(f"{i}. Product ID: {product_id}, Score: {dist:.4f}")
