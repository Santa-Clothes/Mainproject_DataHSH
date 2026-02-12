"""
FAISS 인덱스 빌드 스크립트
========================

네이버 제품 CSV로부터 FAISS 인덱스 생성
"""

import sys
from pathlib import Path
import argparse

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.vector_index import build_naver_index
from models.embedding_generator import FashionCLIPEmbeddingGenerator
from utils.config import get_system_config


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for Naver products")
    parser.add_argument(
        "--naver_csv",
        type=str,
        default=None,
        help="Naver CSV path (default: from .env)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (default: from .env)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output index path (default: from .env)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for embedding generation (default: from .env)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device (default: auto-detect)"
    )

    args = parser.parse_args()

    # 설정 로드
    config = get_system_config()

    # 파라미터 설정 (CLI > .env > default)
    naver_csv_path = args.naver_csv or config.naver_csv_path
    checkpoint_path = args.checkpoint or config.checkpoint_path
    output_path = args.output or config.faiss_index_path
    batch_size = args.batch_size or config.embedding_batch_size
    device = args.device or config.device

    print("\n" + "="*80)
    print("FAISS Index Builder")
    print("="*80)
    print(f"\n[Configuration]")
    print(f"  Naver CSV: {naver_csv_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output: {output_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device or 'auto'}")
    print("="*80)

    # 임베딩 생성기 초기화
    print(f"\n[1] Initializing embedding generator...")
    generator = FashionCLIPEmbeddingGenerator(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # FAISS 인덱스 빌드
    print(f"\n[2] Building FAISS index...")
    index = build_naver_index(
        naver_csv_path=naver_csv_path,
        embedding_generator=generator,
        output_path=output_path,
        batch_size=batch_size,
    )

    print(f"\n{'='*80}")
    print(f"✓ FAISS index built successfully!")
    print(f"{'='*80}")
    print(f"\n[Index Stats]")
    stats = index.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"\n[Usage]")
    print(f"  Set USE_FAISS=true in .env")
    print(f"  Set FAISS_INDEX_PATH={output_path}")
    print(f"  Restart API server")
    print("="*80)


if __name__ == "__main__":
    main()
