"""
FashionCLIP 임베딩 시각화 (t-SNE & UMAP)

나인오즈 제품들의 FashionCLIP 임베딩을 t-SNE와 UMAP으로 2D/3D 시각화합니다.
카테고리별로 색상을 구분하여 표시합니다.
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.embedding_generator import FashionCLIPEmbeddingGenerator


def load_nineoz_data(csv_path: str, limit: int = None) -> pd.DataFrame:
    """나인오즈 제품 데이터 로드"""
    print(f"\n[1/5] Loading Nine Oz product data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if limit:
        df = df.head(limit)

    print(f"  ✓ Loaded {len(df)} products")

    # 이미지 URL이 있는 제품만 선택
    df = df[df['image_url'].notna()].copy()
    print(f"  ✓ Filtered to {len(df)} products with images")

    return df


def generate_embeddings(
    generator: FashionCLIPEmbeddingGenerator,
    df: pd.DataFrame
) -> np.ndarray:
    """
    FashionCLIP 임베딩 생성

    Args:
        generator: 임베딩 생성기
        df: 제품 DataFrame

    Returns:
        임베딩 행렬 [num_products, embedding_dim]
    """
    print(f"\n[2/5] Generating FashionCLIP embeddings...")

    image_urls = df['image_url'].tolist()

    # 배치로 임베딩 생성
    embeddings = generator.generate_embeddings_batch(
        image_sources=image_urls,
        batch_size=32,
        normalize=True,
        show_progress=True
    )

    print(f"  ✓ Generated embeddings: {embeddings.shape}")
    return embeddings


def apply_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42
) -> np.ndarray:
    """
    t-SNE 적용

    Args:
        embeddings: 임베딩 행렬
        n_components: 출력 차원 (2 or 3)
        perplexity: t-SNE perplexity
        random_state: 랜덤 시드

    Returns:
        t-SNE 결과 [num_samples, n_components]
    """
    print(f"\n[3/5] Applying t-SNE (perplexity={perplexity}, dim={n_components})...")

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        verbose=1
    )

    tsne_result = tsne.fit_transform(embeddings)
    print(f"  ✓ t-SNE completed: {tsne_result.shape}")

    return tsne_result


def apply_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    UMAP 적용

    Args:
        embeddings: 임베딩 행렬
        n_components: 출력 차원 (2 or 3)
        n_neighbors: UMAP n_neighbors
        min_dist: UMAP min_dist
        random_state: 랜덤 시드

    Returns:
        UMAP 결과 [num_samples, n_components]
    """
    print(f"\n[4/5] Applying UMAP (neighbors={n_neighbors}, dim={n_components})...")

    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )

    umap_result = umap_model.fit_transform(embeddings)
    print(f"  ✓ UMAP completed: {umap_result.shape}")

    return umap_result


def plot_2d(
    result: np.ndarray,
    labels: pd.Series,
    title: str,
    output_path: str
):
    """
    2D 시각화

    Args:
        result: 2D 좌표 [num_samples, 2]
        labels: 카테고리 라벨
        title: 플롯 제목
        output_path: 저장 경로
    """
    plt.figure(figsize=(16, 12))

    # 카테고리별 색상
    unique_labels = labels.unique()
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    # 산점도
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            result[mask, 0],
            result[mask, 1],
            c=[color_map[label]],
            label=label,
            alpha=0.6,
            s=50
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def plot_3d(
    result: np.ndarray,
    labels: pd.Series,
    title: str,
    output_path: str
):
    """
    3D 시각화

    Args:
        result: 3D 좌표 [num_samples, 3]
        labels: 카테고리 라벨
        title: 플롯 제목
        output_path: 저장 경로
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 카테고리별 색상
    unique_labels = labels.unique()
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    # 3D 산점도
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            result[mask, 0],
            result[mask, 1],
            result[mask, 2],
            c=[color_map[label]],
            label=label,
            alpha=0.6,
            s=50
        )

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_zlabel('Dimension 3', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize FashionCLIP embeddings with t-SNE and UMAP")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/csv/internal_products_rows.csv",
        help="Nine Oz products CSV file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/multi_domain/best_model.pt",
        help="FashionCLIP checkpoint path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of products (for testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visualizations",
        help="Output directory"
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity (default: 30)"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15)"
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        help="Generate 3D visualizations (default: 2D only)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("FashionCLIP Embedding Visualization")
    print("="*80)
    print(f"CSV file: {args.csv}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dimensions: {'2D & 3D' if args.__dict__['3d'] else '2D only'}")
    print("="*80)

    # 1. 데이터 로드
    df = load_nineoz_data(args.csv, limit=args.limit)

    # 2. 임베딩 생성기 로드
    print(f"\n[Preparing] Loading FashionCLIP model...")
    generator = FashionCLIPEmbeddingGenerator(
        checkpoint_path=args.checkpoint,
        device="cpu"  # GPU 사용 시 "cuda"
    )

    # 3. 임베딩 생성
    embeddings = generate_embeddings(generator, df)

    # 카테고리 라벨
    category_col = 'kfashion_item_category'
    if category_col not in df.columns:
        print(f"[WARNING] Column '{category_col}' not found, using 'unknown'")
        df[category_col] = 'unknown'

    labels = df[category_col].fillna('unknown')

    # 4. t-SNE 시각화
    print("\n" + "="*80)
    print("t-SNE Visualization")
    print("="*80)

    # 2D t-SNE
    tsne_2d = apply_tsne(embeddings, n_components=2, perplexity=args.perplexity)
    plot_2d(
        tsne_2d,
        labels,
        f"t-SNE 2D Visualization (FashionCLIP Embeddings, perplexity={args.perplexity})",
        f"{args.output_dir}/tsne_2d.png"
    )

    # 3D t-SNE (옵션)
    if args.__dict__['3d']:
        tsne_3d = apply_tsne(embeddings, n_components=3, perplexity=args.perplexity)
        plot_3d(
            tsne_3d,
            labels,
            f"t-SNE 3D Visualization (FashionCLIP Embeddings, perplexity={args.perplexity})",
            f"{args.output_dir}/tsne_3d.png"
        )

    # 5. UMAP 시각화
    print("\n" + "="*80)
    print("UMAP Visualization")
    print("="*80)

    # 2D UMAP
    umap_2d = apply_umap(embeddings, n_components=2, n_neighbors=args.n_neighbors)
    plot_2d(
        umap_2d,
        labels,
        f"UMAP 2D Visualization (FashionCLIP Embeddings, neighbors={args.n_neighbors})",
        f"{args.output_dir}/umap_2d.png"
    )

    # 3D UMAP (옵션)
    if args.__dict__['3d']:
        umap_3d = apply_umap(embeddings, n_components=3, n_neighbors=args.n_neighbors)
        plot_3d(
            umap_3d,
            labels,
            f"UMAP 3D Visualization (FashionCLIP Embeddings, neighbors={args.n_neighbors})",
            f"{args.output_dir}/umap_3d.png"
        )

    # 완료
    print("\n" + "="*80)
    print("✅ Visualization Complete!")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"  - tsne_2d.png")
    if args.__dict__['3d']:
        print(f"  - tsne_3d.png")
    print(f"  - umap_2d.png")
    if args.__dict__['3d']:
        print(f"  - umap_3d.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
