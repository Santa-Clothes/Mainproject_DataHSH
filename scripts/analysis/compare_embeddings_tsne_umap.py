"""
나인오즈 임베딩 비교 시각화: 512차원 vs 768차원
================================================

nineounce_product_vectors_rows.csv (512차원, 사전계산 x_coord/y_coord 포함)
nineounce_product_vectors.csv      (768차원, FashionCLIP 신규)

두 임베딩에 대해 t-SNE / UMAP 을 각각 계산하고 나란히 비교합니다.

캐시 지원: data/visualizations/cache_*.npy 에 중간 결과 저장
           → 중단 후 재시작 시 계산 건너뜀

실행:
    python scripts/analysis/compare_embeddings_tsne_umap.py
"""

import gc
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # GUI 없이 저장만
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 경로 설정 ───────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent.parent
ROWS_CSV     = ROOT / "nineounce_product_vectors_rows.csv"   # 512차원 + x_coord/y_coord
NEW_CSV      = ROOT / "nineounce_product_vectors.csv"        # 768차원 FashionCLIP
PRODUCTS_CSV = ROOT / "nineounce_products_rows.csv"

OUT_DIR  = ROOT / "data" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 캐시 경로
CACHE = {
    "tsne_512":  OUT_DIR / "cache_tsne_512.npy",
    "umap_512":  OUT_DIR / "cache_umap_512.npy",
    "tsne_768":  OUT_DIR / "cache_tsne_768.npy",
    "umap_768":  OUT_DIR / "cache_umap_768.npy",
}

# ── 스타일 / 카테고리 색상 ───────────────────────────────────────────────────
STYLE_META = {
    "CAS": ("#4C72B0", "캐주얼"),
    "NAT": ("#55A868", "내추럴"),
    "FEM": ("#C44E52", "페미닌"),
    "ETH": ("#8172B2", "에스닉"),
    "CNT": ("#937860", "컨트리"),
    "SUB": ("#DA8BC3", "서브컬처"),
    "TRD": ("#8C8C8C", "트래디셔널"),
    "GNL": ("#CCB974", "젠더리스"),
    "SPT": ("#64B5CD", "스포티"),
    "MAN": ("#FF7F0E", "매니시"),
}

CAT_META = {
    "CA": ("#e6194b", "CA·캐주얼"),
    "JP": ("#3cb44b", "JP·점프수트"),
    "KT": ("#ffe119", "KT·니트"),
    "OP": ("#4363d8", "OP·원피스"),
    "PT": ("#f58231", "PT·팬츠"),
    "TS": ("#911eb4", "TS·티셔츠"),
    "BL": ("#42d4f4", "BL·블라우스"),
    "CT": ("#f032e6", "CT·코트"),
    "JK": ("#bfef45", "JK·재킷"),
    "SK": ("#fabed4", "SK·스커트"),
    "VT": ("#469990", "VT·베스트"),
    "DP": ("#dcbeff", "DP·드레스"),
    "WS": ("#9A6324", "WS·와이드"),
    "ST": ("#800000", "ST·스트랩"),
    "LG": ("#808000", "LG·레깅스"),
}

def _color(meta: dict, key: str):
    return meta.get(key, ("#AAAAAA", key))


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_embeddings(csv_path: Path, label: str) -> tuple[np.ndarray, list, list]:
    """
    CSV → (embeddings[N, D], product_ids, styles)
    x_coord/y_coord 컬럼은 무시하고 embedding 만 파싱
    """
    print(f"\n[로드] {csv_path.name}")
    df = pd.read_csv(csv_path, usecols=["product_id", "embedding"])

    prod_df = pd.read_csv(PRODUCTS_CSV, usecols=["product_id", "style_id", "category_id"])
    df = df.merge(prod_df, on="product_id", how="left")
    df["style_id"]    = df["style_id"].fillna("UNK")
    df["category_id"] = df["category_id"].fillna("UNK")

    n = len(df)
    # 첫 행으로 차원 파악
    dim = len(json.loads(df["embedding"].iloc[0]))
    print(f"  행 수: {n} | 임베딩 차원: {dim}")

    embs = np.empty((n, dim), dtype=np.float32)
    print(f"  임베딩 파싱 중...")
    for i, row in enumerate(tqdm(df["embedding"], desc="  파싱", leave=False)):
        embs[i] = json.loads(row)

    styles     = df["style_id"].tolist()
    categories = df["category_id"].tolist()
    product_ids = df["product_id"].tolist()

    # 불필요한 DataFrame 해제
    del df, prod_df
    gc.collect()

    print(f"  [OK] shape={embs.shape}")
    return embs, styles, categories


# ══════════════════════════════════════════════════════════════════════════════
# 2. t-SNE / UMAP 계산 (캐시 지원)
# ══════════════════════════════════════════════════════════════════════════════
def compute_tsne(embs: np.ndarray, cache_path: Path, tag: str) -> np.ndarray:
    if cache_path.exists():
        print(f"  [캐시] t-SNE {tag} 로드: {cache_path.name}")
        coords = np.load(str(cache_path))
        print(f"         shape={coords.shape}")
        return coords

    print(f"  t-SNE {tag} 계산 중... (perplexity=40, max_iter=800, 약 3~6분)")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=800,
        random_state=42,
        verbose=1,
    )
    coords = tsne.fit_transform(embs)
    np.save(str(cache_path), coords)
    print(f"  [OK] t-SNE {tag} 완료 → 캐시 저장")
    del tsne
    gc.collect()
    return coords


def compute_umap(embs: np.ndarray, cache_path: Path, tag: str) -> np.ndarray:
    if cache_path.exists():
        print(f"  [캐시] UMAP {tag} 로드: {cache_path.name}")
        coords = np.load(str(cache_path))
        print(f"         shape={coords.shape}")
        return coords

    print(f"  UMAP {tag} 계산 중... (n_neighbors=30, min_dist=0.1, 약 1~3분)")
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn 미설치: pip install umap-learn")

    reducer = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        verbose=True,
    )
    coords = reducer.fit_transform(embs)
    np.save(str(cache_path), coords)
    print(f"  [OK] UMAP {tag} 완료 → 캐시 저장")
    del reducer
    gc.collect()
    return coords


# ══════════════════════════════════════════════════════════════════════════════
# 3. 플롯 유틸
# ══════════════════════════════════════════════════════════════════════════════
def scatter_by_label(ax, coords: np.ndarray, labels: list, meta: dict, title: str):
    """coords[N,2], labels[N] → scatter with legend"""
    labels_arr = np.array(labels)
    unique     = sorted(set(labels))

    for key in unique:
        mask = labels_arr == key
        color, name = _color(meta, key)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            alpha=0.5, s=6, c=color,
            label=f"{name} ({mask.sum()})",
        )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.legend(
        loc="upper right",
        fontsize=6,
        markerscale=2.5,
        framealpha=0.7,
        ncol=1,
    )
    ax.grid(True, alpha=0.25)
    ax.set_xticks([])
    ax.set_yticks([])


# ══════════════════════════════════════════════════════════════════════════════
# 4. 메인 비교 플롯
# ══════════════════════════════════════════════════════════════════════════════
def plot_comparison(
    coords_512_tsne, coords_512_umap,
    coords_768_tsne, coords_768_umap,
    styles_512, styles_768,
    cats_512,   cats_768,
    precomp_x, precomp_y,
):
    """
    3행 4열 비교 플롯
    열: 512 t-SNE | 512 UMAP | 768 t-SNE | 768 UMAP
    행: 단색 | 스타일별 색상 | 카테고리별 색상
    """
    print("\n[플롯] 3행 × 4열 비교 플롯 생성 중...")

    fig, axes = plt.subplots(3, 4, figsize=(28, 18))
    fig.suptitle(
        "나인오즈 임베딩 비교 시각화\n"
        "열: 512차원(구버전) t-SNE | 512차원 UMAP | 768차원(FashionCLIP) t-SNE | 768차원 UMAP\n"
        "행: 전체 | 스타일별 | 카테고리별",
        fontsize=13, fontweight="bold", y=1.01,
    )

    col_data = [
        (coords_512_tsne, styles_512, cats_512, "512차원  t-SNE"),
        (coords_512_umap, styles_512, cats_512, "512차원  UMAP"),
        (coords_768_tsne, styles_768, cats_768, "768차원  t-SNE"),
        (coords_768_umap, styles_768, cats_768, "768차원  UMAP"),
    ]

    for col, (coords, styles, cats, tag) in enumerate(col_data):
        # 행0: 단색
        ax = axes[0, col]
        color = "#4C72B0" if "512" in tag else "#C44E52"
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.35, s=5, c=color)
        ax.set_title(f"{tag}\n전체", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_xticks([]); ax.set_yticks([])

        # 행1: 스타일별
        scatter_by_label(axes[1, col], coords, styles, STYLE_META,
                         f"{tag}\n스타일별")

        # 행2: 카테고리별
        scatter_by_label(axes[2, col], coords, cats, CAT_META,
                         f"{tag}\n카테고리별")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "compare_512_768_tsne_umap.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [저장] {out}")

    # ── 사전계산 좌표 vs 신규 768 UMAP 단독 비교 ──────────────────────────
    print("[플롯] 사전계산 좌표 vs 768 UMAP 비교 플롯...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 14))
    fig2.suptitle(
        "사전계산 x_coord/y_coord (512차원)  vs  신규 FashionCLIP UMAP (768차원)",
        fontsize=13, fontweight="bold",
    )

    precomp = np.column_stack([precomp_x, precomp_y])
    pairs = [
        (axes2[0, 0], precomp,         styles_512, STYLE_META, "사전계산 좌표 — 스타일별"),
        (axes2[0, 1], coords_768_umap, styles_768, STYLE_META, "768차원 UMAP — 스타일별"),
        (axes2[1, 0], precomp,         cats_512,   CAT_META,   "사전계산 좌표 — 카테고리별"),
        (axes2[1, 1], coords_768_umap, cats_768,   CAT_META,   "768차원 UMAP — 카테고리별"),
    ]
    for ax, coords, labels, meta, title in pairs:
        scatter_by_label(ax, coords, labels, meta, title)

    plt.tight_layout()
    out2 = OUT_DIR / "precomp_vs_768_umap.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [저장] {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 엔트리포인트
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("나인오즈 임베딩 비교 시각화: 512차원 vs 768차원 FashionCLIP")
    print("=" * 70)
    print(f"출력 폴더: {OUT_DIR}")

    # ── 512차원 로드 ─────────────────────────────────────────────────────────
    print("\n[1/4] 512차원 임베딩 로드")
    embs_512, styles_512, cats_512 = load_embeddings(ROWS_CSV, "512dim")

    # 사전계산 좌표도 별도로 읽어둠 (행2와 비교용)
    raw_rows = pd.read_csv(ROWS_CSV, usecols=["product_id", "x_coord", "y_coord"])
    prod_df  = pd.read_csv(PRODUCTS_CSV, usecols=["product_id", "style_id", "category_id"])
    raw_rows = raw_rows.merge(prod_df, on="product_id", how="left")
    precomp_x = raw_rows["x_coord"].values
    precomp_y = raw_rows["y_coord"].values
    del raw_rows, prod_df
    gc.collect()

    # ── 512차원 t-SNE / UMAP ─────────────────────────────────────────────────
    print("\n[2/4] 512차원 t-SNE + UMAP")
    coords_512_tsne = compute_tsne(embs_512, CACHE["tsne_512"], "512차원")
    coords_512_umap = compute_umap(embs_512, CACHE["umap_512"], "512차원")
    del embs_512
    gc.collect()

    # ── 768차원 로드 ─────────────────────────────────────────────────────────
    print("\n[3/4] 768차원 임베딩 로드")
    embs_768, styles_768, cats_768 = load_embeddings(NEW_CSV, "768dim")

    # ── 768차원 t-SNE / UMAP ─────────────────────────────────────────────────
    print("\n[4/4] 768차원 t-SNE + UMAP")
    coords_768_tsne = compute_tsne(embs_768, CACHE["tsne_768"], "768차원")
    coords_768_umap = compute_umap(embs_768, CACHE["umap_768"], "768차원")
    del embs_768
    gc.collect()

    # ── 플롯 생성 ────────────────────────────────────────────────────────────
    plot_comparison(
        coords_512_tsne, coords_512_umap,
        coords_768_tsne, coords_768_umap,
        styles_512, styles_768,
        cats_512,   cats_768,
        precomp_x,  precomp_y,
    )

    print("\n" + "=" * 70)
    print("완료!")
    print(f"  compare_512_768_tsne_umap.png  : 3행x4열 전체 비교")
    print(f"  precomp_vs_768_umap.png        : 사전계산 vs 768 UMAP 단독 비교")
    print(f"출력 폴더: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
