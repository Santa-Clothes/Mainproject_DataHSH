"""
나인오즈 임베딩 시각화 비교
===========================
1단계: 기존 임베딩 (512차원, 타인 생성) - x_coord/y_coord 바로 플롯
2단계: 새 임베딩 (768차원, FashionCLIP) - 이미지 다운로드 → 임베딩 생성 → t-SNE/UMAP
3단계: 두 결과 비교
"""

import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import json
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ── 설정 ─────────────────────────────────────────────────────────────────────
PRODUCTS_CSV   = project_root / "nineounce_products_rows.csv"
VECTORS_CSV    = project_root / "nineounce_product_vectors_rows.csv"
OUTPUT_DIR     = project_root / "data" / "visualizations"
EMB_CACHE_PATH = project_root / "data" / "visualizations" / "nineounce_fashionclip_embeddings.npy"
IDS_CACHE_PATH = project_root / "data" / "visualizations" / "nineounce_fashionclip_ids.npy"

# t-SNE / UMAP 결과 캐시 경로
TSNE_768_CACHE  = OUTPUT_DIR / "cache_tsne_768.npy"
UMAP_768_CACHE  = OUTPUT_DIR / "cache_umap_768.npy"
TSNE_512_CACHE  = OUTPUT_DIR / "cache_tsne_512.npy"
STYLES_768_CACHE = OUTPUT_DIR / "cache_styles_768.npy"
STYLES_512_CACHE = OUTPUT_DIR / "cache_styles_512.npy"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# style_id → 한글 레이블
STYLE_LABELS = {
    "CAS": "캐주얼", "NAT": "내추럴", "FEM": "페미닌",
    "ETH": "에스닉", "CNT": "컨트리", "SUB": "서브컬처",
    "TRD": "트래디셔널", "GNL": "젠더리스", "SPT": "스포티", "MAN": "매니시"
}

STYLE_COLORS = {
    "CAS": "#4C72B0", "NAT": "#55A868", "FEM": "#C44E52",
    "ETH": "#8172B2", "CNT": "#937860", "SUB": "#DA8BC3",
    "TRD": "#8C8C8C", "GNL": "#CCB974", "SPT": "#64B5CD", "MAN": "#FF7F0E"
}


def get_color_and_label(style_id):
    color = STYLE_COLORS.get(style_id, "#AAAAAA")
    label = STYLE_LABELS.get(style_id, style_id)
    return color, label


# ═══════════════════════════════════════════════════════════════════════════════
# 1단계: 기존 임베딩 시각화 (x_coord, y_coord 바로 사용)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_existing_embedding():
    print("\n" + "="*70)
    print("[1단계] 기존 임베딩 시각화 (512차원, 사전 계산된 좌표)")
    print("="*70)

    vec_df  = pd.read_csv(VECTORS_CSV, usecols=['product_id', 'x_coord', 'y_coord'])
    prod_df = pd.read_csv(PRODUCTS_CSV, usecols=['product_id', 'style_id'])

    merged = vec_df.merge(prod_df, on='product_id', how='left')
    merged['style_id'] = merged['style_id'].fillna('UNK')
    print(f"  병합 결과: {len(merged)}개")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("기존 임베딩 (512차원 · 사전 계산 좌표)", fontsize=16, fontweight='bold')

    styles = merged['style_id'].unique()

    for ax_idx, title in enumerate(["사전 계산 좌표 (전체)", "사전 계산 좌표 (스타일별 색상)"]):
        ax = axes[ax_idx]
        if ax_idx == 0:
            ax.scatter(merged['x_coord'], merged['y_coord'],
                       alpha=0.4, s=8, c='steelblue')
        else:
            for style in styles:
                mask = merged['style_id'] == style
                color, label = get_color_and_label(style)
                ax.scatter(merged.loc[mask, 'x_coord'],
                           merged.loc[mask, 'y_coord'],
                           alpha=0.5, s=8, c=color, label=f"{label}({mask.sum()})")
            ax.legend(loc='upper right', fontsize=8, markerscale=2)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x_coord")
        ax.set_ylabel("y_coord")
        ax.grid(True, alpha=0.3)

    out = OUTPUT_DIR / "01_existing_embedding.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [저장] {out}")

    # vec_df의 embedding 컬럼은 여기서 안 쓰므로 바로 해제
    del vec_df
    gc.collect()

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 2단계: FashionCLIP 임베딩 생성 (메모리 최적화: 미리 할당 배열 사용)
# ═══════════════════════════════════════════════════════════════════════════════
API_URL = "http://localhost:8001/embed"


def generate_fashionclip_embeddings():
    print("\n" + "="*70)
    print("[2단계] FashionCLIP 임베딩 생성 (768차원) - API 서버 활용")
    print("="*70)

    # 캐시 확인
    if EMB_CACHE_PATH.exists() and IDS_CACHE_PATH.exists():
        print("  [캐시 발견] 저장된 임베딩 로드...")
        embeddings = np.load(str(EMB_CACHE_PATH))
        product_ids = np.load(str(IDS_CACHE_PATH), allow_pickle=True).tolist()
        print(f"  [OK] {len(product_ids)}개 임베딩 로드 완료 (shape: {embeddings.shape})")
        return embeddings, product_ids

    # API 서버 연결 확인
    try:
        health = requests.get("http://localhost:8001/health", timeout=5).json()
        if not health.get("model_loaded"):
            raise RuntimeError("API 서버 모델 미로드")
        print(f"  [OK] API 서버 연결 확인 (model_loaded: {health['model_loaded']})")
    except Exception as e:
        raise RuntimeError(f"API 서버에 연결할 수 없습니다: {e}\n  먼저 서버를 실행하세요: uvicorn api.search_api:app --port 8001")

    prod_df = pd.read_csv(PRODUCTS_CSV, usecols=['product_id', 'image_url'])
    n = len(prod_df)
    print(f"  대상: {n}개 이미지")
    print(f"  API: {API_URL}")

    embeddings = np.zeros((n, 768), dtype=np.float32)
    product_ids = []
    valid_mask = np.zeros(n, dtype=bool)
    error_count = 0

    session = requests.Session()

    for i, (_, row) in enumerate(tqdm(prod_df.iterrows(), total=n, desc="임베딩 생성")):
        try:
            # 이미지 다운로드
            img_resp = session.get(row['image_url'], timeout=10)
            if img_resp.status_code != 200:
                error_count += 1
                continue

            # API로 임베딩 요청
            img_bytes = img_resp.content
            api_resp = session.post(
                API_URL,
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                timeout=30
            )
            if api_resp.status_code != 200:
                error_count += 1
                continue

            emb = np.array(api_resp.json()["embedding"], dtype=np.float32)
            embeddings[i] = emb
            valid_mask[i] = True
            product_ids.append(str(row['product_id']))

            del img_bytes, img_resp, api_resp
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\n  [오류] {row['product_id']}: {e}")

        if (i + 1) % 100 == 0:
            gc.collect()

    session.close()

    # 유효한 행만 남기기
    embeddings = embeddings[valid_mask]
    print(f"\n  [완료] 성공: {len(product_ids)}개 / 실패: {error_count}개")
    print(f"  임베딩 shape: {embeddings.shape}")

    # 캐시 저장
    np.save(str(EMB_CACHE_PATH), embeddings)
    np.save(str(IDS_CACHE_PATH), np.array(product_ids))
    print(f"  [캐시 저장] {EMB_CACHE_PATH}")

    del prod_df
    gc.collect()

    return embeddings, product_ids


# ═══════════════════════════════════════════════════════════════════════════════
# 512차원 임베딩 파싱 (청크 단위로 메모리 최적화)
# ═══════════════════════════════════════════════════════════════════════════════
def _load_512dim_embeddings(chunk_size=500):
    """nineounce_product_vectors_rows.csv 의 embedding 컬럼을 청크 단위로 파싱"""
    print("  기존 512차원 임베딩 로드 중 (청크 단위)...")

    # 필요한 컬럼만 읽기
    vec_df = pd.read_csv(VECTORS_CSV, usecols=['product_id', 'embedding'])
    prod_df = pd.read_csv(PRODUCTS_CSV, usecols=['product_id', 'style_id'])
    merged = vec_df.merge(prod_df, on='product_id', how='left')
    merged['style_id'] = merged['style_id'].fillna('UNK')
    del vec_df, prod_df
    gc.collect()

    n = len(merged)
    embs_512 = np.empty((n, 512), dtype=np.float32)

    for start in tqdm(range(0, n, chunk_size), desc="512차원 파싱"):
        end = min(start + chunk_size, n)
        for j, emb_str in enumerate(merged['embedding'].iloc[start:end]):
            embs_512[start + j] = json.loads(emb_str)

    styles_512 = merged['style_id'].tolist()

    # embedding 컬럼 해제 (큰 문자열 데이터)
    del merged
    gc.collect()

    print(f"  [OK] 512차원 임베딩 로드 완료 (shape: {embs_512.shape})")
    return embs_512, styles_512


# ═══════════════════════════════════════════════════════════════════════════════
# 3단계: t-SNE + UMAP 실행 및 비교 플롯
# ═══════════════════════════════════════════════════════════════════════════════
def run_tsne_umap_and_compare(embeddings, product_ids, existing_merged):
    print("\n" + "="*70)
    print("[3단계] t-SNE + UMAP 실행")
    print("="*70)

    prod_df = pd.read_csv(PRODUCTS_CSV, usecols=['product_id', 'style_id'])
    id_to_style = dict(zip(prod_df['product_id'].astype(str), prod_df['style_id']))
    del prod_df
    gc.collect()

    # styles_768 캐시
    if STYLES_768_CACHE.exists():
        print("  [캐시] styles_768 로드...")
        styles_new = np.load(str(STYLES_768_CACHE), allow_pickle=True).tolist()
    else:
        styles_new = [id_to_style.get(pid, 'UNK') for pid in product_ids]
        np.save(str(STYLES_768_CACHE), np.array(styles_new))

    # t-SNE (768차원) - 캐시 확인
    if TSNE_768_CACHE.exists():
        print("  [캐시] FashionCLIP t-SNE 로드...")
        tsne_coords = np.load(str(TSNE_768_CACHE))
        print(f"  [OK] shape: {tsne_coords.shape}")
    else:
        print("  FashionCLIP t-SNE 실행 중... (2~5분 소요)")
        tsne = TSNE(n_components=2, perplexity=40, n_iter=500, random_state=42, verbose=0)
        tsne_coords = tsne.fit_transform(embeddings)
        np.save(str(TSNE_768_CACHE), tsne_coords)
        print(f"  [OK] t-SNE 완료 → 캐시 저장 (shape: {tsne_coords.shape})")
        del tsne
    gc.collect()

    # UMAP (768차원) - 캐시 확인
    if UMAP_768_CACHE.exists():
        print("  [캐시] FashionCLIP UMAP 로드...")
        umap_coords = np.load(str(UMAP_768_CACHE))
        print(f"  [OK] shape: {umap_coords.shape}")
    else:
        print("  UMAP 실행 중...")
        from umap import UMAP
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
        umap_coords = reducer.fit_transform(embeddings)
        np.save(str(UMAP_768_CACHE), umap_coords)
        print(f"  [OK] UMAP 완료 → 캐시 저장 (shape: {umap_coords.shape})")
        del reducer
    gc.collect()

    # 512차원 임베딩 로드 및 t-SNE - 캐시 확인
    if TSNE_512_CACHE.exists() and STYLES_512_CACHE.exists():
        print("  [캐시] 기존 512차원 t-SNE 로드...")
        tsne_512 = np.load(str(TSNE_512_CACHE))
        styles_512 = np.load(str(STYLES_512_CACHE), allow_pickle=True).tolist()
        print(f"  [OK] shape: {tsne_512.shape}")
    else:
        embs_512, styles_512 = _load_512dim_embeddings()
        print("  기존 512차원 t-SNE 실행 중...")
        tsne_512_model = TSNE(n_components=2, perplexity=40, n_iter=500, random_state=42)
        tsne_512 = tsne_512_model.fit_transform(embs_512)
        np.save(str(TSNE_512_CACHE), tsne_512)
        np.save(str(STYLES_512_CACHE), np.array(styles_512))
        print(f"  [OK] 512차원 t-SNE 완료 → 캐시 저장")
        del embs_512, tsne_512_model
    gc.collect()

    # ── 비교 플롯 (2행 3열) ──────────────────────────────────────────────────
    all_styles = sorted(set(styles_new) | set(existing_merged['style_id'].tolist()))

    def scatter_by_style(ax, x, y, style_list, title):
        x_arr = np.array(x) if not isinstance(x, np.ndarray) else x
        y_arr = np.array(y) if not isinstance(y, np.ndarray) else y
        sl_arr = np.array(style_list)
        for style in all_styles:
            mask = sl_arr == style
            if not mask.any():
                continue
            color, label = get_color_and_label(style)
            ax.scatter(x_arr[mask], y_arr[mask], alpha=0.5, s=8, c=color,
                       label=f"{label}({mask.sum()})")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, markerscale=2)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("나인오즈 임베딩 시각화 비교\n"
                 "상단: 기존(512차원·사전계산) | 하단: FashionCLIP(768차원·신규)",
                 fontsize=14, fontweight='bold')

    ex_x = existing_merged['x_coord'].values
    ex_y = existing_merged['y_coord'].values
    ex_styles = existing_merged['style_id'].tolist()

    axes[0, 0].scatter(ex_x, ex_y, alpha=0.4, s=6, c='steelblue')
    axes[0, 0].set_title("기존 · 사전계산 좌표 (전체)", fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    scatter_by_style(axes[0, 1], ex_x, ex_y, ex_styles, "기존 · 사전계산 (스타일별)")
    scatter_by_style(axes[0, 2], tsne_512[:, 0], tsne_512[:, 1], styles_512,
                     "기존 · t-SNE 재계산 (512차원)")

    axes[1, 0].scatter(tsne_coords[:, 0], tsne_coords[:, 1], alpha=0.4, s=6, c='tomato')
    axes[1, 0].set_title("FashionCLIP · t-SNE 전체", fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    scatter_by_style(axes[1, 1], tsne_coords[:, 0], tsne_coords[:, 1], styles_new,
                     "FashionCLIP · t-SNE (스타일별)")
    scatter_by_style(axes[1, 2], umap_coords[:, 0], umap_coords[:, 1], styles_new,
                     "FashionCLIP · UMAP (스타일별)")

    out = OUTPUT_DIR / "02_embedding_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [저장] {out}")

    # 중간 데이터 해제 후 UMAP 단독 플롯
    del tsne_512, styles_512
    gc.collect()

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
    fig2.suptitle("FashionCLIP UMAP vs 기존 사전계산 좌표", fontsize=14, fontweight='bold')

    scatter_by_style(axes2[0], umap_coords[:, 0], umap_coords[:, 1], styles_new,
                     "FashionCLIP UMAP (768차원 · 신규)")
    scatter_by_style(axes2[1], ex_x, ex_y, ex_styles,
                     "기존 사전계산 좌표 (512차원)")

    out2 = OUTPUT_DIR / "03_umap_comparison.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [저장] {out2}")

    return tsne_coords, umap_coords, styles_new


# ═══════════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*70)
    print("나인오즈 임베딩 시각화 비교")
    print("="*70)
    print(f"출력 폴더: {OUTPUT_DIR}")

    # 1단계: 기존 좌표 시각화
    existing_merged = plot_existing_embedding()
    gc.collect()

    # 2단계: FashionCLIP 임베딩 생성 (캐시 있으면 스킵)
    embeddings, product_ids = generate_fashionclip_embeddings()
    gc.collect()

    # 3단계: t-SNE/UMAP 비교
    tsne_coords, umap_coords, styles_new = run_tsne_umap_and_compare(
        embeddings, product_ids, existing_merged
    )

    print("\n" + "="*70)
    print("완료!")
    print(f"  01_existing_embedding.png  — 기존 512차원 사전계산 좌표")
    print(f"  02_embedding_comparison.png — 전체 비교 (2행 3열)")
    print(f"  03_umap_comparison.png      — UMAP 단독 비교")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print("="*70)
