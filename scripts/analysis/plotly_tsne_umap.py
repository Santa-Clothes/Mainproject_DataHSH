"""
나인오즈 임베딩 Plotly 인터랙티브 시각화
==========================================
캐시된 t-SNE / UMAP 좌표로 인터랙티브 HTML 생성

출력:
  data/visualizations/interactive_style.html     <- 스타일별 색상
  data/visualizations/interactive_category.html  <- 카테고리별 색상

실행:
  python scripts/analysis/plotly_tsne_umap.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT     = Path(__file__).parent.parent.parent
OUT_DIR  = ROOT / "data" / "visualizations"

# ── 캐시 파일 ──────────────────────────────────────────────────────────────
CACHE = {
    "precomp":   None,                           # x_coord / y_coord (rows.csv)
    "tsne_512":  OUT_DIR / "cache_tsne_512.npy",
    "umap_512":  OUT_DIR / "cache_umap_512.npy",
    "tsne_768":  OUT_DIR / "cache_tsne_768.npy",
    "umap_768":  OUT_DIR / "cache_umap_768.npy",
}

# ── 스타일 / 카테고리 색상표 ────────────────────────────────────────────────
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
    "CA": ("#e6194b", "CA 캐주얼"),
    "JP": ("#3cb44b", "JP 점프수트"),
    "KT": ("#ffe119", "KT 니트"),
    "OP": ("#4363d8", "OP 원피스"),
    "PT": ("#f58231", "PT 팬츠"),
    "TS": ("#911eb4", "TS 티셔츠"),
    "BL": ("#42d4f4", "BL 블라우스"),
    "CT": ("#f032e6", "CT 코트"),
    "JK": ("#bfef45", "JK 재킷"),
    "SK": ("#fabed4", "SK 스커트"),
    "VT": ("#469990", "VT 베스트"),
    "DP": ("#dcbeff", "DP 드레스"),
    "WS": ("#9A6324", "WS 와이드"),
    "ST": ("#800000", "ST 스트랩"),
    "LG": ("#808000", "LG 레깅스"),
}


# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    """캐시 + CSV 로드"""
    # 레이블 (공통)
    rows_csv = ROOT / "nineounce_product_vectors_rows.csv"
    prod_csv = ROOT / "nineounce_products_rows.csv"

    vec_df  = pd.read_csv(rows_csv, usecols=["product_id", "x_coord", "y_coord"])
    prod_df = pd.read_csv(prod_csv, usecols=["product_id", "product_name",
                                              "style_id", "category_id"])
    df = vec_df.merge(prod_df, on="product_id", how="left")
    df["style_id"]    = df["style_id"].fillna("UNK")
    df["category_id"] = df["category_id"].fillna("UNK")
    df["product_name"] = df["product_name"].fillna("")

    product_ids  = df["product_id"].tolist()
    product_names = df["product_name"].tolist()
    styles       = df["style_id"].tolist()
    categories   = df["category_id"].tolist()
    precomp      = np.column_stack([df["x_coord"].values, df["y_coord"].values])

    coords = {
        "precomp":  precomp,
        "tsne_512": np.load(str(CACHE["tsne_512"])),
        "umap_512": np.load(str(CACHE["umap_512"])),
        "tsne_768": np.load(str(CACHE["tsne_768"])),
        "umap_768": np.load(str(CACHE["umap_768"])),
    }

    print(f"[OK] 데이터 로드 완료: {len(product_ids)}개")
    return coords, product_ids, product_names, styles, categories


# ══════════════════════════════════════════════════════════════════════════════
def build_figure(
    coords: dict,
    product_ids: list,
    product_names: list,
    label_list: list,
    meta: dict,
    title: str,
) -> go.Figure:
    """
    5개 서브플롯 (precomp | 512 t-SNE | 512 UMAP | 768 t-SNE | 768 UMAP)
    label_list 에 따라 색상 구분
    """

    subplot_specs = [
        {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"},
        {"type": "scatter"}, {"type": "scatter"},
    ]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "사전계산 x_coord/y_coord",
            "512차원 t-SNE",
            "512차원 UMAP",
            "768차원 t-SNE (FashionCLIP)",
            "768차원 UMAP (FashionCLIP)",
            "",          # 빈 칸
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    # 서브플롯 위치 매핑
    plot_positions = [
        ("precomp",  1, 1),
        ("tsne_512", 1, 2),
        ("umap_512", 1, 3),
        ("tsne_768", 2, 1),
        ("umap_768", 2, 2),
    ]

    labels_arr = np.array(label_list)
    unique_labels = sorted(set(label_list))

    # 첫 번째 서브플롯에서만 legend 표시 (중복 방지)
    shown_in_legend = set()

    for coord_key, row, col in plot_positions:
        xy = coords[coord_key]
        for lbl in unique_labels:
            mask = labels_arr == lbl
            color, name = meta.get(lbl, ("#AAAAAA", lbl))
            show_legend = lbl not in shown_in_legend

            hover = [
                f"<b>{name}</b><br>"
                f"ID: {pid}<br>"
                f"상품: {pname[:30]}"
                for pid, pname in zip(
                    np.array(product_ids)[mask],
                    np.array(product_names)[mask],
                )
            ]

            fig.add_trace(
                go.Scatter(
                    x=xy[mask, 0],
                    y=xy[mask, 1],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=4,
                        opacity=0.65,
                        line=dict(width=0),
                    ),
                    name=name,
                    legendgroup=lbl,
                    showlegend=show_legend,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover,
                ),
                row=row, col=col,
            )
            shown_in_legend.add(lbl)

    # 축 숨기기 (2D 임베딩이라 숫자 의미 없음)
    for i in range(1, 6):
        row = 1 if i <= 3 else 2
        col = i if i <= 3 else i - 3
        fig.update_xaxes(showticklabels=False, showgrid=False,
                         zeroline=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, showgrid=False,
                         zeroline=False, row=row, col=col)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18),
            x=0.5,
        ),
        height=900,
        width=1600,
        legend=dict(
            title="레이블",
            itemsizing="constant",
            tracegroupgap=2,
            font=dict(size=11),
        ),
        paper_bgcolor="white",
        plot_bgcolor="#f9f9f9",
        font=dict(family="Malgun Gothic, Arial", size=12),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Plotly 인터랙티브 시각화 생성")
    print("=" * 60)

    # 캐시 파일 존재 확인
    missing = [k for k, p in CACHE.items() if p and not Path(p).exists()]
    if missing:
        print(f"[ERROR] 캐시 파일 없음: {missing}")
        print("  먼저 compare_embeddings_tsne_umap.py 를 실행하세요.")
        return

    coords, pids, pnames, styles, cats = load_data()

    # ── 스타일별 ─────────────────────────────────────────────────────────────
    print("[1/2] 스타일별 HTML 생성 중...")
    fig_style = build_figure(
        coords, pids, pnames, styles, STYLE_META,
        "나인오즈 임베딩 비교 — 스타일별 (CAS/FEM/ETH...)",
    )
    out_style = OUT_DIR / "interactive_style.html"
    fig_style.write_html(str(out_style), include_plotlyjs="cdn")
    print(f"  [저장] {out_style}")

    # ── 카테고리별 ────────────────────────────────────────────────────────────
    print("[2/2] 카테고리별 HTML 생성 중...")
    fig_cat = build_figure(
        coords, pids, pnames, cats, CAT_META,
        "나인오즈 임베딩 비교 — 카테고리별 (BL/OP/SK...)",
    )
    out_cat = OUT_DIR / "interactive_category.html"
    fig_cat.write_html(str(out_cat), include_plotlyjs="cdn")
    print(f"  [저장] {out_cat}")

    print("\n완료!")
    print(f"  {out_style.name}  <- 스타일별 색상")
    print(f"  {out_cat.name}   <- 카테고리별 색상")
    print(f"폴더: {OUT_DIR}")
    print("브라우저에서 HTML 파일을 열면 줌/호버/범례 클릭 가능합니다.")


if __name__ == "__main__":
    main()
