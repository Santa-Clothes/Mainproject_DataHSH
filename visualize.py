"""
FAISS 인덱스에서 768차원 임베딩 추출 → t-SNE / UMAP 인터랙티브 시각화
결과: visualize_tsne_umap.html  (plotly)
"""

import os
import numpy as np
import faiss
from sklearn.manifold import TSNE
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv(override=True)

# ── 설정 ─────────────────────────────────────────────────────────────
INDEX_PATH  = "data/indexes/naver.index"
IDS_PATH    = "data/indexes/naver.ids.npy"
SAMPLE_N    = 2000
RANDOM_SEED = 42
OUTPUT_HTML = "visualize_tsne_umap.html"

# 카테고리별 고대비 색상 (tab20 대신 직접 지정)
CAT_COLORS = {
    "BL": "#E63946",   # 레드
    "OP": "#F4A261",   # 오렌지
    "SK": "#2A9D8F",   # 청록
    "PT": "#457B9D",   # 파랑
    "JK": "#A8DADC",   # 하늘
    "CT": "#6D4C41",   # 갈색
    "KN": "#9C27B0",   # 보라
    "TS": "#4CAF50",   # 초록
    "JP": "#FF9800",   # 진오렌지
    "SH": "#F06292",   # 핑크
    "?":  "#B0BEC5",   # 미분류 (회색)
}

CAT_NAMES = {
    "BL": "BL 블라우스", "OP": "OP 원피스", "SK": "SK 스커트",
    "PT": "PT 팬츠",    "JK": "JK 재킷",   "CT": "CT 코트",
    "KN": "KN 니트",    "TS": "TS 티셔츠",  "JP": "JP 점프수트",
    "SH": "SH 셔츠",    "?":  "미분류",
}

# ── 1. FAISS 벡터 추출 ────────────────────────────────────────────────
print("[1] FAISS 로드 및 벡터 추출...")
index    = faiss.read_index(INDEX_PATH)
ids      = np.load(IDS_PATH, allow_pickle=True)
all_vecs = index.reconstruct_n(0, index.ntotal)
print(f"    총 {index.ntotal}개, {index.d}차원")

# ── 2. Supabase 카테고리 로드 ─────────────────────────────────────────
print("[2] 카테고리 로드...")
cat_map = {}
try:
    from supabase import create_client
    client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    table  = os.getenv("NAVER_TABLE", "naver_products")
    rows, page_size, offset = [], 1000, 0
    while True:
        resp = client.table(table).select("product_id,category_id,title,price") \
                     .range(offset, offset + page_size - 1).execute()
        if not resp.data: break
        rows.extend(resp.data)
        if len(resp.data) < page_size: break
        offset += page_size
    cat_map = {str(r["product_id"]): {
        "cat":   str(r.get("category_id", "?")),
        "title": str(r.get("title", ""))[:40],
        "price": r.get("price", 0),
    } for r in rows}
    print(f"    {len(cat_map)}개 상품 로드")
except Exception as e:
    print(f"    [WARN] {e}")

# ── 3. 샘플링 ─────────────────────────────────────────────────────────
print(f"[3] {SAMPLE_N}개 샘플링...")
rng  = np.random.default_rng(RANDOM_SEED)
sidx = rng.choice(index.ntotal, size=min(SAMPLE_N, index.ntotal), replace=False)
vecs = all_vecs[sidx].astype(np.float32)
pids = ids[sidx]

cats   = np.array([cat_map.get(str(p), {}).get("cat",   "?") for p in pids])
titles = np.array([cat_map.get(str(p), {}).get("title", "") for p in pids])
prices = np.array([cat_map.get(str(p), {}).get("price", 0)  for p in pids])

# ── 4. 차원 축소 ──────────────────────────────────────────────────────
print("[4] t-SNE...")
tsne_xy = TSNE(
    n_components=2, perplexity=40, learning_rate="auto",
    init="pca", random_state=RANDOM_SEED, n_jobs=-1
).fit_transform(vecs)

print("[5] UMAP...")
umap_xy = umap.UMAP(
    n_components=2, n_neighbors=20, min_dist=0.08,
    metric="cosine", random_state=RANDOM_SEED
).fit_transform(vecs)

# ── 5. Plotly 인터랙티브 시각화 ────────────────────────────────────────
print("[6] HTML 생성...")

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["t-SNE  (perplexity=40)", "UMAP  (cosine, n_neighbors=20)"],
    horizontal_spacing=0.06,
)

unique_cats = sorted(set(cats), key=lambda c: (c == "?", c))

for col_idx, (xy, method) in enumerate([(tsne_xy, "tsne"), (umap_xy, "umap")], start=1):
    for cat in unique_cats:
        mask  = cats == cat
        color = CAT_COLORS.get(cat, "#B0BEC5")
        name  = CAT_NAMES.get(cat, cat)
        n_pts = mask.sum()

        hover = [
            f"<b>{name}</b><br>"
            f"상품명: {t}<br>"
            f"가격: ₩{p:,}<br>"
            f"ID: {pid}"
            for t, p, pid in zip(titles[mask], prices[mask], pids[mask])
        ]

        fig.add_trace(
            go.Scatter(
                x=xy[mask, 0], y=xy[mask, 1],
                mode="markers",
                name=f"{name} ({n_pts})" if col_idx == 1 else name,
                legendgroup=cat,
                showlegend=(col_idx == 1),
                marker=dict(
                    color=color,
                    size=7,
                    opacity=0.75,
                    line=dict(width=0.4, color="white"),
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            ),
            row=1, col=col_idx,
        )

# ── 레이아웃 ──────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=f"K-Fashion 임베딩 공간 시각화 — FashionCLIP 768차원 → 2D  |  n={len(vecs):,}",
        font=dict(size=17, color="#263238"),
        x=0.5,
    ),
    paper_bgcolor="#F5F7FF",
    plot_bgcolor="#FFFFFF",
    legend=dict(
        title="<b>카테고리</b>",
        font=dict(size=11),
        itemsizing="constant",
        bordercolor="#DDE",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.9)",
    ),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Malgun Gothic, Arial",
    ),
    height=640,
    margin=dict(t=80, b=40, l=40, r=40),
)

# 각 서브플롯 배경
for col in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False, row=1, col=col)
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False, row=1, col=col)

fig.write_html(
    OUTPUT_HTML,
    include_plotlyjs="cdn",
    config={"displayModeBar": True, "scrollZoom": True},
)
print(f"\n[완료] {OUTPUT_HTML}")
print("브라우저에서 열면 줌/호버/범례 클릭으로 카테고리별 필터링 가능")
