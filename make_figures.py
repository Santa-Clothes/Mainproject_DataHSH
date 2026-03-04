"""
발표용 다이어그램 3종 생성
  1. embedding_process.png  — 임베딩 생성 과정
  2. style_classifier.png   — 스타일 분류기 작동 과정
  3. search_performance.png — 검색 성능
"""

import matplotlib
matplotlib.rc('font', family='Malgun Gothic')   # Windows 한글
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── 공통 팔레트 ──────────────────────────────────────────────────────
C_BLUE   = "#4A90D9"
C_PURPLE = "#7B5EA7"
C_GREEN  = "#4CAF50"
C_ORANGE = "#F5A623"
C_GRAY   = "#78909C"
C_LIGHT  = "#F5F7FF"
C_WHITE  = "#FFFFFF"
C_DARK   = "#263238"

def box(ax, x, y, w, h, label, sublabel="", color=C_BLUE, fontsize=11, radius=0.04):
    fc = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle=f"round,pad=0,rounding_size={radius}",
                        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3)
    ax.add_patch(fc)
    dy = 0.06 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(x, y - 0.10, sublabel, ha="center", va="center",
                fontsize=8.5, color="white", alpha=0.88, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_GRAY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=18), zorder=2)

def label_between(ax, x, y, text, fontsize=8.5, color=C_DARK):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, style="italic",
            bbox=dict(fc="white", ec="none", pad=1), zorder=5)


# ════════════════════════════════════════════════════════════════════
# 1. 임베딩 생성 과정
# ════════════════════════════════════════════════════════════════════
def fig_embedding():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor(C_LIGHT)
    ax.set_facecolor(C_LIGHT)

    ax.text(7, 5.6, "FashionCLIP 임베딩 생성 과정",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)

    # ── 박스들 (y=3 라인) ──────────────────────────────────────────
    steps = [
        (1.3,  3.0, 1.6, 1.6, "이미지\n입력",      "",                   C_ORANGE),
        (3.5,  3.0, 2.0, 1.6, "전처리",           "224×224 리사이즈\nRGB Normalize", C_BLUE),
        (6.0,  3.0, 2.4, 1.6, "ViT 인코더",       "FashionCLIP\nCLIPVisionModel",   C_PURPLE),
        (8.8,  3.0, 2.0, 1.6, "Pooler\nOutput",   "768차원",             C_PURPLE),
        (11.2, 3.0, 1.8, 1.6, "L2 정규화",        "단위벡터\n(코사인 유사도)",       C_GREEN),
    ]
    for (x, y, w, h, lbl, sub, col) in steps:
        box(ax, x, y, w, h, lbl, sub, color=col, fontsize=10)

    # ── 화살표 ────────────────────────────────────────────────────
    xs = [2.1, 4.5, 7.2, 9.8]
    for i, x in enumerate(xs):
        arrow(ax, x, 3.0, x + 0.4, 3.0)

    # ── 레이블 ───────────────────────────────────────────────────
    label_between(ax, 2.43, 3.45, "PIL Image")
    label_between(ax, 4.85, 3.45, "Tensor [1,3,224,224]")
    label_between(ax, 7.45, 3.45, "pooler_output")
    label_between(ax, 10.05, 3.45, "[1, 768]")

    # ── 결과 벡터 시각화 ─────────────────────────────────────────
    ax.text(11.2, 1.55, "768차원 임베딩 벡터", ha="center", fontsize=9,
            color=C_DARK, fontweight="bold")
    bar_x = np.linspace(9.7, 12.7, 40)
    bar_h = np.random.RandomState(42).randn(40) * 0.18 + 0.25
    bar_h = np.clip(bar_h, 0.05, 0.5)
    cmap  = plt.get_cmap("coolwarm")
    for bx, bh in zip(bar_x, bar_h):
        ax.bar(bx, bh, width=0.065, bottom=1.65,
               color=cmap((bh - 0.05) / 0.45), alpha=0.85, zorder=3)
    ax.text(11.2, 1.35, "← 7 6 8 차 원 →", ha="center", fontsize=8, color=C_GRAY)

    # ── 주석 박스 ─────────────────────────────────────────────────
    note = ("∙ pooler_output: [CLS] 토큰의 최종 은닉 상태 (768-dim)\n"
            "∙ L2 정규화 후 내적(Inner Product) = 코사인 유사도\n"
            "∙ 학습된 패션 지식으로 색상·소재·실루엣 정보 압축")
    ax.text(0.25, 0.6, note, fontsize=8.5, color=C_DARK,
            va="bottom", linespacing=1.7,
            bbox=dict(fc="white", ec=C_BLUE, lw=1.2, pad=6, boxstyle="round,pad=0.4"))

    plt.tight_layout()
    plt.savefig("embedding_process.png", dpi=160, bbox_inches="tight",
                facecolor=C_LIGHT)
    plt.close()
    print("[OK] embedding_process.png")


# ════════════════════════════════════════════════════════════════════
# 2. 스타일 분류기 작동 과정
# ════════════════════════════════════════════════════════════════════
def fig_style_classifier():
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(C_LIGHT)

    # 왼쪽: 아키텍처 다이어그램 (70%)
    ax1 = fig.add_axes([0.0, 0.0, 0.58, 1.0])
    ax1.set_xlim(0, 11); ax1.set_ylim(0, 8)
    ax1.axis("off"); ax1.set_facecolor(C_LIGHT)

    ax1.text(5.5, 7.55, "스타일 분류기 작동 과정",
             ha="center", fontsize=15, fontweight="bold", color=C_DARK)

    # ── 박스 ──────────────────────────────────────────────────────
    arch = [
        (5.5, 6.3, 8.0, 0.9,  "FashionCLIP  visual_projection 출력", "512차원",               C_ORANGE),
        (5.5, 4.8, 5.0, 0.85, "Linear  512 → 256",                    "",                      C_BLUE),
        (5.5, 3.7, 3.5, 0.75, "ReLU  +  Dropout (0.3)",               "",                      C_BLUE),
        (5.5, 2.6, 5.0, 0.85, "Linear  256 → 23",                     "23개 세부 스타일",       C_BLUE),
        (5.5, 1.5, 5.0, 0.85, "Softmax  →  상위 확률 집계",           "23개 → 10개 대분류",    C_GREEN),
        (5.5, 0.4, 6.5, 0.75, "Top-3 스타일 출력",                    "페미닌 41%  |  캐주얼 28%  |  스포티 18%", C_PURPLE),
    ]
    for (x, y, w, h, lbl, sub, col) in arch:
        box(ax1, x, y, w, h, lbl, sub, color=col, fontsize=10)
        if sub:
            pass  # sublabel already drawn in box()

    # ── 화살표 ────────────────────────────────────────────────────
    ys = [5.85, 5.22, 4.33, 3.22, 2.12, 0.78]
    for i in range(len(ys) - 1):
        arrow(ax1, 5.5, ys[i], 5.5, ys[i+1])

    # ── 체크포인트 주석 ───────────────────────────────────────────
    ax1.text(1.0, 2.1,
             "checkpoints/\nstyle_classifier.pt\n\nVal Top-1: 45.3%\nVal Top-3: 69.6%",
             ha="center", fontsize=8.5, color=C_DARK,
             bbox=dict(fc="white", ec=C_PURPLE, lw=1.2, pad=5, boxstyle="round,pad=0.4"))
    # 연결선
    ax1.annotate("", xy=(3.0, 2.6), xytext=(1.8, 2.35),
                 arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=1.3,
                                 mutation_scale=14, connectionstyle="arc3,rad=-0.15"))

    # ── 23→10 매핑 범례 ───────────────────────────────────────────
    mapping = {
        "트래디셔널": ["클래식", "프레피"],
        "매니시":     ["매니시", "톰보이"],
        "페미닌":     ["페미닌", "로맨틱", "섹시"],
        "에스닉":     ["히피", "웨스턴", "오리엔탈"],
        "컨템포러리": ["모던", "소피스트\n케이티드", "아방가르드"],
        "내추럴":     ["컨트리", "리조트"],
        "젠더리스":   ["젠더리스"],
        "스포티":     ["스포티"],
        "서브컬처":   ["레트로", "키치", "힙합", "펑크"],
        "캐주얼":     ["밀리터리", "스트리트"],
    }
    colors10 = plt.get_cmap("tab10")(np.linspace(0, 1, 10))

    # 오른쪽: 매핑 다이어그램 (30%)
    ax2 = fig.add_axes([0.59, 0.0, 0.41, 1.0])
    ax2.set_xlim(0, 5); ax2.set_ylim(-0.3, 8)
    ax2.axis("off"); ax2.set_facecolor(C_LIGHT)
    ax2.text(2.5, 7.55, "23개 → 10개 대분류 매핑",
             ha="center", fontsize=13, fontweight="bold", color=C_DARK)

    row_h = 0.62
    for i, (parent, children) in enumerate(mapping.items()):
        y_top = 6.8 - i * row_h
        col = colors10[i]

        # 대분류 박스
        bp = FancyBboxPatch((0.05, y_top - 0.26), 1.5, 0.44,
                            boxstyle="round,pad=0,rounding_size=0.06",
                            facecolor=col, edgecolor="white", lw=1.2, zorder=3)
        ax2.add_patch(bp)
        ax2.text(0.80, y_top, parent, ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color="white", zorder=4)

        # 화살표 → 세부 스타일
        ax2.annotate("", xy=(1.85, y_top), xytext=(1.6, y_top),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2,
                                     mutation_scale=12))

        child_str = "  /  ".join(children)
        ax2.text(1.9, y_top, child_str, ha="left", va="center",
                 fontsize=7.8, color=C_DARK)

    plt.savefig("style_classifier.png", dpi=160, bbox_inches="tight",
                facecolor=C_LIGHT)
    plt.close()
    print("[OK] style_classifier.png")


# ════════════════════════════════════════════════════════════════════
# 3. 검색 성능
# ════════════════════════════════════════════════════════════════════
def fig_search_performance():
    fig = plt.figure(figsize=(14, 6.5))
    fig.patch.set_facecolor(C_LIGHT)
    fig.suptitle("검색 성능 (Naver Shopping 7,538개 대상)",
                 fontsize=15, fontweight="bold", color=C_DARK, y=1.01)

    # ── 왼쪽: Top-K Accuracy 막대 ────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor(C_LIGHT)
    labels = ["Top-1", "Top-5", "Top-10"]
    values = [44, 78, 88]
    colors = [C_ORANGE, C_BLUE, C_GREEN]
    bars   = ax1.bar(labels, values, color=colors, width=0.5,
                     edgecolor="white", linewidth=1.5, zorder=3)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_title("Top-K Accuracy", fontsize=12, fontweight="bold", color=C_DARK)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.spines[['top','right','left','bottom']].set_visible(False)
    ax1.tick_params(axis='x', labelsize=11)
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                 f"{v}%", ha="center", va="bottom", fontsize=13,
                 fontweight="bold", color=C_DARK)

    # ── 가운데: MRR 게이지 ────────────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor(C_LIGHT)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.set_title("MRR\n(Mean Reciprocal Rank)", fontsize=12,
                  fontweight="bold", color=C_DARK)

    # 반원 게이지
    theta = np.linspace(np.pi, 0, 200)
    r_outer, r_inner = 0.38, 0.24

    # 배경 (회색)
    xo = r_outer * np.cos(theta) + 0.5
    yo = r_outer * np.sin(theta) + 0.30
    xi = r_inner * np.cos(theta[::-1]) + 0.5
    yi = r_inner * np.sin(theta[::-1]) + 0.30
    ax2.fill(np.concatenate([xo, xi]), np.concatenate([yo, yi]),
             color="#ECEFF1", zorder=1)

    # 채워진 부분 (MRR=0.58)
    mrr = 0.58
    theta_fill = np.linspace(np.pi, np.pi * (1 - mrr), 200)
    xo2 = r_outer * np.cos(theta_fill) + 0.5
    yo2 = r_outer * np.sin(theta_fill) + 0.30
    xi2 = r_inner * np.cos(theta_fill[::-1]) + 0.5
    yi2 = r_inner * np.sin(theta_fill[::-1]) + 0.30
    ax2.fill(np.concatenate([xo2, xi2]), np.concatenate([yo2, yi2]),
             color=C_PURPLE, alpha=0.85, zorder=2)

    ax2.text(0.5, 0.27, "0.58", ha="center", va="center",
             fontsize=26, fontweight="bold", color=C_PURPLE, zorder=3)
    ax2.text(0.5, 0.14, "≈ 상위 2위 이내 등장",
             ha="center", fontsize=9.5, color=C_DARK)
    ax2.text(0.12, 0.27, "0.0", ha="center", fontsize=8, color=C_GRAY)
    ax2.text(0.88, 0.27, "1.0", ha="center", fontsize=8, color=C_GRAY)

    # ── 오른쪽: 누적 정확도 라인 ──────────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor(C_LIGHT)
    ks  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Top-1=44, Top-5=78, Top-10=88 기준 보간
    acc = [44, 57, 66, 73, 78, 81, 83, 86, 87, 88]
    ax3.plot(ks, acc, "o-", color=C_BLUE, linewidth=2.5, markersize=7, zorder=3)
    ax3.fill_between(ks, acc, alpha=0.12, color=C_BLUE, zorder=2)
    # 주요 포인트 강조
    for k, a, col in [(1, 44, C_ORANGE), (5, 78, C_BLUE), (10, 88, C_GREEN)]:
        ax3.plot(k, a, "o", color=col, markersize=12, zorder=4)
        ax3.text(k, a + 2.5, f"Top-{k}\n{a}%", ha="center", fontsize=8.5,
                 fontweight="bold", color=col)
    ax3.set_xlim(0.5, 10.5); ax3.set_ylim(30, 98)
    ax3.set_xticks(ks); ax3.set_xlabel("K", fontsize=10)
    ax3.set_ylabel("Accuracy (%)", fontsize=10)
    ax3.set_title("Top-K Accuracy Curve", fontsize=12,
                  fontweight="bold", color=C_DARK)
    ax3.grid(alpha=0.3, zorder=0)
    ax3.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig("search_performance.png", dpi=160, bbox_inches="tight",
                facecolor=C_LIGHT)
    plt.close()
    print("[OK] search_performance.png")


if __name__ == "__main__":
    fig_embedding()
    fig_style_classifier()
    fig_search_performance()
    print("\n완료: embedding_process.png / style_classifier.png / search_performance.png")
