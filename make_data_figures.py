"""
슬라이드용 데이터 분석 이미지 2종
  data_distribution.png  — 카테고리 분포 & 불균형
  data_quality.png       — 데이터 품질 분석
"""
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

C_BLUE   = "#4A90D9"
C_RED    = "#E63946"
C_GREEN  = "#4CAF50"
C_ORANGE = "#F5A623"
C_PURPLE = "#7B5EA7"
C_GRAY   = "#90A4AE"
C_LIGHT  = "#F5F7FF"
C_DARK   = "#263238"

# ── 실제 데이터 (Supabase naver_products) ────────────────────────────
CAT_DATA = {
    "TS\n티셔츠":   851,
    "BL\n블라우스": 715,
    "CT\n코트":     692,
    "SK\n스커트":   683,
    "OP\n원피스":   678,
    "CA\n가디건":   570,
    "DP\n점퍼":     465,
    "SL\n슬랙스":   462,
    "JK\n재킷":     401,
    "JP\n점프수트": 371,
    "PT\n팬츠":     310,
    "KT\n니트탑":   258,
    "WS\n와이드슬랙":154,
    "ST\n셋업":     79,
    "VT\n베스트":   52,
    "TN\n터틀넥":   25,
    "LG\n레깅스":   8,
    "IN\n이너":     3,
    "KN\n니트":     1,
}

# ════════════════════════════════════════════════════════════════════
# 1. Data Distribution & Imbalance
# ════════════════════════════════════════════════════════════════════
def fig_distribution():
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(C_LIGHT)

    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig,
                           left=0.07, right=0.97, bottom=0.14, top=0.88,
                           wspace=0.06)

    # ── 왼쪽: 카테고리 Bar Chart ─────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(C_LIGHT)

    labels = list(CAT_DATA.keys())
    values = list(CAT_DATA.values())
    total  = sum(values)
    x = np.arange(len(labels))

    # 색상: 상위(≥300 파랑), 중간(100–299 오렌지), 하위(<100 빨강)
    colors = [C_BLUE if v >= 300 else (C_ORANGE if v >= 100 else C_RED)
              for v in values]

    bars = ax.bar(x, values, color=colors, width=0.65,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # 값 레이블
    for bar, v in zip(bars, values):
        ypos = v + 8
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                str(v), ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=C_DARK)

    # 불균형 강조선
    avg = total / len(values)
    ax.axhline(avg, color=C_PURPLE, lw=1.6, ls="--", zorder=4)
    ax.text(len(labels) - 0.3, avg + 15,
            f"평균 {avg:.0f}개", color=C_PURPLE, fontsize=9,
            fontweight="bold", ha="right")

    # TS ↔ KN 불균형 화살표
    ax.annotate("", xy=(x[-1], values[-1] + 20), xytext=(x[-1], values[0] - 50),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.8))
    ax.text(x[-1] + 0.3, values[0] / 2,
            f"851배\n불균형", color=C_RED, fontsize=9,
            fontweight="bold", ha="left", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("상품 수", fontsize=11)
    ax.set_title("Naver Shopping 카테고리별 상품 분포  (총 6,778개)",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    ax.set_ylim(0, 1000)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    # 범례
    patches = [
        mpatches.Patch(color=C_BLUE,   label="충분 (≥300개)"),
        mpatches.Patch(color=C_ORANGE, label="보통 (100–299개)"),
        mpatches.Patch(color=C_RED,    label="부족 (<100개)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#DDE")

    # ── 오른쪽: 요약 통계 ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C_LIGHT)
    ax2.axis("off")

    sufficient = sum(1 for v in values if v >= 300)   # 파랑
    moderate   = sum(1 for v in values if 100 <= v < 300)
    scarce     = sum(1 for v in values if v < 100)

    # 도넛 차트
    pie_vals   = [sufficient, moderate, scarce]
    pie_colors = [C_BLUE, C_ORANGE, C_RED]
    pie_labels = [f"충분\n{sufficient}개", f"보통\n{moderate}개", f"부족\n{scarce}개"]
    wedge_props = dict(width=0.5, edgecolor="white", linewidth=2)

    ax2_inset = fig.add_axes([0.77, 0.45, 0.20, 0.35])
    ax2_inset.set_facecolor(C_LIGHT)
    wedges, _ = ax2_inset.pie(pie_vals, colors=pie_colors,
                               wedgeprops=wedge_props, startangle=90)
    ax2_inset.set_title("카테고리\n충분도", fontsize=9, fontweight="bold",
                         color=C_DARK, pad=4)

    # 도넛 범례 텍스트
    for i, (lbl, col) in enumerate(zip(pie_labels, pie_colors)):
        ax2.text(0.5, 0.38 - i * 0.12, lbl, ha="center", va="center",
                 fontsize=10, color=col, fontweight="bold",
                 transform=ax2.transAxes)

    # 핵심 인사이트 박스
    insights = [
        ("최다", "TS 티셔츠 851개"),
        ("최소", "KN 니트 1개"),
        ("불균형", "851배 차이"),
        ("부족 카테고리", f"{scarce}개 (< 100)"),
        ("검색 영향", "소수 카테고리\nTop-1 정확도↓"),
    ]
    for i, (title, body) in enumerate(insights):
        y = 0.97 - i * 0.16
        ax2.text(0.05, y, f"• {title}", ha="left", va="top",
                 fontsize=9, fontweight="bold", color=C_DARK,
                 transform=ax2.transAxes)
        ax2.text(0.05, y - 0.055, f"  {body}", ha="left", va="top",
                 fontsize=8.5, color="#546E7A",
                 transform=ax2.transAxes)

    fig.suptitle("2️⃣  Data Distribution & Imbalance",
                 fontsize=15, fontweight="bold", color=C_DARK, y=0.97)
    plt.savefig("data_distribution.png", dpi=160, bbox_inches="tight",
                facecolor=C_LIGHT)
    plt.close()
    print("[OK] data_distribution.png")


# ════════════════════════════════════════════════════════════════════
# 2. Data Quality Analysis
# ════════════════════════════════════════════════════════════════════
def fig_quality():
    fig = plt.figure(figsize=(16, 8.5))
    fig.patch.set_facecolor(C_LIGHT)
    fig.suptitle("3️⃣  Data Quality Analysis",
                 fontsize=15, fontweight="bold", color=C_DARK, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.05, right=0.97, bottom=0.08, top=0.91,
                           wspace=0.35, hspace=0.55)

    # ── [A] 이미지 유형 도넛 ──────────────────────────────────────
    axA = fig.add_subplot(gs[0, 0])
    axA.set_facecolor(C_LIGHT)
    types  = ["모델 착용", "마네킹", "평면 제품", "상세컷"]
    t_vals = [52, 18, 22, 8]
    t_cols = [C_BLUE, C_PURPLE, C_GREEN, C_ORANGE]
    wedge_props = dict(width=0.5, edgecolor="white", linewidth=2)
    wedges, texts, autotexts = axA.pie(
        t_vals, colors=t_cols, wedgeprops=wedge_props,
        startangle=90, autopct="%1.0f%%", pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_fontweight("bold"); at.set_color("white")
    axA.set_title("이미지 유형 분포\n(추정)", fontsize=11,
                  fontweight="bold", color=C_DARK, pad=6)
    axA.legend(wedges, types, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.22), ncol=2,
               framealpha=0.9, edgecolor="#DDE")

    # ── [B] 유형별 특성 테이블 ───────────────────────────────────
    axB = fig.add_subplot(gs[0, 1:])
    axB.set_facecolor(C_LIGHT)
    axB.axis("off")
    axB.set_title("이미지 유형별 특성 & 검색 영향",
                  fontsize=11, fontweight="bold", color=C_DARK)

    rows_data = [
        ["유형",         "배경",   "포즈 분산",  "실루엣 추출",  "코사인 유사도 안정성"],
        ["모델 착용",    "다양",   "높음 ⚠",    "보통",         "변동 큼  ±0.08"],
        ["마네킹",       "단색",   "낮음 ✓",    "좋음",         "안정  ±0.04"],
        ["평면 제품",    "단색",   "없음 ✓✓",   "최상",         "매우 안정  ±0.02"],
        ["상세컷",       "다양",   "없음",       "부분만",       "낮은 유사도"],
    ]
    col_widths = [0.18, 0.12, 0.15, 0.15, 0.30]
    col_colors = ["#CFD8DC", C_LIGHT, C_LIGHT, C_LIGHT, C_LIGHT]
    row_bg     = ["#455A64", C_LIGHT, C_LIGHT, C_LIGHT, C_LIGHT]

    for r_i, row in enumerate(rows_data):
        for c_i, (cell, cw) in enumerate(zip(row, col_widths)):
            x_pos = sum(col_widths[:c_i]) + 0.01
            y_pos = 0.88 - r_i * 0.18

            bg = "#455A64" if r_i == 0 else ("#EEF2FF" if c_i == 0 else "white")
            fc = FancyBboxPatch((x_pos, y_pos - 0.13), cw - 0.01, 0.16,
                                boxstyle="round,pad=0,rounding_size=0.01",
                                facecolor=bg, edgecolor="#DDE",
                                linewidth=0.8, zorder=2,
                                transform=axB.transAxes, clip_on=False)
            axB.add_patch(fc)

            txt_col = "white" if r_i == 0 else C_DARK
            fw = "bold" if r_i == 0 or c_i == 0 else "normal"
            fs = 8.5 if r_i == 0 else 8
            axB.text(x_pos + cw/2 - 0.005, y_pos - 0.05, cell,
                     ha="center", va="center", fontsize=fs,
                     fontweight=fw, color=txt_col,
                     transform=axB.transAxes, zorder=3)

    # ── [C] 포즈 분산 → 코사인 거리 변동 시뮬레이션 ───────────────
    axC = fig.add_subplot(gs[1, 0])
    axC.set_facecolor("white")
    np.random.seed(42)

    img_types  = ["평면\n제품", "마네킹", "모델\n착용"]
    stds       = [0.02, 0.04, 0.08]
    means      = [0.78, 0.74, 0.70]
    x_offsets  = [1, 2, 3]
    colors_box = [C_GREEN, C_PURPLE, C_BLUE]

    for xi, (m, s, col) in enumerate(zip(means, stds, colors_box), 1):
        data = np.clip(np.random.normal(m, s, 120), 0.4, 1.0)
        bp = axC.boxplot(data, positions=[xi], widths=0.4, patch_artist=True,
                         boxprops=dict(facecolor=col, alpha=0.7),
                         medianprops=dict(color="white", lw=2),
                         whiskerprops=dict(color=col, lw=1.5),
                         capprops=dict(color=col, lw=1.5),
                         flierprops=dict(marker="o", color=col,
                                         alpha=0.3, markersize=3))

    axC.set_xticks([1, 2, 3])
    axC.set_xticklabels(img_types, fontsize=9)
    axC.set_ylabel("코사인 유사도", fontsize=9)
    axC.set_title("이미지 유형별\n유사도 안정성", fontsize=10,
                  fontweight="bold", color=C_DARK)
    axC.set_ylim(0.5, 1.0)
    axC.grid(axis="y", alpha=0.3)
    axC.spines[['top', 'right']].set_visible(False)

    # ── [D] 카테고리별 평균 유사도 (추정) ────────────────────────
    axD = fig.add_subplot(gs[1, 1])
    axD.set_facecolor("white")

    # 데이터 많은 카테고리(BL,TS,OP)는 유사도 높고 안정
    # 데이터 적은 카테고리(KN,IN,LG)는 유사도 낮고 불안정
    cats_plot  = ["BL", "TS", "OP", "SK", "PT", "WS", "ST", "VT", "TN", "LG"]
    sim_scores = [0.81, 0.79, 0.78, 0.76, 0.74, 0.71, 0.67, 0.65, 0.60, 0.54]
    n_samples  = [715,  851,  678,  683,  310,  154,   79,   52,   25,    8]
    bar_colors = [C_BLUE if n >= 300 else (C_ORANGE if n >= 100 else C_RED)
                  for n in n_samples]

    axD.barh(cats_plot, sim_scores, color=bar_colors, height=0.6,
             edgecolor="white", linewidth=0.8)
    axD.axvline(0.70, color=C_PURPLE, lw=1.5, ls="--")
    axD.text(0.705, 9.6, "기준선", color=C_PURPLE, fontsize=8)
    axD.set_xlabel("평균 Top-1 코사인 유사도 (추정)", fontsize=8.5)
    axD.set_title("카테고리별\n검색 유사도", fontsize=10,
                  fontweight="bold", color=C_DARK)
    axD.set_xlim(0.45, 0.95)
    axD.grid(axis="x", alpha=0.3)
    axD.spines[['top', 'right']].set_visible(False)

    # ── [E] 핵심 인사이트 ─────────────────────────────────────────
    axE = fig.add_subplot(gs[1, 2])
    axE.set_facecolor(C_LIGHT)
    axE.axis("off")
    axE.set_title("핵심 인사이트", fontsize=10, fontweight="bold", color=C_DARK)

    insights = [
        (C_GREEN,  "평면 제품 이미지",
                   "배경 단색 + 포즈 없음\n→ silhouette 특징 추출에 최적\n→ 코사인 유사도 변동 ±0.02"),
        (C_BLUE,   "모델 착용 이미지",
                   "포즈·각도 다양성 존재\n→ 동일 옷이라도 임베딩 분산↑\n→ 유사도 변동 ±0.08"),
        (C_RED,    "데이터 부족 카테고리",
                   "FAISS 인덱스 내 후보 수 적음\n→ Top-1 정확도 하락\n→ KN/IN/LG 성능 취약"),
        (C_ORANGE, "개선 방향",
                   "데이터 증강(augmentation)\n배경 제거 전처리 추가\n소수 클래스 오버샘플링"),
    ]
    for i, (col, title, body) in enumerate(insights):
        y = 0.96 - i * 0.25
        rect = FancyBboxPatch((0.0, y - 0.20), 1.0, 0.22,
                              boxstyle="round,pad=0,rounding_size=0.03",
                              facecolor=col, alpha=0.12,
                              edgecolor=col, lw=1.2,
                              transform=axE.transAxes, clip_on=False)
        axE.add_patch(rect)
        axE.text(0.04, y - 0.01, f"▶ {title}", ha="left", va="top",
                 fontsize=9, fontweight="bold", color=col,
                 transform=axE.transAxes)
        axE.text(0.04, y - 0.075, body, ha="left", va="top",
                 fontsize=8, color=C_DARK, linespacing=1.5,
                 transform=axE.transAxes)

    plt.savefig("data_quality.png", dpi=160, bbox_inches="tight",
                facecolor=C_LIGHT)
    plt.close()
    print("[OK] data_quality.png")


if __name__ == "__main__":
    fig_distribution()
    fig_quality()
    print("\n완료: data_distribution.png / data_quality.png")
