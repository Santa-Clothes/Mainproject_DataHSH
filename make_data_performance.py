"""
슬라이드 5: Data → Performance Connection 이미지 생성
"""
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np

C_BLUE   = "#4A90D9"
C_RED    = "#E63946"
C_GREEN  = "#4CAF50"
C_ORANGE = "#F5A623"
C_PURPLE = "#7B5EA7"
C_TEAL   = "#2A9D8F"
C_GRAY   = "#78909C"
C_LIGHT  = "#F5F7FF"
C_DARK   = "#263238"

def rounded_box(ax, x, y, w, h, text, sub="", fc="#4A90D9", tc="white",
                fontsize=10, radius=0.015, bold=True):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=fc, edgecolor="white",
                           linewidth=1.5, zorder=3,
                           transform=ax.transData, clip_on=False)
    ax.add_patch(patch)
    dy = 0.018 if sub else 0
    ax.text(x + w/2, y + h/2 + dy, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=tc, zorder=4)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.025, sub,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color=tc, alpha=0.88, zorder=4)

def arrow_h(ax, x1, x2, y, color=C_GRAY, label=""):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.2, mutation_scale=18), zorder=2)
    if label:
        ax.text((x1+x2)/2, y + 0.012, label,
                ha="center", va="bottom", fontsize=8, color=color,
                style="italic", zorder=5)

# ── 메인 ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor(C_LIGHT)

# 제목
fig.text(0.5, 0.96, "Data  →  Performance Connection",
         ha="center", va="top", fontsize=18, fontweight="bold", color=C_DARK)
fig.text(0.5, 0.925, "데이터 특성이 모델 성능의 상한선을 결정한다",
         ha="center", va="top", fontsize=11, color=C_GRAY, style="italic")

# ── GridSpec: 상단(연결 다이어그램) / 하단(개선 방향) ─────────────
gs = gridspec.GridSpec(2, 1, figure=fig,
                       left=0.03, right=0.97,
                       top=0.88, bottom=0.04,
                       hspace=0.10,
                       height_ratios=[2.8, 1])

# ════════════════════════════════════════════════════
# 상단: 데이터 특성 → 성능 영향 4행 연결
# ════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0])
ax.set_facecolor(C_LIGHT)
ax.set_xlim(0, 10); ax.set_ylim(0, 1)
ax.axis("off")

rows = [
    # (데이터 특성 텍스트, 서브, 색, 성능영향 텍스트, 서브, 색, 수치뱃지, 뱃지색)
    ("카테고리 불균형",     "TS 851개  vs  KN 1개",
     C_RED,
     "소수 카테고리 Top-1 정확도 하락",  "FAISS 후보 부족 → 검색 풀 제한",
     "#FFCDD2", C_RED,
     "851배 차이", C_RED),

    ("모델 착용 이미지 多",  "전체 이미지의 약 52%",
     C_ORANGE,
     "동일 쿼리 유사도 변동",           "포즈 분산 → 코사인 거리 ±0.08",
     "#FFF3E0", C_ORANGE,
     "±0.08 변동", C_ORANGE),

    ("스타일 레이블 23개",   "시각 경계가 모호한 클래스 다수",
     C_PURPLE,
     "스타일 분류 Top-1  45.3%",       "모던·페미닌·스트리트 혼동",
     "#EDE7F6", C_PURPLE,
     "Top-3  69.6%", C_PURPLE),

    ("학습 샘플 부족",       "클래스당 평균 22장 (총 500장)",
     C_BLUE,
     "분류 난이도 상승",                "과적합 위험 / 소수 클래스 학습 취약",
     "#E3F2FD", C_BLUE,
     "22장/클래스", C_BLUE),
]

row_h   = 0.175
row_gap = 0.035
start_y = 0.92

for i, (lt, ls, lc, rt, rs, rfc, rc, badge, bc) in enumerate(rows):
    y = start_y - i * (row_h + row_gap)

    # 왼쪽 박스 (데이터 특성)
    rounded_box(ax, 0.02, y - row_h, 3.0, row_h,
                lt, ls, fc=lc, fontsize=10)

    # 화살표
    arrow_h(ax, 3.02, 3.55, y - row_h/2, color=lc)

    # 가운데 뱃지 (수치)
    rounded_box(ax, 3.55, y - row_h + 0.04, 1.30, row_h - 0.08,
                badge, fc=bc, tc="white", fontsize=9, radius=0.025)

    # 화살표
    arrow_h(ax, 4.85, 5.38, y - row_h/2, color=rc)

    # 오른쪽 박스 (성능 영향)
    rounded_box(ax, 5.38, y - row_h, 4.58, row_h,
                rt, rs, fc=rfc, tc=C_DARK, fontsize=10)

    # 구분선 (마지막 제외)
    if i < len(rows) - 1:
        sep_y = y - row_h - row_gap/2
        ax.plot([0.01, 9.97], [sep_y, sep_y],
                color="#DDE", lw=0.8, zorder=1)

# 컬럼 헤더
ax.text(1.52,  start_y + 0.04, "데이터 특성",
        ha="center", fontsize=11, fontweight="bold", color=C_DARK)
ax.text(4.20,  start_y + 0.04, "수치",
        ha="center", fontsize=11, fontweight="bold", color=C_DARK)
ax.text(7.67,  start_y + 0.04, "성능 영향",
        ha="center", fontsize=11, fontweight="bold", color=C_DARK)

# 헤더 밑줄
for x1, x2 in [(0.02, 3.02), (3.55, 4.85), (5.38, 9.96)]:
    ax.plot([x1, x2], [start_y, start_y], color=C_DARK, lw=1.5)

# ════════════════════════════════════════════════════
# 하단: 개선 방향 3열
# ════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#EEF2FF")
ax2.set_xlim(0, 10); ax2.set_ylim(0, 1)
ax2.axis("off")

ax2.text(5.0, 0.88, "개선 방향",
         ha="center", va="top", fontsize=12,
         fontweight="bold", color=C_DARK)

improvements = [
    (C_TEAL,   "데이터 증강\n(Augmentation)",
               "회전·플립·색상변환으로\n소수 카테고리 샘플 확대"),
    (C_BLUE,   "배경 제거 전처리",
               "모델 착용 이미지의\n포즈 분산 영향 감소"),
    (C_ORANGE, "오버샘플링",
               "SMOTE 등으로 소수 클래스\n임베딩 보간 생성"),
    (C_GREEN,  "추가 데이터 수집",
               "Crawling 확대로\n검색 풀 다양성 확보"),
]

box_w = 2.10
gap   = 0.20
total_w = len(improvements) * box_w + (len(improvements)-1) * gap
start_x = (10 - total_w) / 2

for j, (col, title, body) in enumerate(improvements):
    bx = start_x + j * (box_w + gap)
    # 배경 박스
    bg = FancyBboxPatch((bx, 0.04), box_w, 0.72,
                        boxstyle="round,pad=0,rounding_size=0.04",
                        facecolor="white", edgecolor=col,
                        linewidth=2, zorder=3)
    ax2.add_patch(bg)
    # 컬러 탑바
    top = FancyBboxPatch((bx, 0.68), box_w, 0.12,
                         boxstyle="round,pad=0,rounding_size=0.03",
                         facecolor=col, edgecolor="none", zorder=4)
    ax2.add_patch(top)
    ax2.text(bx + box_w/2, 0.74, title,
             ha="center", va="center", fontsize=9.5,
             fontweight="bold", color="white", zorder=5)
    ax2.text(bx + box_w/2, 0.37, body,
             ha="center", va="center", fontsize=9,
             color=C_DARK, linespacing=1.6, zorder=5)

plt.savefig("data_performance.png", dpi=160, bbox_inches="tight",
            facecolor=C_LIGHT)
plt.close()
print("[OK] data_performance.png")
