"""
K-Fashion 유사 상품 검색 - Gradio 데모
=======================================

FastAPI 서버(포트 8001)를 먼저 실행한 후 이 스크립트를 실행하세요.

실행:
    uvicorn api.search_api:app --host 0.0.0.0 --port 8001 --reload  # API 서버
    python gradio_demo.py                                              # Gradio 데모
"""

import io
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import gradio as gr
from PIL import Image

matplotlib.use("Agg")

# ── 설정 ──────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8001"
CATEGORIES = ["전체", "BL", "OP", "SK", "PT", "JK", "JP", "CT", "OT", "SE"]

# 한글 폰트 설정 (Windows)
def _set_korean_font():
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    for name in candidates:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()


# ── 이미지 다운로드 헬퍼 ─────────────────────────────────────────────────────
def _fetch_image(url: str, timeout: int = 5) -> Image.Image | None:
    """URL에서 이미지를 가져옵니다. 실패하면 None 반환."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _pil_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


# ── 스타일 차트 생성 ──────────────────────────────────────────────────────────
def _make_style_chart(styles: list[dict]) -> plt.Figure:
    names = [s["style"] for s in styles]
    scores = [s["score"] for s in styles]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"][: len(styles)]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("확률")
    ax.set_title("K-Fashion 스타일 분류 (Top-3)")
    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=10,
        )
    plt.tight_layout()
    return fig


# ── 메인 검색 함수 ────────────────────────────────────────────────────────────
def search_fashion(image: Image.Image, top_k: int, category: str):
    """이미지를 검색하고 결과를 반환합니다."""
    if image is None:
        return [], None, "이미지를 업로드해주세요."

    img_bytes = _pil_to_bytes(image)
    files = {"file": ("query.jpg", img_bytes, "image/jpeg")}
    params = {"top_k": int(top_k)}
    if category and category != "전체":
        params["category_filter"] = category

    # ── 검색 API 호출 ─────────────────────────────────────────────────────
    try:
        resp = requests.post(
            f"{API_BASE}/search/upload", files=files, params=params, timeout=30
        )
        resp.raise_for_status()
        search_data = resp.json()
    except requests.exceptions.ConnectionError:
        return [], None, "❌ API 서버에 연결할 수 없습니다. 포트 8001을 확인하세요."
    except Exception as e:
        return [], None, f"❌ 검색 오류: {e}"

    results = search_data.get("results", [])
    if not results:
        return [], None, "검색 결과가 없습니다."

    # ── 갤러리 이미지 준비 ────────────────────────────────────────────────
    gallery_items = []
    for r in results:
        rank = r.get("rank", "?")
        title = r.get("title", "")[:30]
        price = int(r.get("price", 0))
        score = r.get("score", 0.0)
        img = _fetch_image(r.get("image_url", ""))
        if img is not None:
            caption = f"#{rank} {title}\n₩{price:,}  |  유사도 {score:.3f}"
            gallery_items.append((img, caption))

    # ── 상품 정보 마크다운 ────────────────────────────────────────────────
    meta = search_data.get("metrics", {})
    info_lines = [
        f"**검색 결과 {len(results)}개** | 소요시간 {meta.get('search_time_ms', '?')}ms\n",
        "| # | 상품명 | 가격 | 유사도 | 카테고리 |",
        "|---|--------|------|--------|----------|",
    ]
    for r in results:
        title = r.get("title", "")[:30]
        price = f"₩{int(r.get('price', 0)):,}"
        score = f"{r.get('score', 0):.3f}"
        cat = r.get("kfashion_category", "")
        rank = r.get("rank", "")
        info_lines.append(f"| {rank} | {title} | {price} | {score} | {cat} |")

    # ── 스타일 분류 API 호출 ──────────────────────────────────────────────
    style_chart = None
    try:
        files2 = {"file": ("query.jpg", img_bytes, "image/jpeg")}
        resp2 = requests.post(f"{API_BASE}/analyze", files=files2, timeout=30)
        if resp2.status_code == 200:
            styles = resp2.json().get("styles", [])
            if styles:
                style_chart = _make_style_chart(styles)
    except Exception:
        pass  # 스타일 분류 실패해도 검색 결과는 표시

    return gallery_items, style_chart, "\n".join(info_lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="K-Fashion 유사 상품 검색",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1200px; margin: auto; }",
    ) as demo:
        gr.Markdown(
            """
            # 👗 K-Fashion 유사 상품 검색
            패션 이미지를 업로드하면 유사한 네이버 쇼핑 상품을 찾아드립니다.
            > **먼저 FastAPI 서버를 실행하세요**: `uvicorn api.search_api:app --port 8001`
            """
        )

        with gr.Row():
            # ── 입력 패널 ───────────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                input_image = gr.Image(
                    type="pil",
                    label="검색할 패션 이미지",
                    height=300,
                )
                top_k = gr.Slider(
                    minimum=1, maximum=20, value=8, step=1, label="결과 수 (top-k)"
                )
                category = gr.Dropdown(
                    choices=CATEGORIES, value="전체", label="카테고리 필터"
                )
                search_btn = gr.Button("🔍 검색", variant="primary", size="lg")
                gr.Examples(
                    examples=[["test_img.jpg"]],
                    inputs=input_image,
                    label="예시 이미지",
                )

            # ── 출력 패널 ───────────────────────────────────────────────
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🖼️ 유사 상품"):
                        result_gallery = gr.Gallery(
                            label="검색 결과",
                            columns=4,
                            height=400,
                            object_fit="contain",
                        )
                        result_info = gr.Markdown()

                    with gr.TabItem("🎨 스타일 분류"):
                        style_chart = gr.Plot(label="K-Fashion 스타일 (Top-3)")
                        gr.Markdown(
                            "_FashionCLIP + MLP 분류기 기반 10개 대분류 스타일 분류_"
                        )

        # 검색 버튼 클릭
        search_btn.click(
            fn=search_fashion,
            inputs=[input_image, top_k, category],
            outputs=[result_gallery, style_chart, result_info],
        )
        # 이미지 업로드 시 자동 검색
        input_image.change(
            fn=search_fashion,
            inputs=[input_image, top_k, category],
            outputs=[result_gallery, style_chart, result_info],
        )

    return demo


# ── 엔트리포인트 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("K-Fashion Gradio 데모")
    print("=" * 60)
    print("전제 조건: FastAPI 서버가 포트 8001에서 실행 중이어야 합니다.")
    print("  uvicorn api.search_api:app --host 0.0.0.0 --port 8001")
    print("=" * 60)

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
