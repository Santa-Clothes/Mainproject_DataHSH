"""
검색 결과 시각화
=================

API 검색 결과를 이미지 그리드로 시각화
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import platform

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from PIL import Image
import requests
from io import BytesIO

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_korean_font():
    """한글 폰트 설정"""
    system = platform.system()

    # Windows
    if system == 'Windows':
        font_candidates = ['Malgun Gothic', 'Gulim', 'Dotum', 'Batang']
    # macOS
    elif system == 'Darwin':
        font_candidates = ['AppleGothic', 'Apple SD Gothic Neo']
    # Linux
    else:
        font_candidates = ['NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']

    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"[Font] Using: {font}")
            break
    else:
        print("[Font] Warning: No Korean font found, text may not display correctly")

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False


def download_image(url: str, timeout: int = 5) -> Optional[Image.Image]:
    """URL에서 이미지 다운로드"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
        return None


def visualize_search_results(
    query_image: Image.Image,
    results: List[Dict],
    query_title: str = "Query Image",
    max_results: int = 10,
    save_path: Optional[str] = None,
):
    """
    검색 결과 시각화

    Args:
        query_image: 쿼리 이미지
        results: 검색 결과 리스트
        query_title: 쿼리 이미지 제목
        max_results: 표시할 최대 결과 수
        save_path: 저장 경로 (None이면 화면에만 표시)
    """
    # 한글 폰트 설정
    setup_korean_font()

    n_results = min(len(results), max_results)

    # 그리드 레이아웃 계산 (3열)
    n_cols = 3
    n_rows = (n_results + n_cols - 1) // n_cols + 1  # +1 for query image row

    # Figure 생성
    fig = plt.figure(figsize=(15, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.3)

    # 쿼리 이미지 표시 (첫 행 중앙)
    ax_query = fig.add_subplot(gs[0, 1])
    ax_query.imshow(query_image)
    ax_query.set_title(f"[Query] {query_title}", fontsize=14, fontweight='bold', pad=10)
    ax_query.axis('off')

    # 결과 이미지 표시
    for idx, result in enumerate(results[:n_results]):
        row = (idx // n_cols) + 1
        col = idx % n_cols

        ax = fig.add_subplot(gs[row, col])

        # 이미지 다운로드
        img = download_image(result['image_url'])

        if img is not None:
            ax.imshow(img)

            # 제목 (길이 제한)
            title = result['title']
            if len(title) > 40:
                title = title[:37] + "..."

            # 점수와 가격
            score = result['score']
            price = result['price']

            label = f"#{idx+1} (Score: {score:.3f})\n{title}\n₩{price:,}"
            ax.set_title(label, fontsize=9, pad=5)
        else:
            # 이미지 로드 실패
            ax.text(0.5, 0.5, 'Image Load Failed',
                   ha='center', va='center', fontsize=10)
            ax.set_title(f"#{idx+1} (Score: {result['score']:.3f})", fontsize=9)

        ax.axis('off')

    plt.suptitle('Fashion Search Results', fontsize=18, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    plt.show()


def visualize_from_api_response(
    api_response: Dict,
    query_image_path: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    API 응답으로부터 시각화

    Args:
        api_response: API 응답 JSON
        query_image_path: 쿼리 이미지 경로 (None이면 첫 번째 결과의 이미지 사용)
        save_path: 저장 경로
    """
    results = api_response.get('results', [])

    if not results:
        print("No results to visualize!")
        return

    # 쿼리 이미지 로드
    if query_image_path:
        query_image = Image.open(query_image_path)
        if query_image.mode != 'RGB':
            query_image = query_image.convert('RGB')
        query_title = Path(query_image_path).name
    else:
        # 쿼리 이미지가 없으면 첫 번째 결과 이미지를 placeholder로 사용
        print("Warning: No query image provided, using placeholder")
        query_image = Image.new('RGB', (224, 224), color='lightgray')
        query_title = "Query Image (not available)"

    # 시각화
    visualize_search_results(
        query_image=query_image,
        results=results,
        query_title=query_title,
        save_path=save_path,
    )


def main():
    """테스트용 메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize search results")
    parser.add_argument("--response", type=str, required=True,
                       help="API response JSON file path")
    parser.add_argument("--query_image", type=str, default=None,
                       help="Query image path")
    parser.add_argument("--save", type=str, default=None,
                       help="Save path for visualization")

    args = parser.parse_args()

    # Load API response
    with open(args.response, 'r', encoding='utf-8') as f:
        api_response = json.load(f)

    # Visualize
    visualize_from_api_response(
        api_response=api_response,
        query_image_path=args.query_image,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
