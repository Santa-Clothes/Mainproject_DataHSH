"""
검색 & 시각화 통합 스크립트
===========================

로컬 이미지를 업로드하여 검색하고 결과를 시각화
"""

import sys
from pathlib import Path
import argparse

import requests
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.visualize_search_results import visualize_from_api_response


def search_and_visualize(
    image_path: str,
    api_url: str = "http://localhost:8001/search/upload",
    category_filter: str = None,
    top_k: int = 10,
    save_path: str = None,
):
    """
    이미지로 검색하고 결과 시각화

    Args:
        image_path: 검색할 이미지 경로
        api_url: API 엔드포인트 URL
        category_filter: 카테고리 필터 (optional)
        top_k: 반환할 결과 수
        save_path: 시각화 저장 경로 (optional)
    """
    print(f"\n{'='*80}")
    print(f"Fashion Image Search & Visualization")
    print(f"{'='*80}")

    # 1. 이미지 파일 확인
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\n[1] Query Image: {image_path.name}")

    # 2. API 호출
    print(f"[2] Calling API: {api_url}")

    # 파일 준비
    with open(image_path, 'rb') as f:
        files = {'file': (image_path.name, f, 'image/png')}

        # 쿼리 파라미터
        params = {'top_k': top_k}
        if category_filter:
            params['category_filter'] = category_filter

        # API 요청
        response = requests.post(api_url, files=files, params=params)

    # 3. 응답 확인
    if response.status_code != 200:
        print(f"[ERROR] API request failed: {response.status_code}")
        print(f"Response: {response.text}")
        return

    api_response = response.json()
    result_count = api_response.get('result_count', 0)
    print(f"[3] Search Results: {result_count} products found")

    # 4. 시각화
    print(f"[4] Visualizing results...")
    visualize_from_api_response(
        api_response=api_response,
        query_image_path=str(image_path),
        save_path=save_path,
    )

    print(f"\n{'='*80}")
    print(f"[OK] Visualization complete!")
    if save_path:
        print(f"[OK] Saved to: {save_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Search by image and visualize results")
    parser.add_argument("image", type=str, help="Query image path")
    parser.add_argument("--api_url", type=str, default="http://localhost:8001/search/upload",
                       help="API endpoint URL")
    parser.add_argument("--category", type=str, default=None,
                       help="Category filter (e.g., 'BL', 'OP', 'SK')")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of results to return")
    parser.add_argument("--save", type=str, default=None,
                       help="Save visualization to file")

    args = parser.parse_args()

    search_and_visualize(
        image_path=args.image,
        api_url=args.api_url,
        category_filter=args.category,
        top_k=args.top_k,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
