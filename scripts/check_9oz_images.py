"""
나인오즈 CSV 파일 및 이미지 확인 도구
"""

import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

def check_csv_structure(csv_path: str):
    """CSV 파일 구조 확인"""
    print(f"\n{'='*60}")
    print(f"CSV 파일 분석: {csv_path}")
    print(f"{'='*60}\n")

    # CSV 로드
    df = pd.read_csv(csv_path)

    # 기본 정보
    print(f"총 제품 수: {len(df)}")
    print(f"\n컬럼 목록:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    # 첫 5개 샘플
    print(f"\n첫 5개 샘플:")
    print(df.head())

    # 이미지 관련 컬럼 찾기
    image_columns = [col for col in df.columns if any(
        keyword in col.lower()
        for keyword in ['image', 'img', '이미지', 'url', 'path', '경로']
    )]

    if image_columns:
        print(f"\n이미지 관련 컬럼: {image_columns}")
        for col in image_columns:
            print(f"\n{col} 샘플:")
            print(df[col].head(3))
    else:
        print("\n⚠️  이미지 관련 컬럼을 찾지 못했습니다.")
        print("   수동으로 확인이 필요합니다.")

    return df, image_columns


def display_images_from_csv(csv_path: str, n_samples: int = 5):
    """CSV에서 이미지 샘플 확인"""
    df = pd.read_csv(csv_path)

    # 이미지 컬럼 추정
    image_columns = [col for col in df.columns if any(
        keyword in col.lower()
        for keyword in ['image', 'img', '이미지', 'url']
    )]

    if not image_columns:
        print("❌ 이미지 컬럼을 찾을 수 없습니다.")
        return

    image_col = image_columns[0]
    print(f"\n이미지 컬럼 사용: {image_col}")

    for idx in range(min(n_samples, len(df))):
        row = df.iloc[idx]
        image_info = row.get(image_col, "")

        print(f"\n--- 제품 {idx + 1} ---")
        print(f"제품명: {row.get('제품명', 'N/A')}")
        print(f"카테고리: {row.get('kfashion_item_category', 'N/A')}")
        print(f"이미지: {image_info}")

        # 이미지가 URL인지 로컬 경로인지 확인
        if isinstance(image_info, str):
            if image_info.startswith('http'):
                print(f"  → URL 이미지 (웹 브라우저로 확인 가능)")
            elif Path(image_info).exists():
                print(f"  → 로컬 파일 존재 ✓")
            else:
                print(f"  → 로컬 파일 없음 ✗")


def open_sample_images(csv_path: str, n_samples: int = 3):
    """샘플 이미지를 실제로 열어서 확인 (URL 또는 로컬)"""
    df = pd.read_csv(csv_path)

    # 이미지 컬럼 찾기
    image_columns = [col for col in df.columns if any(
        keyword in col.lower()
        for keyword in ['image', 'img', '이미지', 'url']
    )]

    if not image_columns:
        print("❌ 이미지 컬럼을 찾을 수 없습니다.")
        return

    image_col = image_columns[0]

    for idx in range(min(n_samples, len(df))):
        row = df.iloc[idx]
        image_info = row.get(image_col, "")

        print(f"\n제품 {idx + 1}: {row.get('제품명', 'N/A')}")

        try:
            if isinstance(image_info, str):
                if image_info.startswith('http'):
                    # URL 이미지
                    response = requests.get(image_info, timeout=10)
                    img = Image.open(BytesIO(response.content))
                    print(f"  ✓ URL 이미지 로드 성공: {img.size}")
                    print(f"  이미지 타입 확인 필요: 평면 제품 vs 모델 착용")
                    # img.show()  # 이미지 뷰어로 열기 (선택)

                elif Path(image_info).exists():
                    # 로컬 파일
                    img = Image.open(image_info)
                    print(f"  ✓ 로컬 이미지 로드 성공: {img.size}")
                    print(f"  이미지 타입 확인 필요: 평면 제품 vs 모델 착용")
                    # img.show()  # 이미지 뷰어로 열기 (선택)
                else:
                    print(f"  ✗ 이미지를 찾을 수 없음: {image_info}")
        except Exception as e:
            print(f"  ❌ 이미지 로드 실패: {e}")


def classify_image_type_manual(csv_path: str, output_path: str = None):
    """이미지 타입을 수동으로 분류하는 도구 (interactive)"""
    df = pd.read_csv(csv_path)

    # 이미지 타입 컬럼이 없으면 추가
    if 'image_type' not in df.columns:
        df['image_type'] = ''

    print("\n이미지 타입 분류 도구")
    print("=" * 60)
    print("각 이미지를 보고 타입을 입력하세요:")
    print("  1 = 평면 제품 (flat_product)")
    print("  2 = 모델 착용 (model_wearing)")
    print("  s = 건너뛰기 (skip)")
    print("  q = 종료 (quit)")
    print("=" * 60)

    # TODO: 실제 구현 시 이미지를 보여주고 사용자 입력 받기
    # 지금은 샘플 코드만 제공

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="나인오즈 CSV 및 이미지 확인 도구")
    parser.add_argument(
        "csv_path",
        type=str,
        help="나인오즈 CSV 파일 경로"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["structure", "display", "open"],
        default="structure",
        help="실행 모드: structure(구조 확인), display(이미지 정보), open(이미지 열기)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="확인할 샘플 수"
    )

    args = parser.parse_args()

    # CSV 파일 존재 확인
    if not Path(args.csv_path).exists():
        print(f"❌ CSV 파일을 찾을 수 없습니다: {args.csv_path}")
        print("\n사용 가능한 경로 예시:")
        print("  - c:/Work/hwangseonghun/nineoz_with_kfashion_categories.csv")
        print("  - ./data/nineoz_with_kfashion_categories.csv")
        sys.exit(1)

    # 모드별 실행
    if args.mode == "structure":
        check_csv_structure(args.csv_path)
    elif args.mode == "display":
        display_images_from_csv(args.csv_path, args.samples)
    elif args.mode == "open":
        open_sample_images(args.csv_path, args.samples)
