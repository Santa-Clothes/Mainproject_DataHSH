"""
이미지 타입 분류 도구 (평면 제품 vs 모델 착용)
Domain Gap 문제 정량화를 위한 도구
"""

import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import sys

def download_and_show_images(csv_path: str, n_samples: int = 10, start_idx: int = 0):
    """
    CSV에서 이미지를 다운로드하고 표시

    Args:
        csv_path: CSV 파일 경로
        n_samples: 확인할 샘플 수
        start_idx: 시작 인덱스
    """
    print(f"\n{'='*80}")
    print(f"이미지 타입 분류: {csv_path}")
    print(f"{'='*80}\n")

    df = pd.read_csv(csv_path)

    # 이미지 타입 컬럼이 없으면 추가
    if 'image_type' not in df.columns:
        df['image_type'] = ''

    results = []

    for idx in range(start_idx, min(start_idx + n_samples, len(df))):
        row = df.iloc[idx]

        print(f"\n{'='*80}")
        print(f"제품 #{idx + 1}")
        print(f"{'='*80}")

        # 제품 정보 출력
        if 'product_id' in row:
            print(f"Product ID: {row['product_id']}")
        if 'product_name' in row:
            print(f"Product Name: {row['product_name']}")
        if 'title' in row:
            print(f"Title: {row['title']}")
        if 'category_id' in row:
            print(f"Category: {row['category_id']}")

        image_url = row.get('image_url', '')
        print(f"\nImage URL: {image_url}")

        # 이미지 다운로드 시도
        try:
            if image_url and isinstance(image_url, str) and image_url.startswith('http'):
                response = requests.get(image_url, timeout=10)
                img = Image.open(BytesIO(response.content))

                print(f"✓ 이미지 로드 성공: {img.size}")
                print(f"\n브라우저에서 확인: {image_url}")

                # 사용자 입력 받기
                print("\n이미지 타입을 선택하세요:")
                print("  1 = flat_product (평면 제품 - 옷만 단독 촬영)")
                print("  2 = model_wearing (모델 착용 - 사람이 입은 사진)")
                print("  3 = mannequin (마네킹 착용)")
                print("  s = skip (건너뛰기)")
                print("  q = quit (종료)")

                choice = input("\n선택 (1/2/3/s/q): ").strip().lower()

                if choice == 'q':
                    print("\n중단합니다.")
                    break
                elif choice == 's':
                    image_type = ''
                elif choice == '1':
                    image_type = 'flat_product'
                elif choice == '2':
                    image_type = 'model_wearing'
                elif choice == '3':
                    image_type = 'mannequin'
                else:
                    print("잘못된 입력입니다. 건너뜁니다.")
                    image_type = ''

                results.append({
                    'index': idx,
                    'product_id': row.get('product_id', ''),
                    'image_url': image_url,
                    'image_type': image_type
                })

            else:
                print("✗ 유효하지 않은 이미지 URL")

        except Exception as e:
            print(f"✗ 이미지 로드 실패: {e}")

    # 결과 저장
    if results:
        output_path = Path(csv_path).parent / f"{Path(csv_path).stem}_classified.csv"

        # 기존 분류 결과가 있으면 로드
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
        else:
            existing_df = df.copy()

        # 새로운 분류 결과 업데이트
        for result in results:
            if result['image_type']:
                existing_df.loc[result['index'], 'image_type'] = result['image_type']

        existing_df.to_csv(output_path, index=False)
        print(f"\n✓ 결과 저장: {output_path}")

        # 통계 출력
        print(f"\n{'='*80}")
        print("분류 통계")
        print(f"{'='*80}")
        print(existing_df['image_type'].value_counts())


def analyze_classification_results(csv_path: str):
    """분류 결과 분석"""
    df = pd.read_csv(csv_path)

    if 'image_type' not in df.columns:
        print("아직 분류되지 않았습니다.")
        return

    print(f"\n{'='*80}")
    print(f"이미지 타입 분류 결과: {csv_path}")
    print(f"{'='*80}\n")

    # 전체 통계
    total = len(df)
    classified = df['image_type'].notna().sum()

    print(f"총 제품 수: {total}")
    print(f"분류 완료: {classified} ({classified/total*100:.1f}%)")
    print(f"미분류: {total - classified}\n")

    # 타입별 통계
    print("타입별 분포:")
    type_counts = df['image_type'].value_counts()
    for image_type, count in type_counts.items():
        print(f"  {image_type}: {count} ({count/total*100:.1f}%)")

    # Domain gap 분석
    flat_count = type_counts.get('flat_product', 0)
    model_count = type_counts.get('model_wearing', 0)

    if flat_count > 0 and model_count > 0:
        print(f"\n⚠️  Domain Gap 감지:")
        print(f"   평면 제품: {flat_count} ({flat_count/classified*100:.1f}%)")
        print(f"   모델 착용: {model_count} ({model_count/classified*100:.1f}%)")


def batch_classify_by_url_pattern(csv_path: str):
    """
    URL 패턴으로 자동 분류 시도 (heuristic)
    """
    df = pd.read_csv(csv_path)

    if 'image_type' not in df.columns:
        df['image_type'] = ''

    # Supabase URL = 9oz internal (평면 제품일 가능성 높음)
    supabase_mask = df['image_url'].str.contains('supabase', na=False)

    # Naver shopping = 모델 착용일 가능성 높음
    naver_mask = df['image_url'].str.contains('shopping-phinf.pstatic.net', na=False)

    print(f"\nURL 패턴 기반 자동 분류:")
    print(f"  Supabase (9oz internal): {supabase_mask.sum()}개")
    print(f"  Naver shopping: {naver_mask.sum()}개")

    print(f"\n⚠️  주의: URL 패턴만으로는 정확하지 않을 수 있습니다.")
    print(f"   실제 이미지를 확인하여 수동 분류하는 것을 권장합니다.")

    confirm = input("\n자동 분류를 적용하시겠습니까? (y/n): ").strip().lower()

    if confirm == 'y':
        df.loc[supabase_mask & (df['image_type'] == ''), 'image_type'] = 'likely_flat_product'
        df.loc[naver_mask & (df['image_type'] == ''), 'image_type'] = 'likely_model_wearing'

        output_path = Path(csv_path).parent / f"{Path(csv_path).stem}_auto_classified.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ 저장 완료: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="이미지 타입 분류 도구")
    parser.add_argument("csv_path", type=str, help="CSV 파일 경로")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["classify", "analyze", "auto"],
        default="classify",
        help="실행 모드: classify(수동 분류), analyze(결과 분석), auto(자동 분류)"
    )
    parser.add_argument("--samples", type=int, default=10, help="분류할 샘플 수")
    parser.add_argument("--start", type=int, default=0, help="시작 인덱스")

    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {args.csv_path}")
        sys.exit(1)

    if args.mode == "classify":
        download_and_show_images(args.csv_path, args.samples, args.start)
    elif args.mode == "analyze":
        analyze_classification_results(args.csv_path)
    elif args.mode == "auto":
        batch_classify_by_url_pattern(args.csv_path)
