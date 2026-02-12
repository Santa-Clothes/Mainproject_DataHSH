"""
임베딩 파이프라인 간단 테스트

실제 CSV 데이터로 임베딩 생성 테스트
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from models.embedding_generator import create_embedding_generator
from utils.config import get_system_config

def main():
    print("\n" + "="*80)
    print("임베딩 파이프라인 테스트")
    print("="*80)

    # 설정 로드
    config = get_system_config()

    # CSV 로드
    print(f"\n[1] CSV 데이터 로드")
    print(f"  Nine Oz: {config.nineoz_csv_path}")
    nineoz_df = pd.read_csv(config.nineoz_csv_path)
    print(f"  [OK] {len(nineoz_df)} 제품 로드됨")

    print(f"\n  Naver: {config.naver_csv_path}")
    naver_df = pd.read_csv(config.naver_csv_path)
    print(f"  [OK] {len(naver_df)} 제품 로드됨")

    # 임베딩 생성기 초기화
    print(f"\n[2] 임베딩 생성기 초기화")
    generator = create_embedding_generator(
        checkpoint_path=config.checkpoint_path,
        device=config.device
    )

    # 샘플 이미지로 테스트 (첫 3개)
    print(f"\n[3] 샘플 이미지 임베딩 생성 테스트")

    # Nine Oz 샘플
    print(f"\n  Nine Oz 샘플 (처음 3개):")
    for idx in range(min(3, len(nineoz_df))):
        row = nineoz_df.iloc[idx]
        image_url = row['image_url']
        product_name = row.get('product_name', row.get('제품명', 'Unknown'))

        print(f"\n  [{idx+1}] {product_name}")
        print(f"      URL: {image_url[:60]}...")

        try:
            embedding = generator.generate_embedding(image_url, normalize=True)
            print(f"      [OK] Embedding shape: {embedding.shape}")
            print(f"      [OK] Embedding norm: {embedding.sum():.4f}")
        except Exception as e:
            print(f"      [ERROR] Failed: {e}")

    # Naver 샘플
    print(f"\n  Naver 샘플 (처음 3개):")
    for idx in range(min(3, len(naver_df))):
        row = naver_df.iloc[idx]
        image_url = row['image_url']
        title = row['title']

        print(f"\n  [{idx+1}] {title}")
        print(f"      URL: {image_url[:60]}...")

        try:
            embedding = generator.generate_embedding(image_url, normalize=True)
            print(f"      [OK] Embedding shape: {embedding.shape}")
            print(f"      [OK] Embedding norm: {embedding.sum():.4f}")
        except Exception as e:
            print(f"      [ERROR] Failed: {e}")

    print("\n" + "="*80)
    print("[OK] 임베딩 파이프라인 테스트 완료!")
    print("="*80)

if __name__ == "__main__":
    main()
