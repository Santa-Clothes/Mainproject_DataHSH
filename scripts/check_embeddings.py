"""
Supabase 임베딩 검증 스크립트
"""
import sys
from pathlib import Path
import json
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.supabase_loader import SupabaseLoader

def check_embeddings():
    loader = SupabaseLoader()

    print("="*80)
    print("Supabase 임베딩 검증")
    print("="*80)

    # Naver 제품 샘플 로드
    print("\n[1] Naver 제품 샘플 로드 (10개)...")
    df = loader.load_table('naver_products', limit=10)

    print(f"  [OK] 로드 완료: {len(df)}개")
    print(f"  컬럼: {df.columns.tolist()}")

    # 임베딩 체크
    print("\n[2] 임베딩 검증...")

    valid_count = 0
    invalid_count = 0
    null_count = 0

    for idx, row in df.iterrows():
        product_id = row['product_id']
        embedding = row['embedding']

        # Null 체크
        if embedding is None or (isinstance(embedding, float) and np.isnan(embedding)):
            print(f"  [X] {product_id}: NULL embedding")
            null_count += 1
            continue

        # 타입 체크
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except:
                print(f"  [X] {product_id}: JSON parsing failed")
                invalid_count += 1
                continue

        # 배열 변환
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        elif not isinstance(embedding, np.ndarray):
            print(f"  [X] {product_id}: Invalid type ({type(embedding)})")
            invalid_count += 1
            continue

        # 차원 체크
        if embedding.shape[0] != 768:
            print(f"  [X] {product_id}: Wrong dimension ({embedding.shape[0]}, expected: 768)")
            invalid_count += 1
            continue

        # 값 범위 체크
        emb_min = embedding.min()
        emb_max = embedding.max()
        emb_mean = embedding.mean()
        emb_std = embedding.std()

        # NaN/Inf 체크
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            print(f"  [X] {product_id}: Contains NaN or Inf")
            invalid_count += 1
            continue

        # 정상
        if idx == 0:
            print(f"  [OK] {product_id}: dim={embedding.shape[0]}, range=[{emb_min:.3f}, {emb_max:.3f}], mean={emb_mean:.3f}, std={emb_std:.3f}")
            print(f"       Sample: {embedding[:5]}")

        valid_count += 1

    print("\n[3] 검증 결과")
    print(f"  [OK] 정상: {valid_count}")
    print(f"  [X] 비정상: {invalid_count}")
    print(f"  [X] NULL: {null_count}")

    # Nine Oz도 체크
    print("\n[4] Nine Oz 제품 샘플 로드 (10개)...")
    df_nineoz = loader.load_table('internal_products', limit=10)
    print(f"  [OK] 로드 완료: {len(df_nineoz)}개")
    print(f"  컬럼: {df_nineoz.columns.tolist()}")

    if 'embedding' in df_nineoz.columns:
        nineoz_valid = 0
        nineoz_invalid = 0

        for idx, row in df_nineoz.iterrows():
            embedding = row['embedding']
            if embedding is None:
                nineoz_invalid += 1
            else:
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                embedding = np.array(embedding)
                if embedding.shape[0] == 768 and not np.isnan(embedding).any():
                    nineoz_valid += 1
                else:
                    nineoz_invalid += 1

        print(f"  [OK] Nine Oz 정상: {nineoz_valid}")
        print(f"  [X] Nine Oz 비정상: {nineoz_invalid}")
    else:
        print(f"  [!] Nine Oz 테이블에 embedding 컬럼 없음")

    print("\n" + "="*80)
    print("검증 완료!")
    print("="*80)

if __name__ == "__main__":
    check_embeddings()
