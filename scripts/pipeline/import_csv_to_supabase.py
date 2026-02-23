"""
CSV 데이터를 Supabase에 업로드하는 스크립트
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.supabase_loader import SupabaseLoader

def import_csv_to_supabase(
    csv_path: str,
    table_name: str,
    batch_size: int = 100,
):
    """
    CSV 파일을 Supabase 테이블에 업로드

    Args:
        csv_path: CSV 파일 경로
        table_name: Supabase 테이블 이름
        batch_size: 배치 크기 (한 번에 업로드할 row 수)
    """
    print(f"\n{'='*80}")
    print(f"Importing {csv_path} → Supabase table '{table_name}'")
    print(f"{'='*80}\n")

    # CSV 로드
    print(f"[1/3] Loading CSV file...")
    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} rows")

    # Supabase 연결
    print(f"\n[2/3] Connecting to Supabase...")
    loader = SupabaseLoader()
    print(f"  ✓ Connected")

    # 데이터 업로드 (배치 단위)
    print(f"\n[3/3] Uploading data in batches of {batch_size}...")

    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Uploading"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_rows)

        batch = df.iloc[start_idx:end_idx]

        # DataFrame을 dict 리스트로 변환
        records = batch.to_dict('records')

        # Supabase에 삽입
        try:
            loader.client.table(table_name).insert(records).execute()
        except Exception as e:
            print(f"\n  ✗ Error at batch {i+1}: {e}")
            raise

    print(f"\n{'='*80}")
    print(f"✓ Successfully uploaded {total_rows} rows to '{table_name}'")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import CSV to Supabase")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file path"
    )
    parser.add_argument(
        "--table",
        type=str,
        required=True,
        help="Supabase table name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for upload (default: 100)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Import both nineoz and naver tables"
    )

    args = parser.parse_args()

    if args.all:
        # 두 테이블 모두 업로드
        print("\n🚀 Importing both tables...\n")

        # Nine Oz
        import_csv_to_supabase(
            csv_path="data/csv/internal_products_rows.csv",
            table_name="internal_products",
            batch_size=args.batch_size
        )

        # Naver
        import_csv_to_supabase(
            csv_path="data/csv/naver_products_rows.csv",
            table_name="naver_products",
            batch_size=args.batch_size
        )

        print("\n✓ All tables imported successfully!")
    else:
        # 단일 테이블 업로드
        import_csv_to_supabase(
            csv_path=args.csv,
            table_name=args.table,
            batch_size=args.batch_size
        )
