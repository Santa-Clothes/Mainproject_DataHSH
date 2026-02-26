"""
Supabase Data Loader
====================

Supabase에서 제품 데이터를 로드하는 유틸리티
"""

import os
from typing import Optional
import pandas as pd
from supabase import create_client, Client

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


class SupabaseLoader:
    """Supabase에서 데이터 로드"""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ):
        """
        Args:
            url: Supabase project URL
            key: Supabase API key (anon or service_role)
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and KEY must be provided or set in environment variables"
            )

        # Supabase 클라이언트 초기화
        self.client: Client = create_client(self.url, self.key)
        print(f"[Supabase] Connected to: {self.url}")

    def load_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        테이블 데이터를 DataFrame으로 로드

        Args:
            table_name: 테이블 이름
            limit: 최대 로드 개수 (None이면 전체)

        Returns:
            DataFrame with product data
        """
        print(f"[Supabase] Loading table: {table_name}")

        try:
            # Supabase 쿼리
            if limit:
                # limit 지정된 경우
                query = self.client.table(table_name).select("*").limit(limit)
                response = query.execute()
                df = pd.DataFrame(response.data)
            else:
                # 전체 데이터 로드 (페이지네이션 처리)
                all_data = []
                page_size = 1000
                offset = 0

                while True:
                    query = self.client.table(table_name).select("*").range(offset, offset + page_size - 1)
                    response = query.execute()

                    if not response.data:
                        break

                    all_data.extend(response.data)

                    if len(response.data) < page_size:
                        break

                    offset += page_size

                df = pd.DataFrame(all_data)

            print(f"[Supabase] Loaded {len(df)} rows from {table_name}")
            return df

        except Exception as e:
            raise RuntimeError(f"Failed to load table {table_name}: {e}")

    def load_nineoz_products(self, table_name: str = "internal_products") -> pd.DataFrame:
        """
        Nine Oz 제품 데이터 로드

        Args:
            table_name: Nine Oz 테이블 이름

        Returns:
            DataFrame with Nine Oz products
        """
        return self.load_table(table_name)

    def load_naver_products(self, table_name: str = "naver_products") -> pd.DataFrame:
        """
        Naver 제품 데이터 로드

        Args:
            table_name: Naver 테이블 이름

        Returns:
            DataFrame with Naver products
        """
        return self.load_table(table_name)


if __name__ == "__main__":
    # 테스트
    import argparse

    parser = argparse.ArgumentParser(description="Test Supabase Loader")
    parser.add_argument("--table", type=str, default="naver_products", help="Table name")
    parser.add_argument("--limit", type=int, default=10, help="Limit rows")
    args = parser.parse_args()

    loader = SupabaseLoader()
    df = loader.load_table(args.table, limit=args.limit)

    print("\n" + "=" * 80)
    print(f"Table: {args.table}")
    print(f"Shape: {df.shape}")
    print("=" * 80)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
