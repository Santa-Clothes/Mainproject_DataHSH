"""
로컬 .npz → Supabase PostgreSQL 직접 BULK 업로드
=================================================

2단계: save_embeddings_npy.py로 저장한 .npz 파일을 psycopg2 COPY로
        Supabase에 한 번에 올린다. REST API 대신 직접 DB 연결 사용.

사전 준비:
  pip install psycopg2-binary

.env에 추가 필요:
  SUPABASE_DB_URL=postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres

  Supabase 대시보드 → Settings → Database → Connection string (Session mode) 복사

실행:
  python scripts/upload_embeddings_pg.py --npz data/embeddings/naver_embeddings.npz
"""

import sys
import io
import os
import argparse
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("[ERROR] psycopg2가 설치되지 않았습니다.")
    print("  pip install psycopg2-binary")
    sys.exit(1)


def get_db_url() -> str:
    url = os.getenv("SUPABASE_DB_URL")
    if not url:
        print("[ERROR] SUPABASE_DB_URL 환경변수가 없습니다.")
        print()
        print("  Supabase 대시보드 → Settings → Database → Connection string")
        print("  (Session mode, port 5432) 복사 후 .env에 추가:")
        print()
        print("  SUPABASE_DB_URL=postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres")
        sys.exit(1)
    return url


def upload_with_copy(npz_path: str, table: str = "naver_products", dry_run: bool = False):
    print("\n" + "=" * 80)
    print("Step 2: .npz → Supabase PostgreSQL BULK 업로드")
    print("=" * 80)

    # 1. .npz 로드
    print(f"\n[1] .npz 파일 로드: {npz_path}")
    if not Path(npz_path).exists():
        print(f"[ERROR] 파일 없음: {npz_path}")
        print("  먼저 python scripts/save_embeddings_npy.py 실행")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]      # shape (N, 768)
    product_ids = data["product_ids"]    # shape (N,)

    print(f"  임베딩: {embeddings.shape}  (dtype={embeddings.dtype})")
    print(f"  제품 수: {len(product_ids)}")

    if dry_run:
        print("\n[Dry Run] 실제 업로드 건너뜀")
        return

    # 2. DB 연결
    print(f"\n[2] PostgreSQL 연결...")
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()
    print("  [OK] 연결 성공")

    try:
        # 3. 임시 테이블 생성
        print(f"\n[3] 임시 테이블 생성...")
        cur.execute("""
            CREATE TEMP TABLE _tmp_embeddings (
                product_id TEXT,
                embedding   vector(768)
            ) ON COMMIT DROP;
        """)
        print("  [OK] _tmp_embeddings 생성")

        # 4. COPY로 임시 테이블에 대량 삽입
        print(f"\n[4] COPY로 데이터 삽입 ({len(product_ids)}개)...")
        buffer = io.StringIO()
        for pid, emb in zip(product_ids, embeddings):
            vec_str = "[" + ",".join(f"{v:.8f}" for v in emb.tolist()) + "]"
            buffer.write(f"{pid}\t{vec_str}\n")
        buffer.seek(0)

        cur.copy_expert(
            "COPY _tmp_embeddings (product_id, embedding) FROM STDIN WITH (FORMAT text)",
            buffer,
        )
        print(f"  [OK] COPY 완료")

        # 5. 본 테이블 UPDATE
        print(f"\n[5] {table} 테이블 UPDATE...")
        cur.execute(f"""
            UPDATE {table} AS t
            SET    embedding = tmp.embedding
            FROM   _tmp_embeddings AS tmp
            WHERE  t.product_id::TEXT = tmp.product_id;
        """)
        updated = cur.rowcount
        print(f"  [OK] 업데이트된 행: {updated}개")

        # 6. 커밋
        conn.commit()
        print(f"\n[OK] 커밋 완료!")

    except Exception as exc:
        conn.rollback()
        print(f"\n[ERROR] 롤백됨: {exc}")
        raise

    finally:
        cur.close()
        conn.close()

    print("\n" + "=" * 80)
    print("업로드 완료!")
    print("=" * 80)
    print("\n다음 단계:")
    print("  python scripts/build_supabase_faiss_index.py")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=".npz → Supabase BULK 업로드 (psycopg2 COPY)")
    parser.add_argument(
        "--npz",
        default="data/embeddings/naver_embeddings.npz",
        help=".npz 파일 경로",
    )
    parser.add_argument(
        "--table",
        default="naver_products",
        help="대상 테이블 이름",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="연결 없이 파일 확인만",
    )
    args = parser.parse_args()

    upload_with_copy(npz_path=args.npz, table=args.table, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
