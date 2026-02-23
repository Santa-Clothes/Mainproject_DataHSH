"""
Config 테스트
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*80)
print("Testing Configuration Loading")
print("="*80)

# dotenv 확인
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # .env 파일 우선
    print("✓ python-dotenv installed and loaded (override=True)")
except ImportError:
    print("✗ python-dotenv NOT installed")
    print("  Run: pip install python-dotenv")

# 환경변수 직접 확인
import os
print(f"\nEnvironment Variables (raw):")
use_faiss_raw = os.getenv('USE_FAISS', 'NOT SET')
print(f"  USE_FAISS: '{use_faiss_raw}' (type: {type(use_faiss_raw).__name__})")
print(f"  USE_FAISS.lower(): '{use_faiss_raw.lower() if use_faiss_raw != 'NOT SET' else 'N/A'}'")
print(f"  USE_FAISS.lower() == 'true': {use_faiss_raw.lower() == 'true' if use_faiss_raw != 'NOT SET' else False}")
print(f"  FAISS_INDEX_PATH: {os.getenv('FAISS_INDEX_PATH', 'NOT SET')}")
print(f"  NINEOZ_CSV_PATH: {os.getenv('NINEOZ_CSV_PATH', 'NOT SET')}")
print(f"  NAVER_CSV_PATH: {os.getenv('NAVER_CSV_PATH', 'NOT SET')}")

# Config 클래스 테스트
from utils.config import get_system_config

config = get_system_config()
print(f"\nSystemConfig:")
print(f"  use_faiss: {config.use_faiss}")
print(f"  faiss_index_path: {config.faiss_index_path}")
print(f"  nineoz_csv_path: {config.nineoz_csv_path}")
print(f"  naver_csv_path: {config.naver_csv_path}")
print(f"  checkpoint_path: {config.checkpoint_path}")

# 파일 존재 확인
print(f"\nFile Existence Check:")
print(f"  FAISS index exists: {Path(config.faiss_index_path).exists()}")
print(f"  Nine Oz CSV exists: {Path(config.nineoz_csv_path).exists()}")
print(f"  Naver CSV exists: {Path(config.naver_csv_path).exists()}")
print(f"  Checkpoint exists: {Path(config.checkpoint_path).exists()}")

print("="*80)
