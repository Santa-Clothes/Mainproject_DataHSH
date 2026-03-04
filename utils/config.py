import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


@dataclass
class SystemConfig:
    """시스템 전반 설정 (환경변수 기반)"""

    # Data Source Configuration
    data_source: str = field(
        default_factory=lambda: os.getenv("DATA_SOURCE", "csv")  # csv or supabase
    )

    # Supabase Configuration
    supabase_url: Optional[str] = field(
        default_factory=lambda: os.getenv("SUPABASE_URL", None)
    )
    supabase_key: Optional[str] = field(
        default_factory=lambda: os.getenv("SUPABASE_KEY", None)
    )
    nineoz_table: str = field(
        default_factory=lambda: os.getenv("NINEOZ_TABLE", "internal_products")
    )
    naver_table: str = field(
        default_factory=lambda: os.getenv("NAVER_TABLE", "naver_products")
    )

    # CSV 데이터 경로 (fallback)
    nineoz_csv_path: str = field(
        default_factory=lambda: os.getenv(
            "NINEOZ_CSV_PATH",
            "data/csv/internal_products_rows.csv"
        )
    )
    naver_csv_path: str = field(
        default_factory=lambda: os.getenv(
            "NAVER_CSV_PATH",
            "data/csv/naver_products_rows.csv"
        )
    )

    # 모델 체크포인트 경로
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv(
            "CHECKPOINT_PATH",
            "checkpoints/multi_domain/best_model.pt"
        )
    )
    baseline_checkpoint_path: str = field(
        default_factory=lambda: os.getenv(
            "BASELINE_CHECKPOINT_PATH",
            "checkpoints/baseline_v5_rtx4090_optimized.pt"
        )
    )

    # API 설정
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    search_api_port: int = field(default_factory=lambda: int(os.getenv("SEARCH_API_PORT", "8001")))

    # 디바이스 설정
    device: Optional[str] = field(default_factory=lambda: os.getenv("DEVICE", None))

    # 임베딩 설정
    precompute_embeddings: bool = field(
        default_factory=lambda: os.getenv("PRECOMPUTE_EMBEDDINGS", "false").lower() == "true"
    )
    embedding_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    )

    # FAISS 벡터 검색
    use_faiss: bool = field(
        default_factory=lambda: os.getenv("USE_FAISS", "true").lower() == "true"
    )
    faiss_index_path: str = field(
        default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "data/indexes/naver.index")
    )

    # 로깅 설정
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/api.log"))

    def __post_init__(self):
        """경로 검증"""
        # Data source 확인
        if self.data_source == "supabase":
            if not self.supabase_url or not self.supabase_key:
                print(f"[WARNING] Supabase URL or KEY not configured")
                print(f"  Will fall back to CSV mode")
                self.data_source = "csv"
            else:
                print(f"[INFO] Using Supabase data source")
                print(f"  Nine Oz table: {self.nineoz_table}")
                print(f"  Naver table: {self.naver_table}")
        else:
            # CSV 파일 존재 확인
            if not Path(self.nineoz_csv_path).exists():
                print(f"[WARNING] Nine Oz CSV not found: {self.nineoz_csv_path}")

            if not Path(self.naver_csv_path).exists():
                print(f"[WARNING] Naver CSV not found: {self.naver_csv_path}")

        # 체크포인트 존재 확인 (optional)
        if self.checkpoint_path and not Path(self.checkpoint_path).exists():
            print(f"[INFO] Checkpoint not found: {self.checkpoint_path}")
            print(f"  Will use pretrained model instead")


# Convenience function to get system config
def get_system_config() -> SystemConfig:
    """시스템 설정 반환"""
    return SystemConfig()
