"""
Configuration classes for the Fashion JSON Encoder system.

This module defines the training configuration and other system settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # override=True: .env 파일이 시스템 환경변수보다 우선
    load_dotenv(override=True)
except ImportError:
    print("[WARNING] python-dotenv not installed. Using system environment variables only.")
    print("  Install with: pip install python-dotenv")


# K-Fashion 전체 카테고리 (23개)
ALL_CATEGORIES = [
    '레트로', '로맨틱', '리조트', '매니시', '모던',
    '밀리터리', '섹시', '소피스트케이티드', '스트리트', '스포티',
    '아방가르드', '오리엔탈', '웨스턴', '젠더리스', '컨트리',
    '클래식', '키치', '톰보이', '펑크', '페미닌',
    '프레피', '히피', '힙합'
]


@dataclass
class DataConfig:
    """데이터 경로 설정 (pathlib 사용)"""

    # 기본 데이터 경로 (환경변수 DATA_ROOT로 오버라이드 가능)
    data_root: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_ROOT", "C:/K-fashion"))
    )

    @property
    def train_data_path(self) -> Path:
        """학습 데이터 경로"""
        return self.data_root / "Training"

    @property
    def val_data_path(self) -> Path:
        """검증 데이터 경로"""
        return self.data_root / "Validation"

    @property
    def train_image_path(self) -> Path:
        """학습 이미지 경로"""
        return self.train_data_path / "원천데이터"

    @property
    def train_label_path(self) -> Path:
        """학습 라벨 경로"""
        return self.train_data_path / "라벨링데이터"

    @property
    def val_image_path(self) -> Path:
        """검증 이미지 경로"""
        return self.val_data_path / "원천데이터"

    @property
    def val_label_path(self) -> Path:
        """검증 라벨 경로"""
        return self.val_data_path / "라벨링데이터"

    def __post_init__(self):
        """경로 존재 여부 확인 (선택적)"""
        if not self.data_root.exists():
            print(f"WARNING: 데이터 경로가 존재하지 않습니다: {self.data_root}")
            print("   config.py에서 data_root를 실제 경로로 수정하세요.")


@dataclass
class TrainingConfig:
    """학습 하이퍼파라미터"""

    batch_size: int = 64
    learning_rate: float = 1e-4
    temperature: float = 0.1  # InfoNCE temperature (패션 도메인 최적화)
    embedding_dim: int = 128  # 필드별 embedding 차원
    hidden_dim: int = 256  # MLP hidden 차원
    output_dim: int = 512  # 최종 출력 차원 (고정)
    dropout_rate: float = 0.1
    weight_decay: float = 1e-5
    max_epochs: int = 100

    # 데이터 관련
    target_categories: List[str] = field(default_factory=lambda: ALL_CATEGORIES)
    image_size: int = 224
    crop_padding: float = 0.1  # BBox 크롭 시 패딩 비율


@dataclass
class TestConfig:
    """테스트용 설정 (빠른 실행을 위한 축소된 파라미터)"""

    batch_size: int = 4  # 테스트용 작은 배치
    learning_rate: float = 1e-3
    temperature: float = 0.07
    embedding_dim: int = 32  # 축소된 임베딩 차원
    hidden_dim: int = 64  # 축소된 은닉층 차원
    output_dim: int = 512  # 출력 차원은 고정
    dropout_rate: float = 0.1
    weight_decay: float = 1e-5
    max_epochs: int = 2  # 테스트용 짧은 에포크

    # 데이터 관련
    target_categories: List[str] = field(default_factory=lambda: ALL_CATEGORIES[:3])  # 테스트용 3개
    image_size: int = 224
    crop_padding: float = 0.1

    # 테스트 전용 설정
    num_samples: int = 20  # 테스트용 샘플 수
    fast_samples: int = 10  # 빠른 모드 샘플 수
    quick_samples: int = 5  # 매우 빠른 모드 샘플 수


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
