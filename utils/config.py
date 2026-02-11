"""
Configuration classes for the Fashion JSON Encoder system.

This module defines the training configuration and other system settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


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

    # 기본 데이터 경로
    data_root: Path = Path("C:/K-fashion")

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
