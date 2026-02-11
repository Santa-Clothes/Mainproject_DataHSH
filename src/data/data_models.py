"""
Core data models for the Fashion JSON Encoder system.

This module defines the data structures used throughout the system for representing
fashion items, processed batches, and model outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch


@dataclass
class FashionItem:
    """단일 패션 아이템 데이터"""

    image_path: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    category: str
    style: List[str]
    silhouette: str
    material: List[str]
    detail: List[str]


@dataclass
class ProcessedBatch:
    """학습용 배치 데이터"""

    images: torch.Tensor  # [batch_size, 3, 224, 224]
    category_ids: torch.Tensor  # [batch_size]
    style_ids: torch.Tensor  # [batch_size, max_style_len]
    silhouette_ids: torch.Tensor  # [batch_size]
    material_ids: torch.Tensor  # [batch_size, max_material_len]
    detail_ids: torch.Tensor  # [batch_size, max_detail_len]

    # 패딩 마스크 (다중 범주형 필드용)
    style_mask: torch.Tensor  # [batch_size, max_style_len]
    material_mask: torch.Tensor  # [batch_size, max_material_len]
    detail_mask: torch.Tensor  # [batch_size, max_detail_len]


@dataclass
class EmbeddingOutput:
    """임베딩 출력 결과"""

    image_embeddings: torch.Tensor  # [batch_size, 512]
    json_embeddings: torch.Tensor  # [batch_size, 512]
    similarity_matrix: torch.Tensor  # [batch_size, batch_size]
    loss: torch.Tensor  # scalar
