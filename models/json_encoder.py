"""
패션 메타데이터를 512차원 임베딩으로 변환하는 JSON 인코더 모델입니다.

이 모듈은 패션 아이템 메타데이터를 처리하고 CLIP 이미지 임베딩과
정렬된 임베딩으로 변환하는 핵심 JSONEncoder 클래스를 구현합니다.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class JSONEncoder(nn.Module):
    """
    패션 메타데이터를 512차원 임베딩으로 변환하는 JSON 인코더입니다.

    인코더는 다양한 유형의 범주형 필드를 처리합니다:
    - 단일 범주형: category, silhouette
    - 다중 범주형: style, material, detail
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
        dropout_rate: float = 0.1,
    ):
        """
        JSON 인코더를 초기화합니다.

        Args:
            vocab_sizes: 필드명을 어휘 크기에 매핑하는 딕셔너리
            embedding_dim: 필드 임베딩의 차원
            hidden_dim: MLP의 은닉층 차원
            output_dim: 최종 출력 차원 (512로 고정)
            dropout_rate: 정규화를 위한 드롭아웃 비율
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 필드별 임베딩 레이어
        self.category_emb = nn.Embedding(vocab_sizes["category"], embedding_dim)
        self.style_emb = nn.Embedding(vocab_sizes["style"], embedding_dim)
        self.silhouette_emb = nn.Embedding(vocab_sizes["silhouette"], embedding_dim)
        self.material_emb = nn.Embedding(vocab_sizes["material"], embedding_dim)
        self.detail_emb = nn.Embedding(vocab_sizes["detail"], embedding_dim)

        # MLP layers
        concat_dim = 5 * embedding_dim  # 5 fields
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the JSON Encoder.

        Args:
            batch: Dictionary containing:
                - 'category': [batch_size] - single categorical
                - 'style': [batch_size, max_style_len] - multi categorical
                - 'silhouette': [batch_size] - single categorical
                - 'material': [batch_size, max_material_len] - multi categorical
                - 'detail': [batch_size, max_detail_len] - multi categorical
                - 'style_mask': [batch_size, max_style_len] - padding mask
                - 'material_mask': [batch_size, max_material_len] - padding mask
                - 'detail_mask': [batch_size, max_detail_len] - padding mask

        Returns:
            torch.Tensor: [batch_size, 512] L2-normalized embeddings
        """
        # Process single categorical fields
        category_emb = self.category_emb(batch["category"])  # [batch_size, embedding_dim]
        silhouette_emb = self.silhouette_emb(batch["silhouette"])  # [batch_size, embedding_dim]

        # Process multi categorical fields with mean pooling
        style_emb = self._process_multi_categorical(
            batch["style"], batch["style_mask"], self.style_emb
        )
        material_emb = self._process_multi_categorical(
            batch["material"], batch["material_mask"], self.material_emb
        )
        detail_emb = self._process_multi_categorical(
            batch["detail"], batch["detail_mask"], self.detail_emb
        )

        # Concatenate all field embeddings
        concat_emb = torch.cat(
            [category_emb, style_emb, silhouette_emb, material_emb, detail_emb], dim=-1
        )  # [batch_size, 5 * embedding_dim]

        # Pass through MLP
        output = self.mlp(concat_emb)  # [batch_size, output_dim]

        # L2 normalize
        output = F.normalize(output, p=2, dim=-1)

        return output

    def _process_multi_categorical(
        self, ids: torch.Tensor, mask: torch.Tensor, embedding_layer: nn.Embedding
    ) -> torch.Tensor:
        """
        Process multi-categorical field with mean pooling.

        Args:
            ids: [batch_size, max_len] - token ids
            mask: [batch_size, max_len] - padding mask (1 for valid, 0 for padding)
            embedding_layer: Embedding layer for this field

        Returns:
            torch.Tensor: [batch_size, embedding_dim] - mean pooled embeddings
        """
        # Get embeddings
        embeddings = embedding_layer(ids)  # [batch_size, max_len, embedding_dim]

        # Apply mask and compute mean
        masked_embeddings = embeddings * mask.unsqueeze(-1)  # [batch_size, max_len, embedding_dim]

        # Sum over valid tokens
        sum_embeddings = masked_embeddings.sum(dim=1)  # [batch_size, embedding_dim]

        # Count valid tokens
        valid_counts = mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]

        # Avoid division by zero
        valid_counts = torch.clamp(valid_counts, min=1.0)

        # Mean pooling
        mean_embeddings = sum_embeddings / valid_counts  # [batch_size, embedding_dim]

        return mean_embeddings
