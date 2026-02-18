"""
JSON과 이미지 임베딩을 정렬하는 대조 학습 시스템입니다.

이 모듈은 대조 학습을 위해 JSON 인코더와 FashionCLIP/CLIP 이미지 인코더를
결합하는 ContrastiveLearner 클래스를 구현합니다. 
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

from .json_encoder import JSONEncoder


class ContrastiveLearner(nn.Module):
    """
    JSON과 이미지 임베딩을 정렬하는 대조 학습 시스템입니다.

    배치 내 네거티브 샘플링과 함께 InfoNCE 손실을 사용하여
    패션 이미지와 JSON 메타데이터 간의 공유 임베딩 공간을 학습합니다.
    FashionCLIP 또는 표준 CLIP을 지원합니다.
    """

    def __init__(
        self,
        json_encoder: JSONEncoder,
        clip_encoder: CLIPVisionModel,
        temperature: float = 0.07,
        using_fashionclip: bool = False,
    ):
        """
        대조 학습 시스템을 초기화합니다.

        Args:
            json_encoder: 메타데이터 처리를 위한 JSONEncoder 모델
            clip_encoder: FashionCLIP 또는 표준 CLIP 비전 모델 (고정됨)
            temperature: InfoNCE 손실을 위한 온도 매개변수
            using_fashionclip: FashionCLIP 사용 여부
        """
        super().__init__()

        self.json_encoder = json_encoder
        self.clip_encoder = clip_encoder
        self.temperature = temperature
        self.using_fashionclip = using_fashionclip

        # Determine output dimension and setup projection if needed
        clip_output_dim = self.clip_encoder.config.hidden_size  # 768 for ViT-B/32
        json_output_dim = self.json_encoder.output_dim  # 512

        if clip_output_dim != json_output_dim:
            # 차원을 맞추기 위한 프로젝션 레이어 추가
            self.image_projection = nn.Linear(clip_output_dim, json_output_dim)
        else:
            self.image_projection = None

        # Freeze CLIP encoder parameters
        for param in self.clip_encoder.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor, json_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for contrastive learning.

        Args:
            images: [batch_size, 3, 224, 224] - batch of images
            json_data: Dictionary containing JSON batch data

        Returns:
            torch.Tensor: InfoNCE loss value (scalar)
        """
        # Get image embeddings from CLIP/FashionCLIP
        # Note: No torch.no_grad() if fine-tuning is enabled
        image_features = self.clip_encoder(images).pooler_output

        # Project to target dimension if needed
        if self.image_projection is not None:
            image_embeddings = self.image_projection(image_features)
        else:
            image_embeddings = image_features

        # Normalize image embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

        # Get JSON embeddings from trainable encoder
        json_embeddings = self.json_encoder(json_data)

        # Compute InfoNCE loss
        loss = self._compute_infonce_loss(image_embeddings, json_embeddings)

        return loss

    def get_embeddings(
        self, images: torch.Tensor, json_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get both image and JSON embeddings without computing loss.

        Args:
            images: [batch_size, 3, 224, 224] - batch of images
            json_data: Dictionary containing JSON batch data

        Returns:
            Dictionary containing:
                - 'image_embeddings': [batch_size, 512]
                - 'json_embeddings': [batch_size, 512]
                - 'similarity_matrix': [batch_size, batch_size]
        """
        # Get image embeddings from CLIP/FashionCLIP
        with torch.no_grad():
            image_features = self.clip_encoder(images).pooler_output

        # Project to target dimension if needed
        if self.image_projection is not None:
            image_embeddings = self.image_projection(image_features)
        else:
            image_embeddings = image_features

        # Normalize image embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

        # Get JSON embeddings
        json_embeddings = self.json_encoder(json_data)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, json_embeddings.T)

        return {
            "image_embeddings": image_embeddings,
            "json_embeddings": json_embeddings,
            "similarity_matrix": similarity_matrix,
        }

    def _compute_infonce_loss(
        self, image_embeddings: torch.Tensor, json_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for contrastive learning.

        Args:
            image_embeddings: [batch_size, 512] - normalized image embeddings
            json_embeddings: [batch_size, 512] - normalized JSON embeddings

        Returns:
            torch.Tensor: InfoNCE loss (scalar)
        """
        batch_size = image_embeddings.size(0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, json_embeddings.T) / self.temperature

        # Create labels (positive pairs are on the diagonal)
        labels = torch.arange(batch_size, device=image_embeddings.device)

        # Compute InfoNCE loss (cross-entropy with similarity as logits)
        loss_i2j = F.cross_entropy(similarity_matrix, labels)
        loss_j2i = F.cross_entropy(similarity_matrix.T, labels)

        # Average both directions
        loss = (loss_i2j + loss_j2i) / 2.0

        return loss
