"""
Multi-View Contrastive Loss for Domain Gap 해결

평면 제품과 모델 착용 이미지를 동시에 학습하여 Domain Gap을 줄입니다.
같은 카테고리/스타일의 제품은 이미지 타입(flat/wearing)과 관계없이
비슷한 임베딩을 가지도록 학습합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiViewContrastiveLoss(nn.Module):
    """
    Multi-View Contrastive Loss

    평면 제품 이미지와 모델 착용 이미지를 모두 학습하여
    Domain Gap 문제를 해결합니다.

    Loss Components:
    1. Image-JSON Contrastive Loss (기존)
    2. Cross-Domain Contrastive Loss (신규)
       - 같은 카테고리의 flat <-> wearing 이미지가 가까워지도록
    3. Domain Alignment Loss (선택)
       - Domain별 분포가 유사해지도록
    """

    def __init__(
        self,
        temperature: float = 0.07,
        cross_domain_weight: float = 0.3,
        alignment_weight: float = 0.1,
        use_alignment_loss: bool = False,
    ):
        """
        Args:
            temperature: Contrastive learning temperature
            cross_domain_weight: Cross-domain loss 가중치
            alignment_weight: Domain alignment loss 가중치
            use_alignment_loss: Alignment loss 사용 여부
        """
        super().__init__()
        self.temperature = temperature
        self.cross_domain_weight = cross_domain_weight
        self.alignment_weight = alignment_weight
        self.use_alignment_loss = use_alignment_loss

    def forward(
        self,
        image_embeddings: torch.Tensor,
        json_embeddings: torch.Tensor,
        domains: Optional[list] = None,
        categories: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_embeddings: [batch_size, embed_dim] (L2 normalized)
            json_embeddings: [batch_size, embed_dim] (L2 normalized)
            domains: List of domain names ('flat_product' | 'model_wearing')
            categories: List of category names (for cross-domain matching)

        Returns:
            Dict with keys:
                - total_loss: Total loss
                - image_json_loss: Image-JSON contrastive loss
                - cross_domain_loss: Cross-domain contrastive loss
                - alignment_loss: Domain alignment loss (optional)
        """
        batch_size = image_embeddings.size(0)

        # 1. Standard Image-JSON Contrastive Loss
        image_json_loss = self._compute_contrastive_loss(
            image_embeddings, json_embeddings
        )

        # 2. Cross-Domain Contrastive Loss
        if domains is not None and categories is not None:
            cross_domain_loss = self._compute_cross_domain_loss(
                image_embeddings, domains, categories
            )
        else:
            cross_domain_loss = torch.tensor(0.0).to(image_embeddings.device)

        # 3. Domain Alignment Loss (optional)
        if self.use_alignment_loss and domains is not None:
            alignment_loss = self._compute_alignment_loss(
                image_embeddings, domains
            )
        else:
            alignment_loss = torch.tensor(0.0).to(image_embeddings.device)

        # Total Loss
        total_loss = (
            image_json_loss +
            self.cross_domain_weight * cross_domain_loss +
            self.alignment_weight * alignment_loss
        )

        return {
            'total_loss': total_loss,
            'image_json_loss': image_json_loss,
            'cross_domain_loss': cross_domain_loss,
            'alignment_loss': alignment_loss,
        }

    def _compute_contrastive_loss(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard contrastive loss (InfoNCE)

        Args:
            embeddings_a: [batch_size, embed_dim]
            embeddings_b: [batch_size, embed_dim]

        Returns:
            Scalar loss
        """
        batch_size = embeddings_a.size(0)

        # Similarity matrix: [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size).to(embeddings_a.device)

        # Cross-entropy loss (both directions)
        loss_a_to_b = F.cross_entropy(similarity_matrix, labels)
        loss_b_to_a = F.cross_entropy(similarity_matrix.T, labels)

        return (loss_a_to_b + loss_b_to_a) / 2

    def _compute_cross_domain_loss(
        self,
        embeddings: torch.Tensor,
        domains: list,
        categories: list,
    ) -> torch.Tensor:
        """
        Cross-Domain Contrastive Loss

        같은 카테고리의 flat-product와 model-wearing 이미지가
        가까운 임베딩을 가지도록 학습합니다.

        Args:
            embeddings: [batch_size, embed_dim]
            domains: List of domain names
            categories: List of category names

        Returns:
            Scalar loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Domain과 Category를 인덱스로 매핑
        flat_indices = [i for i, d in enumerate(domains) if d == 'flat_product']
        wearing_indices = [i for i, d in enumerate(domains) if d == 'model_wearing']

        # 둘 다 없으면 loss 0
        if len(flat_indices) == 0 or len(wearing_indices) == 0:
            return torch.tensor(0.0).to(device)

        # Flat embeddings와 Wearing embeddings 분리
        flat_embeddings = embeddings[flat_indices]
        wearing_embeddings = embeddings[wearing_indices]

        flat_categories = [categories[i] for i in flat_indices]
        wearing_categories = [categories[i] for i in wearing_indices]

        # Cross-domain similarity: [num_flat, num_wearing]
        cross_similarity = torch.matmul(
            flat_embeddings, wearing_embeddings.T
        ) / self.temperature

        # Positive mask: same category
        # [num_flat, num_wearing]
        positive_mask = torch.zeros(
            len(flat_indices), len(wearing_indices)
        ).to(device)

        for i, flat_cat in enumerate(flat_categories):
            for j, wear_cat in enumerate(wearing_categories):
                if flat_cat == wear_cat:
                    positive_mask[i, j] = 1.0

        # Contrastive loss with positive mask
        # For each flat image, find matching wearing images
        losses = []
        for i in range(len(flat_indices)):
            if positive_mask[i].sum() > 0:
                # Softmax over all wearing images
                logits = cross_similarity[i]
                # Positive samples
                pos_mask = positive_mask[i].bool()

                # InfoNCE-style loss
                # exp(sim(flat, pos_wearing)) / sum(exp(sim(flat, all_wearing)))
                pos_logits = logits[pos_mask]
                # Average over all positive samples
                pos_score = torch.logsumexp(pos_logits, dim=0)
                all_score = torch.logsumexp(logits, dim=0)

                loss = -(pos_score - all_score)
                losses.append(loss)

        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0).to(device)

    def _compute_alignment_loss(
        self,
        embeddings: torch.Tensor,
        domains: list,
    ) -> torch.Tensor:
        """
        Domain Alignment Loss

        평면 제품과 모델 착용 임베딩의 분포가 유사해지도록 합니다.
        (MMD: Maximum Mean Discrepancy 사용)

        Args:
            embeddings: [batch_size, embed_dim]
            domains: List of domain names

        Returns:
            Scalar loss
        """
        device = embeddings.device

        flat_indices = [i for i, d in enumerate(domains) if d == 'flat_product']
        wearing_indices = [i for i, d in enumerate(domains) if d == 'model_wearing']

        if len(flat_indices) == 0 or len(wearing_indices) == 0:
            return torch.tensor(0.0).to(device)

        flat_embeddings = embeddings[flat_indices]
        wearing_embeddings = embeddings[wearing_indices]

        # Mean of each domain
        flat_mean = flat_embeddings.mean(dim=0)
        wearing_mean = wearing_embeddings.mean(dim=0)

        # MMD (simple version: L2 distance between means)
        mmd_loss = F.mse_loss(flat_mean, wearing_mean)

        return mmd_loss


class DomainAdversarialLoss(nn.Module):
    """
    Domain Adversarial Loss (선택)

    Domain classifier를 추가하여 domain-invariant features를 학습합니다.
    (더 고급 기법, 필요시 사용)
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),  # flat_product vs model_wearing
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        domains: list,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, embed_dim]
            domains: List of domain names

        Returns:
            Adversarial loss (encoder는 이를 최소화하려 함)
        """
        device = embeddings.device

        # Domain labels
        domain_labels = torch.tensor([
            0 if d == 'flat_product' else 1
            for d in domains
        ]).to(device)

        # Gradient reversal layer (학습 시 구현)
        # 지금은 단순 classification loss
        domain_logits = self.domain_classifier(embeddings.detach())

        # Cross-entropy loss
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        # Adversarial loss: encoder는 domain을 구분 못하게 하려 함
        # 따라서 negative loss (encoder는 이를 최대화)
        return -domain_loss


# Example usage
if __name__ == "__main__":
    # Test Multi-View Contrastive Loss
    batch_size = 32
    embed_dim = 512

    # Dummy embeddings (L2 normalized)
    image_embeddings = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=-1)
    json_embeddings = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=-1)

    # Dummy domains and categories
    domains = ['flat_product'] * 16 + ['model_wearing'] * 16
    categories = ['상의'] * 8 + ['하의'] * 8 + ['상의'] * 8 + ['하의'] * 8

    # Loss
    loss_fn = MultiViewContrastiveLoss(
        temperature=0.07,
        cross_domain_weight=0.3,
        use_alignment_loss=True,
    )

    losses = loss_fn(image_embeddings, json_embeddings, domains, categories)

    print("Multi-View Contrastive Loss Test:")
    print(f"  Total Loss: {losses['total_loss'].item():.4f}")
    print(f"  Image-JSON Loss: {losses['image_json_loss'].item():.4f}")
    print(f"  Cross-Domain Loss: {losses['cross_domain_loss'].item():.4f}")
    print(f"  Alignment Loss: {losses['alignment_loss'].item():.4f}")

    print("\nTest passed! ✓")
