"""
Validation utilities for the Fashion JSON Encoder system.

This module provides validators for input data, model states, and training processes
to ensure system correctness and catch errors early.
"""

from typing import Dict

import torch
import torch.nn as nn


class InputValidator:
    """Validator for input data validation."""

    @staticmethod
    def validate_json_batch(batch: Dict[str, torch.Tensor]) -> None:
        """
        Validate JSON batch data for correctness.

        Args:
            batch: Dictionary containing JSON batch tensors

        Raises:
            ValueError: If validation fails
        """
        required_fields = ["category", "style", "silhouette", "material", "detail"]
        required_masks = ["style_mask", "material_mask", "detail_mask"]

        # Check required fields
        for field in required_fields:
            if field not in batch:
                raise ValueError(f"Required field '{field}' missing from batch")

        for mask in required_masks:
            if mask not in batch:
                raise ValueError(f"Required mask '{mask}' missing from batch")

        # Check batch size consistency
        batch_size = batch["category"].size(0)
        for field, tensor in batch.items():
            if tensor.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch in field '{field}': "
                    f"expected {batch_size}, got {tensor.size(0)}"
                )

        # Check data types
        for field, tensor in batch.items():
            if not tensor.dtype == torch.long:
                raise ValueError(f"Field '{field}' must be torch.long type, " f"got {tensor.dtype}")

        # Check tensor dimensions
        if len(batch["category"].shape) != 1:
            raise ValueError("Category field must be 1D tensor")
        if len(batch["silhouette"].shape) != 1:
            raise ValueError("Silhouette field must be 1D tensor")

        for field in ["style", "material", "detail"]:
            if len(batch[field].shape) != 2:
                raise ValueError(f"Multi-categorical field '{field}' must be 2D tensor")
            mask_field = f"{field}_mask"
            if batch[field].shape != batch[mask_field].shape:
                raise ValueError(f"Shape mismatch between '{field}' and '{mask_field}'")

    @staticmethod
    def validate_image_batch(images: torch.Tensor) -> None:
        """
        Validate image batch data for correctness.

        Args:
            images: Image batch tensor

        Raises:
            ValueError: If validation fails
        """
        if len(images.shape) != 4:
            raise ValueError(
                f"Images must be 4D tensor [batch, channels, height, width], "
                f"got shape {images.shape}"
            )

        if images.shape[1] != 3:
            raise ValueError(f"Images must have 3 channels (RGB), got {images.shape[1]}")

        if images.shape[2] != 224 or images.shape[3] != 224:
            raise ValueError(
                f"Images must be 224x224 pixels, " f"got {images.shape[2]}x{images.shape[3]}"
            )

        if not images.dtype == torch.float32:
            raise ValueError(f"Images must be float32 type, got {images.dtype}")

    @staticmethod
    def validate_style_description(style_desc) -> None:
        """
        Validate style description for API requests.

        Args:
            style_desc: Style description object

        Raises:
            ValueError: If validation fails
        """
        if not style_desc.category:
            raise ValueError("Category is required")

        if not style_desc.style or len(style_desc.style) == 0:
            raise ValueError("At least one style tag is required")

        # 카테고리 유효성 검사
        valid_categories = ["상의", "하의", "아우터", "원피스", "세트", "기타"]
        if style_desc.category not in valid_categories:
            raise ValueError(f"Invalid category. Must be one of: {valid_categories}")

        # 스타일 태그 길이 제한
        if len(style_desc.style) > 10:
            raise ValueError("Too many style tags. Maximum 10 allowed.")

        # 각 필드의 문자열 길이 제한
        for style_tag in style_desc.style:
            if len(style_tag) > 50:
                raise ValueError("Style tag too long. Maximum 50 characters.")

        if style_desc.material and len(style_desc.material) > 10:
            raise ValueError("Too many material tags. Maximum 10 allowed.")

        if style_desc.detail and len(style_desc.detail) > 15:
            raise ValueError("Too many detail tags. Maximum 15 allowed.")


class ModelValidator:
    """Validator for model state and output validation."""

    @staticmethod
    def validate_output_dimension(output: torch.Tensor, expected_dim: int = 512) -> None:
        """
        Validate model output dimensions.

        Args:
            output: Model output tensor
            expected_dim: Expected output dimension

        Raises:
            ValueError: If validation fails
        """
        if len(output.shape) != 2:
            raise ValueError(
                f"Output must be 2D tensor [batch_size, dim], " f"got shape {output.shape}"
            )

        if output.shape[-1] != expected_dim:
            raise ValueError(f"Output dimension must be {expected_dim}, " f"got {output.shape[-1]}")

    @staticmethod
    def validate_normalization(embeddings: torch.Tensor, tolerance: float = 1e-6) -> None:
        """
        Validate that embeddings are L2 normalized.

        Args:
            embeddings: Embedding tensor to validate
            tolerance: Tolerance for normalization check

        Raises:
            ValueError: If validation fails
        """
        norms = torch.norm(embeddings, dim=-1)
        expected_norms = torch.ones_like(norms)

        if not torch.allclose(norms, expected_norms, atol=tolerance):
            max_diff = torch.max(torch.abs(norms - expected_norms)).item()
            raise ValueError(
                f"Embeddings must be L2 normalized. " f"Max deviation from 1.0: {max_diff:.8f}"
            )

    @staticmethod
    def validate_clip_frozen(
        clip_model: nn.Module, original_params: Dict[str, torch.Tensor]
    ) -> None:
        """
        Validate that CLIP model parameters remain frozen.

        Args:
            clip_model: CLIP model to validate
            original_params: Dictionary of original parameter values

        Raises:
            ValueError: If validation fails
        """
        for name, param in clip_model.named_parameters():
            if name not in original_params:
                raise ValueError(f"Parameter '{name}' not found in original parameters")

            if not torch.equal(param, original_params[name]):
                raise ValueError(f"CLIP parameter '{name}' has been modified during training")

    @staticmethod
    def validate_vocab_sizes(vocab_sizes: Dict[str, int]) -> None:
        """
        Validate vocabulary sizes dictionary.

        Args:
            vocab_sizes: Dictionary mapping field names to vocabulary sizes

        Raises:
            ValueError: If validation fails
        """
        required_fields = ["category", "style", "silhouette", "material", "detail"]

        for field in required_fields:
            if field not in vocab_sizes:
                raise ValueError(f"Required vocabulary field '{field}' missing")

            if not isinstance(vocab_sizes[field], int) or vocab_sizes[field] <= 0:
                raise ValueError(
                    f"Vocabulary size for '{field}' must be positive integer, "
                    f"got {vocab_sizes[field]}"
                )


class TrainingErrorHandler:
    """Handler for training process errors and anomalies."""

    @staticmethod
    def handle_loss_explosion(loss: torch.Tensor, threshold: float = 100.0) -> None:
        """
        Detect and handle loss explosion during training.

        Args:
            loss: Current loss value
            threshold: Threshold for loss explosion detection

        Raises:
            RuntimeError: If loss explosion is detected
        """
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is not finite: {loss.item()}")

        if loss.item() > threshold:
            raise RuntimeError(f"Loss exploded: {loss.item():.4f} > {threshold}")

    @staticmethod
    def handle_gradient_issues(
        model: nn.Module, max_norm: float = 10.0, min_norm: float = 1e-8
    ) -> None:
        """
        Detect gradient explosion or vanishing during training.

        Args:
            model: Model to check gradients for
            max_norm: Maximum allowed gradient norm
            min_norm: Minimum expected gradient norm

        Raises:
            RuntimeError: If gradient issues are detected
        """
        total_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count == 0:
            raise RuntimeError("No gradients found in model")

        total_norm = total_norm ** (1.0 / 2)

        if total_norm > max_norm:
            raise RuntimeError(f"Gradient norm too large: {total_norm:.4f} > {max_norm}")
        elif total_norm < min_norm:
            raise RuntimeError(f"Gradient norm too small: {total_norm:.8f} < {min_norm}")

    @staticmethod
    def validate_batch_consistency(
        images: torch.Tensor, json_batch: Dict[str, torch.Tensor]
    ) -> None:
        """
        Validate that image and JSON batches have consistent sizes.

        Args:
            images: Image batch tensor
            json_batch: JSON batch dictionary

        Raises:
            ValueError: If batch sizes are inconsistent
        """
        image_batch_size = images.size(0)
        json_batch_size = json_batch["category"].size(0)

        if image_batch_size != json_batch_size:
            raise ValueError(
                f"Batch size mismatch: images={image_batch_size}, " f"json={json_batch_size}"
            )
