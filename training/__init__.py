"""
Training module for Fashion JSON Encoder.

This module provides the training pipeline implementation including:
- FashionTrainer: Main trainer class for both standalone and contrastive learning
- Training script: Command-line interface for training
- Utilities for checkpoint management, logging, and evaluation
"""

from .trainer import FashionTrainer, create_trainer_from_data_module

__all__ = ["FashionTrainer", "create_trainer_from_data_module"]
