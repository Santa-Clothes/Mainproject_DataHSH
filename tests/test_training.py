"""
Tests for the training system.

This module tests the training components including the FashionTrainer class
and training pipeline functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training.trainer import FashionTrainer
from utils.config import TrainingConfig, TestConfig
from src.data.data_models import ProcessedBatch


class TestFashionTrainer(unittest.TestCase):
    """Test cases for FashionTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / 'checkpoints'
        self.log_dir = Path(self.temp_dir) / 'logs'
        
        # 테스트용 설정 사용 (빠른 실행)
        self.config = TestConfig()
        
        self.vocab_sizes = {
            'category': 10,
            'style': 20,
            'silhouette': 15,
            'material': 25,
            'detail': 30
        }
        
        self.device = 'cpu'  # Use CPU for tests
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = FashionTrainer(
            config=self.config,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir)
        )
        
        # Check that models are initialized
        self.assertIsNotNone(trainer.json_encoder)
        self.assertIsNotNone(trainer.contrastive_learner)
        self.assertIsNotNone(trainer.clip_encoder)
        
        # Check that training components are initialized
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)
        self.assertIsNotNone(trainer.writer)
        
        # Check directories are created
        self.assertTrue(self.checkpoint_dir.exists())
        self.assertTrue(self.log_dir.exists())
        
        trainer.close()
    
    def test_convert_batch_to_dict(self):
        """Test batch conversion to dictionary format."""
        trainer = FashionTrainer(
            config=self.config,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir)
        )
        
        # Create mock batch
        batch = ProcessedBatch(
            images=torch.randn(2, 3, 224, 224),
            category_ids=torch.tensor([1, 2]),
            style_ids=torch.tensor([[1, 2], [3, 0]]),
            silhouette_ids=torch.tensor([1, 2]),
            material_ids=torch.tensor([[1, 0], [2, 3]]),
            detail_ids=torch.tensor([[1, 2], [3, 4]]),
            style_mask=torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
            material_mask=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            detail_mask=torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        )
        
        json_batch = trainer._convert_batch_to_dict(batch)
        
        # Check all required keys are present
        required_keys = ['category', 'style', 'silhouette', 'material', 'detail',
                        'style_mask', 'material_mask', 'detail_mask']
        for key in required_keys:
            self.assertIn(key, json_batch)
        
        # Check tensor shapes
        self.assertEqual(json_batch['category'].shape, (2,))
        self.assertEqual(json_batch['style'].shape, (2, 2))
        self.assertEqual(json_batch['style_mask'].shape, (2, 2))
        
        trainer.close()
    
    def test_move_batch_to_device(self):
        """Test moving batch to device."""
        trainer = FashionTrainer(
            config=self.config,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir)
        )
        
        # Create mock batch
        batch = ProcessedBatch(
            images=torch.randn(2, 3, 224, 224),
            category_ids=torch.tensor([1, 2]),
            style_ids=torch.tensor([[1, 2], [3, 0]]),
            silhouette_ids=torch.tensor([1, 2]),
            material_ids=torch.tensor([[1, 0], [2, 3]]),
            detail_ids=torch.tensor([[1, 2], [3, 4]]),
            style_mask=torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
            material_mask=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            detail_mask=torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        )
        
        moved_batch = trainer._move_batch_to_device(batch)
        
        # Check that all tensors are on the correct device
        self.assertEqual(moved_batch.images.device.type, self.device)
        self.assertEqual(moved_batch.category_ids.device.type, self.device)
        self.assertEqual(moved_batch.style_ids.device.type, self.device)
        
        trainer.close()
    
    def test_compute_validation_metrics(self):
        """Test validation metrics computation."""
        trainer = FashionTrainer(
            config=self.config,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir)
        )
        
        # Create mock similarity matrix (perfect alignment)
        batch_size = 4
        similarity_matrix = torch.eye(batch_size)  # Perfect diagonal
        
        metrics = trainer._compute_validation_metrics(similarity_matrix)
        
        # Check metric keys
        expected_keys = ['top1_accuracy', 'top5_accuracy', 'mean_reciprocal_rank', 'avg_positive_similarity']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # With perfect diagonal, top-1 accuracy should be 1.0
        self.assertAlmostEqual(metrics['top1_accuracy'], 1.0, places=4)
        self.assertAlmostEqual(metrics['avg_positive_similarity'], 1.0, places=4)
        
        trainer.close()
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        trainer = FashionTrainer(
            config=self.config,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir)
        )
        
        # Save a checkpoint
        trainer._save_checkpoint(epoch=0, val_loss=0.5, is_best=True)
        
        # Check that checkpoint files exist
        checkpoint_file = self.checkpoint_dir / 'checkpoint_epoch_1.pt'
        best_file = self.checkpoint_dir / 'best_model.pt'
        
        self.assertTrue(checkpoint_file.exists())
        self.assertTrue(best_file.exists())
        
        # Test loading
        original_state = trainer.json_encoder.state_dict()
        
        # Modify model state
        with torch.no_grad():
            for param in trainer.json_encoder.parameters():
                param.fill_(0.123)
        
        # Load checkpoint
        trainer.load_checkpoint(str(best_file))
        
        # Check that state is restored (approximately, due to potential precision differences)
        loaded_state = trainer.json_encoder.state_dict()
        for key in original_state:
            self.assertTrue(torch.allclose(original_state[key], loaded_state[key], atol=1e-6))
        
        trainer.close()
    
    @patch('training.trainer.CLIPVisionModel')
    def test_json_encoder_forward_pass(self, mock_clip):
        """Test JSON encoder forward pass."""
        # Mock CLIP model with proper config
        mock_clip_instance = Mock()
        mock_clip_instance.to.return_value = mock_clip_instance
        mock_clip_instance.parameters.return_value = []  # Empty list for parameters
        mock_clip_instance.config.hidden_size = 512  # Set proper hidden size
        mock_clip.from_pretrained.return_value = mock_clip_instance
        
        trainer = FashionTrainer(
            config=self.config,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir)
        )
        
        # Create mock JSON batch
        json_batch = {
            'category': torch.tensor([1, 2]),
            'style': torch.tensor([[1, 2], [3, 0]]),
            'silhouette': torch.tensor([1, 2]),
            'material': torch.tensor([[1, 0], [2, 3]]),
            'detail': torch.tensor([[1, 2], [3, 4]]),
            'style_mask': torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
            'material_mask': torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            'detail_mask': torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        }
        
        # Forward pass
        embeddings = trainer.json_encoder(json_batch)
        
        # Check output shape and normalization
        self.assertEqual(embeddings.shape, (2, 512))
        
        # Check L2 normalization
        norms = torch.norm(embeddings, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(2), atol=1e-6))
        
        trainer.close()


class TestTrainingUtilities(unittest.TestCase):
    """Test training utility functions."""
    
    def test_early_stopping_logic(self):
        """Test early stopping logic."""
        trainer = FashionTrainer(
            config=TestConfig(),  # 테스트용 설정 사용
            vocab_sizes={'category': 10, 'style': 10, 'silhouette': 10, 'material': 10, 'detail': 10},
            device='cpu',
            checkpoint_dir='temp_checkpoints',
            log_dir='temp_logs'
        )
        
        # Test case 1: Not enough epochs
        val_losses = [1.0, 0.9, 0.8]
        self.assertFalse(trainer._should_early_stop(val_losses, patience=5))
        
        # Test case 2: Improving losses
        val_losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        self.assertFalse(trainer._should_early_stop(val_losses, patience=5))
        
        # Test case 3: Stagnant losses (should trigger early stopping)
        val_losses = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.assertTrue(trainer._should_early_stop(val_losses, patience=5))
        
        trainer.close()


if __name__ == '__main__':
    unittest.main()