"""
Training loop implementation for Fashion JSON Encoder.

This module implements the training pipeline including JSON Encoder standalone training,
full contrastive learning training, optimizer setup, learning rate scheduling,
checkpointing, and validation metrics with real-time monitoring.
"""

import os
import time
import json
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPVisionModel

from models.json_encoder import JSONEncoder
from models.contrastive_learner import ContrastiveLearner
from data.fashion_dataset import FashionDataModule, ProcessedBatch
from utils.config import TrainingConfig
from utils.training_monitor import TrainingMonitor


class FashionTrainer:
    """
    Trainer class for Fashion JSON Encoder system.
    
    Handles both standalone JSON Encoder training and full contrastive learning
    with CLIP image encoder integration.
    """
    
    def __init__(self,
                 config: TrainingConfig,
                 vocab_sizes: Dict[str, int],
                 device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs',
                 finetune_clip: bool = False,
                 finetune_layers: int = 2):
        """
        Initialize the Fashion Trainer.
        
        Args:
            config: Training configuration
            vocab_sizes: Vocabulary sizes for each field
            device: Device to use for training ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            finetune_clip: Whether to fine-tune CLIP encoder
            finetune_layers: Number of last layers to unfreeze (2 or 4)
        """
        self.config = config
        self.vocab_sizes = vocab_sizes
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training monitor (simple version)
        self.monitor = TrainingMonitor(results_dir="results")
        
        # Initialize models
        self.json_encoder = None
        self.contrastive_learner = None
        self.clip_encoder = None
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize models
        self._initialize_models(finetune_clip=finetune_clip, finetune_layers=finetune_layers)
        
        # Initialize training components
        self._initialize_training_components()
        
        print(f"FashionTrainer initialized on device: {device}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Log directory: {self.log_dir}")
    
    def _initialize_models(self, finetune_clip: bool = False, finetune_layers: int = 2):
        """Initialize JSON Encoder and Contrastive Learner models.
        
        Args:
            finetune_clip: Whether to fine-tune CLIP encoder
            finetune_layers: Number of last layers to unfreeze (2 or 4)
        """
        # Initialize JSON Encoder
        self.json_encoder = JSONEncoder(
            vocab_sizes=self.vocab_sizes,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Initialize FashionCLIP from Hugging Face
        print("Loading FashionCLIP model from Hugging Face (patrickjohncyh/fashion-clip)...")
        try:
            from transformers import CLIPVisionModel
            self.clip_encoder = CLIPVisionModel.from_pretrained("patrickjohncyh/fashion-clip")
            self.clip_encoder = self.clip_encoder.to(self.device)
            self.using_fashionclip = True
            print("✓ FashionCLIP loaded successfully (fashion-specific)")
        except Exception as e:
            print(f"Warning: Could not load FashionCLIP ({e})")
            print("Falling back to standard CLIP (openai/clip-vit-base-patch32)...")
            from transformers import CLIPVisionModel
            self.clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_encoder = self.clip_encoder.to(self.device)
            self.using_fashionclip = False
        
        # Fine-tuning setup
        self.finetune_clip = finetune_clip
        if finetune_clip:
            self._setup_clip_finetuning(finetune_layers)
        
        # Initialize Contrastive Learner
        self.contrastive_learner = ContrastiveLearner(
            json_encoder=self.json_encoder,
            clip_encoder=self.clip_encoder,
            temperature=self.config.temperature,
            using_fashionclip=self.using_fashionclip
        ).to(self.device)
        
        # Count trainable parameters
        json_trainable = sum(p.numel() for p in self.json_encoder.parameters() if p.requires_grad)
        clip_trainable = sum(p.numel() for p in self.clip_encoder.parameters() if p.requires_grad)
        total_trainable = json_trainable + clip_trainable
        
        print(f"Models initialized:")
        print(f"  Image Encoder: {'FashionCLIP (fashion-specific)' if self.using_fashionclip else 'Standard CLIP'}")
        print(f"  Fine-tuning: {'Enabled (last {} layers)'.format(finetune_layers) if finetune_clip else 'Disabled (frozen)'}")
        print(f"  JSON Encoder parameters: {sum(p.numel() for p in self.json_encoder.parameters()):,}")
        print(f"  CLIP Encoder parameters: {sum(p.numel() for p in self.clip_encoder.parameters()):,}")
        print(f"  Trainable parameters:")
        print(f"    - JSON Encoder: {json_trainable:,}")
        print(f"    - CLIP Encoder: {clip_trainable:,}")
        print(f"    - Total: {total_trainable:,}")
    
    def _setup_clip_finetuning(self, num_layers: int = 2):
        """Setup CLIP fine-tuning by unfreezing last N layers.
        
        Args:
            num_layers: Number of last layers to unfreeze (2 or 4)
        """
        # First freeze all parameters
        for param in self.clip_encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last N layers
        total_layers = 12  # ViT-B/32 has 12 transformer layers
        layers_to_unfreeze = list(range(total_layers - num_layers, total_layers))
        
        print(f"  Unfreezing CLIP layers: {layers_to_unfreeze}")
        
        for name, param in self.clip_encoder.named_parameters():
            for layer_idx in layers_to_unfreeze:
                if f'encoder.layers.{layer_idx}' in name:
                    param.requires_grad = True
                    break
        
        # Also unfreeze post_layernorm and projection if they exist
        for name, param in self.clip_encoder.named_parameters():
            if 'post_layernorm' in name or 'visual_projection' in name:
                param.requires_grad = True
    
    def _initialize_training_components(self):
        """Initialize optimizer, scheduler, and tensorboard writer."""
        # Collect trainable parameters with different learning rates
        if hasattr(self, 'finetune_clip') and self.finetune_clip:
            # Differential learning rates for fine-tuning
            param_groups = [
                {
                    'params': [p for p in self.json_encoder.parameters() if p.requires_grad],
                    'lr': self.config.learning_rate,
                    'name': 'json_encoder'
                },
                {
                    'params': [p for p in self.clip_encoder.parameters() if p.requires_grad],
                    'lr': self.config.learning_rate * 0.1,  # 10x lower LR for CLIP
                    'name': 'clip_encoder'
                }
            ]
            self.optimizer = optim.Adam(param_groups, weight_decay=self.config.weight_decay)
            print(f"  Using differential learning rates:")
            print(f"    - JSON Encoder: {self.config.learning_rate}")
            print(f"    - CLIP Encoder: {self.config.learning_rate * 0.1} (10x lower)")
        else:
            # Standard optimizer (only JSON encoder)
            self.optimizer = optim.Adam(
                self.json_encoder.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        print("Training components initialized:")
        print(f"  Optimizer: Adam (lr={self.config.learning_rate}, wd={self.config.weight_decay})")
        print(f"  Scheduler: CosineAnnealingLR")
        print(f"  Tensorboard logs: {self.log_dir}")
    
    def train_json_encoder_standalone(self, 
                                    train_loader: DataLoader,
                                    val_loader: DataLoader,
                                    num_epochs: int = 5) -> Dict[str, Any]:
        """
        Train JSON Encoder in standalone mode for sanity checking.
        
        This method trains only the JSON Encoder without CLIP integration
        to verify that the model can learn meaningful representations from
        JSON metadata alone.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs for standalone training
            
        Returns:
            Dictionary containing training statistics and output distribution analysis
        """
        print(f"\n{'='*60}")
        print("STARTING JSON ENCODER STANDALONE TRAINING")
        print(f"{'='*60}")
        
        # Print dataset and dataloader info
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Train dataloader batches: {len(train_loader)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")
        print(f"Val dataloader batches: {len(val_loader)}")
        
        # Start monitoring
        self.monitor.start_training("Stage 1: JSON Encoder Standalone", num_epochs)
        
        self.json_encoder.train()
        
        # Simple MSE loss for standalone training (reconstruct random target)
        criterion = nn.MSELoss()
        
        # Training statistics
        train_losses = []
        val_losses = []
        output_stats = []
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device FIRST
                batch = self._move_batch_to_device(batch)
                
                # Convert ProcessedBatch to dict format
                json_batch = self._convert_batch_to_dict(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                embeddings = self.json_encoder(json_batch)
                
                # Create random target for sanity check (on same device)
                target = torch.randn_like(embeddings).to(self.device)
                loss = criterion(embeddings, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                # Log every 50 batches
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss, val_stats = self._validate_json_encoder_standalone(val_loader, criterion)
            val_losses.append(val_loss)
            output_stats.append(val_stats)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Output Stats: mean={val_stats['mean']:.4f}, std={val_stats['std']:.4f}, norm={val_stats['norm']:.4f}")
            
            # Update monitoring
            epoch_metrics = {
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'l2_norm': val_stats['norm']
            }
            self.monitor.update_epoch(epoch + 1, epoch_metrics)
            
            # Log to tensorboard
            self.writer.add_scalar('Standalone/Train_Loss', avg_train_loss, epoch)
            self.writer.add_scalar('Standalone/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Standalone/Output_Mean', val_stats['mean'], epoch)
            self.writer.add_scalar('Standalone/Output_Std', val_stats['std'], epoch)
            self.writer.add_scalar('Standalone/Output_Norm', val_stats['norm'], epoch)
        
        # Final output distribution analysis
        final_analysis = self._analyze_output_distribution(val_loader)
        
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'output_stats': output_stats,
            'final_analysis': final_analysis,
            'num_epochs': num_epochs
        }
        
        print(f"\n{'='*60}")
        print("JSON ENCODER STANDALONE TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")
        print(f"Output distribution analysis:")
        for key, value in final_analysis.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Finish monitoring
        self.monitor.finish_training()
        
        return results
    
    def _validate_json_encoder_standalone(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Validate JSON encoder in standalone mode."""
        self.json_encoder.eval()
        
        total_loss = 0.0
        all_embeddings = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                json_batch = self._convert_batch_to_dict(batch)
                embeddings = self.json_encoder(json_batch)
                
                # Random target for consistency with training (on same device)
                target = torch.randn_like(embeddings).to(self.device)
                loss = criterion(embeddings, target)
                
                total_loss += loss.item()
                all_embeddings.append(embeddings.cpu())
                num_batches += 1
        
        # Compute output statistics
        all_embeddings = torch.cat(all_embeddings, dim=0)
        stats = {
            'mean': all_embeddings.mean().item(),
            'std': all_embeddings.std().item(),
            'norm': torch.norm(all_embeddings, dim=-1).mean().item()
        }
        
        self.json_encoder.train()
        return total_loss / num_batches, stats
    
    def _analyze_output_distribution(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Analyze the output distribution of JSON encoder."""
        self.json_encoder.eval()
        
        all_embeddings = []
        field_embeddings = {
            'category': [],
            'style': [],
            'silhouette': [],
            'material': [],
            'detail': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                json_batch = self._convert_batch_to_dict(batch)
                embeddings = self.json_encoder(json_batch)
                all_embeddings.append(embeddings.cpu())
                
                # Analyze field-specific patterns (simplified)
                batch_size = embeddings.size(0)
                for field in field_embeddings.keys():
                    # Store some statistics per field (this is a simplified analysis)
                    field_embeddings[field].append(embeddings[:min(batch_size, 10)].cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Compute comprehensive statistics
        analysis = {
            'total_samples': all_embeddings.size(0),
            'embedding_dim': all_embeddings.size(1),
            'mean': all_embeddings.mean().item(),
            'std': all_embeddings.std().item(),
            'min': all_embeddings.min().item(),
            'max': all_embeddings.max().item(),
            'norm_mean': torch.norm(all_embeddings, dim=-1).mean().item(),
            'norm_std': torch.norm(all_embeddings, dim=-1).std().item(),
            'is_normalized': torch.allclose(torch.norm(all_embeddings, dim=-1), torch.ones(all_embeddings.size(0)), atol=1e-3)
        }
        
        self.json_encoder.train()
        return analysis
    
    def train_contrastive_learning(self,
                                 train_loader: DataLoader,
                                 val_loader: DataLoader,
                                 num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the full contrastive learning system.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config.max_epochs if None)
            
        Returns:
            Dictionary containing training history and final metrics
        """
        if num_epochs is None:
            num_epochs = self.config.max_epochs
        
        print(f"\n{'='*60}")
        print("STARTING CONTRASTIVE LEARNING TRAINING")
        print(f"{'='*60}")
        print(f"Training for {num_epochs} epochs")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Temperature: {self.config.temperature}")
        
        # Print dataset and dataloader info
        print(f"\nTrain dataset size: {len(train_loader.dataset)}")
        print(f"Train dataloader batches: {len(train_loader)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")
        print(f"Val dataloader batches: {len(val_loader)}")
        
        # Start monitoring
        self.monitor.start_training("Stage 2: Contrastive Learning", num_epochs)
        
        # Training history
        train_losses = []
        val_losses = []
        learning_rates = []
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader, epoch)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            learning_rates.append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            self.current_epoch = epoch + 1
            
            # Logging
            print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Val Metrics: {val_metrics}")
            
            # Update monitoring
            epoch_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'top1_accuracy': val_metrics.get('top1_accuracy', 0),
                'top5_accuracy': val_metrics.get('top5_accuracy', 0),
                'mrr': val_metrics.get('mean_reciprocal_rank', 0),
                'positive_similarity': val_metrics.get('avg_positive_similarity', 0),
                'negative_similarity': val_metrics.get('avg_negative_similarity', 0),
                'l2_norm': 1.0  # L2 normalized embeddings
            }
            self.monitor.update_epoch(epoch + 1, epoch_metrics)
            
            # Tensorboard logging
            self.writer.add_scalar('Training/Loss', train_loss, epoch)
            self.writer.add_scalar('Validation/Loss', val_loss, epoch)
            self.writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
            
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Validation/{metric_name}', metric_value, epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self._save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping check (optional)
            if self._should_early_stop(val_losses):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Final evaluation
        final_metrics = self._final_evaluation(val_loader)
        
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'final_metrics': final_metrics,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(train_losses)
        }
        
        print(f"\n{'='*60}")
        print("CONTRASTIVE LEARNING TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final metrics: {final_metrics}")
        
        # Finish monitoring
        self.monitor.finish_training()
        
        return results
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.contrastive_learner.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            json_batch = self._convert_batch_to_dict(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = self.contrastive_learner(batch.images, json_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.json_encoder.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                self.writer.add_scalar('Training/Batch_Loss', loss.item(), self.global_step)
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.contrastive_learner.eval()
        
        total_loss = 0.0
        all_similarities = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                json_batch = self._convert_batch_to_dict(batch)
                
                # Compute loss
                loss = self.contrastive_learner(batch.images, json_batch)
                total_loss += loss.item()
                
                # Compute embeddings for metrics
                embeddings = self.contrastive_learner.get_embeddings(batch.images, json_batch)
                similarities = embeddings['similarity_matrix']
                all_similarities.append(similarities.cpu())
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Compute validation metrics
        if all_similarities:
            # 각 배치의 similarity matrix를 개별적으로 처리
            all_metrics = []
            for similarities in all_similarities:
                batch_metrics = self._compute_validation_metrics(similarities)
                all_metrics.append(batch_metrics)
            
            # 평균 계산
            metrics = {}
            if all_metrics:
                for key in all_metrics[0].keys():
                    metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
            else:
                metrics = {'top1_accuracy': 0.0, 'top5_accuracy': 0.0, 'mean_reciprocal_rank': 0.0}
        else:
            metrics = {'top1_accuracy': 0.0, 'top5_accuracy': 0.0, 'mean_reciprocal_rank': 0.0}
        
        return avg_loss, metrics
    
    def _compute_validation_metrics(self, similarity_matrix: torch.Tensor) -> Dict[str, float]:
        """Compute validation metrics from similarity matrix."""
        batch_size = similarity_matrix.size(0)
        
        # Top-1 accuracy (diagonal should be highest in each row)
        top1_correct = (similarity_matrix.argmax(dim=1) == torch.arange(batch_size)).float().mean()
        
        # Top-5 accuracy
        top5_indices = similarity_matrix.topk(k=min(5, batch_size), dim=1)[1]
        top5_correct = (top5_indices == torch.arange(batch_size).unsqueeze(1)).any(dim=1).float().mean()
        
        # Mean reciprocal rank
        ranks = (similarity_matrix.argsort(dim=1, descending=True) == torch.arange(batch_size).unsqueeze(1)).nonzero()[:, 1] + 1
        mrr = (1.0 / ranks.float()).mean()
        
        return {
            'top1_accuracy': top1_correct.item(),
            'top5_accuracy': top5_correct.item(),
            'mean_reciprocal_rank': mrr.item(),
            'avg_positive_similarity': similarity_matrix.diag().mean().item()
        }
    
    def _final_evaluation(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Perform final comprehensive evaluation."""
        self.contrastive_learner.eval()
        
        all_image_embeddings = []
        all_json_embeddings = []
        all_similarities = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                json_batch = self._convert_batch_to_dict(batch)
                
                embeddings = self.contrastive_learner.get_embeddings(batch.images, json_batch)
                
                all_image_embeddings.append(embeddings['image_embeddings'].cpu())
                all_json_embeddings.append(embeddings['json_embeddings'].cpu())
                all_similarities.append(embeddings['similarity_matrix'].cpu())
        
        # Concatenate all embeddings (각 배치별로 처리)
        if all_image_embeddings and all_json_embeddings and all_similarities:
            # 임베딩은 차원이 같으므로 concatenate 가능
            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            all_json_embeddings = torch.cat(all_json_embeddings, dim=0)
            
            # Similarity matrix는 각 배치별로 메트릭 계산 후 평균
            batch_metrics = []
            for similarities in all_similarities:
                batch_metric = self._compute_validation_metrics(similarities)
                batch_metrics.append(batch_metric)
            
            # 평균 계산
            avg_metrics = {}
            if batch_metrics:
                for key in batch_metrics[0].keys():
                    avg_metrics[key] = sum(m[key] for m in batch_metrics) / len(batch_metrics)
        else:
            all_image_embeddings = torch.empty(0, 512)
            all_json_embeddings = torch.empty(0, 512)
            avg_metrics = {'top1_accuracy': 0.0, 'top5_accuracy': 0.0, 'mean_reciprocal_rank': 0.0}
        
        # Compute comprehensive metrics
        if all_image_embeddings.size(0) > 0:
            metrics = {
                'num_samples': all_image_embeddings.size(0),
                'embedding_dim': all_image_embeddings.size(1),
                
                # Embedding quality
                'image_embedding_norm': torch.norm(all_image_embeddings, dim=-1).mean().item(),
                'json_embedding_norm': torch.norm(all_json_embeddings, dim=-1).mean().item(),
                
                # Retrieval metrics (배치별 평균 사용)
                **avg_metrics
            }
        else:
            metrics = {
                'num_samples': 0,
                'embedding_dim': 512,
                'image_embedding_norm': 0.0,
                'json_embedding_norm': 0.0,
                **avg_metrics
            }
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'json_encoder_state_dict': self.json_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'vocab_sizes': self.vocab_sizes
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Notify monitor
        self.monitor.save_checkpoint(str(checkpoint_path))
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved: {best_path}")
            self.monitor.save_checkpoint(str(best_path))
        
        # Keep only last 3 checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoint_files) > keep_last:
            # Sort by modification time
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
            # Remove oldest files
            for old_file in checkpoint_files[:-keep_last]:
                old_file.unlink()
    
    def _should_early_stop(self, val_losses: list, patience: int = 10, min_delta: float = 1e-4) -> bool:
        """Check if early stopping should be triggered."""
        if len(val_losses) < patience + 1:
            return False
        
        # Check if validation loss hasn't improved for 'patience' epochs
        # Compare current loss with the best loss from the last 'patience' epochs (excluding current)
        current_loss = val_losses[-1]
        recent_losses = val_losses[-(patience+1):-1]  # Last 'patience' losses excluding current
        best_recent_loss = min(recent_losses)
        
        # Early stop if current loss is not significantly better than the best recent loss
        return current_loss >= best_recent_loss - min_delta
    
    def _convert_batch_to_dict(self, batch: ProcessedBatch) -> Dict[str, torch.Tensor]:
        """Convert ProcessedBatch to dictionary format for model input."""
        return {
            'category': batch.category_ids,
            'style': batch.style_ids,
            'silhouette': batch.silhouette_ids,
            'material': batch.material_ids,
            'detail': batch.detail_ids,
            'style_mask': batch.style_mask,
            'material_mask': batch.material_mask,
            'detail_mask': batch.detail_mask
        }
    
    def _move_batch_to_device(self, batch: ProcessedBatch) -> ProcessedBatch:
        """
        Move batch tensors to the specified device.
        Ensures all tensors including index tensors are on the correct device.
        """
        return ProcessedBatch(
            images=batch.images.to(self.device),
            category_ids=batch.category_ids.to(self.device),
            style_ids=batch.style_ids.to(self.device),
            silhouette_ids=batch.silhouette_ids.to(self.device),
            material_ids=batch.material_ids.to(self.device),
            detail_ids=batch.detail_ids.to(self.device),
            style_mask=batch.style_mask.to(self.device),
            material_mask=batch.material_mask.to(self.device),
            detail_mask=batch.detail_mask.to(self.device)
        )
    
    def _safe_move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Safely move a tensor to the target device.
        Handles both data and index tensors.
        """
        if tensor.device != torch.device(self.device):
            return tensor.to(self.device)
        return tensor
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.json_encoder.load_state_dict(checkpoint['json_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        
        return checkpoint
    
    def close(self):
        """Close tensorboard writer and cleanup resources."""
        if self.writer:
            self.writer.close()
        print("Trainer resources cleaned up")


def create_trainer_from_data_module(data_module: FashionDataModule,
                                  config: TrainingConfig,
                                  device: str = 'cuda',
                                  checkpoint_dir: str = 'checkpoints',
                                  log_dir: str = 'logs') -> FashionTrainer:
    """
    Create a FashionTrainer from a FashionDataModule.
    
    Args:
        data_module: Initialized FashionDataModule
        config: Training configuration
        device: Device to use for training
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        
    Returns:
        Initialized FashionTrainer
    """
    vocab_sizes = data_module.get_vocab_sizes()
    
    trainer = FashionTrainer(
        config=config,
        vocab_sizes=vocab_sizes,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    return trainer