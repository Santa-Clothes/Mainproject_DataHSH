#!/usr/bin/env python3
"""
Baseline v3 Training Script - FashionCLIP Integration

This script trains the Fashion JSON Encoder with FashionCLIP instead of standard CLIP.
Expected improvements:
- Better fashion-specific feature extraction
- Improved alignment with fashion domain
- Higher Top-1 and Top-5 accuracy

Target Performance:
- Top-1: 30-35% (vs v2: 22%)
- Top-5: 70-75% (vs v2: 64%)
- MRR: 0.45-0.50 (vs v2: 0.41)
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.fashion_dataset import FashionDataModule
from training.trainer import FashionTrainer
from utils.config import TrainingConfig


def main():
    print("="*80)
    print("BASELINE V3 TRAINING - FashionCLIP Integration")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        temperature=0.1,  # Optimized from v2
        max_epochs=10,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout_rate=0.1,
        weight_decay=1e-5,
        target_categories=["레트로", "로맨틱", "리조트"]
    )
    
    print("Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Image Encoder: FashionCLIP (fashion-specific)")
    print()
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Setup data
    print("Loading dataset...")
    dataset_path = "C:/sample/라벨링데이터"
    
    data_module = FashionDataModule(
        dataset_path=dataset_path,
        target_categories=config.target_categories,
        batch_size=config.batch_size,
        num_workers=4,
        image_size=224
    )
    
    data_module.setup()
    vocab_sizes = data_module.get_vocab_sizes()
    
    print(f"Dataset loaded:")
    print(f"  Train samples: {len(data_module.train_dataset)}")
    print(f"  Val samples: {len(data_module.val_dataset)}")
    print(f"  Vocab sizes: {vocab_sizes}")
    print()
    
    # Initialize trainer
    print("Initializing trainer with FashionCLIP...")
    trainer = FashionTrainer(
        config=config,
        vocab_sizes=vocab_sizes,
        device=device,
        checkpoint_dir='checkpoints',
        log_dir='logs/baseline_v3'
    )
    print()
    
    # Training
    print("Starting training...")
    print("-" * 80)
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    start_time = time.time()
    
    # Train contrastive learning (skip standalone for v3)
    results = trainer.train_contrastive_learning(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.max_epochs
    )
    
    training_time = time.time() - start_time
    
    print("-" * 80)
    print("Training completed!")
    print(f"Total training time: {training_time/60:.1f} minutes")
    print()
    
    # Save results
    print("Saving results...")
    
    final_metrics = results['final_metrics']
    
    v3_results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "Fashion JSON Encoder Baseline v3",
        "version": "v3.0",
        "base_model": "baseline_v2_final_best_model.pt",
        "key_change": "FashionCLIP integration (fashion-specific image encoder)",
        "configuration": {
            "temperature": config.temperature,
            "batch_size": config.batch_size,
            "epochs": config.max_epochs,
            "learning_rate": config.learning_rate,
            "image_encoder": "FashionCLIP (patrickjohncyh/fashion-clip)",
            "dataset": f"K-Fashion {len(data_module.train_dataset) + len(data_module.val_dataset)} items",
            "class_distribution": {
                "레트로": 196,
                "로맨틱": 994,
                "리조트": 998
            }
        },
        "final_performance": {
            "top1_accuracy": final_metrics.get('top1_accuracy', 0.0),
            "top5_accuracy": final_metrics.get('top5_accuracy', 0.0),
            "mrr": final_metrics.get('mean_reciprocal_rank', 0.0),
            "validation_loss": results['best_val_loss'],
            "positive_similarity": final_metrics.get('avg_positive_similarity', 0.0)
        },
        "training_progression": {
            "train_losses": results['train_losses'],
            "val_losses": results['val_losses'],
            "learning_rates": results['learning_rates']
        },
        "improvements_over_v2": {
            "image_encoder": "Standard CLIP → FashionCLIP",
            "domain_adaptation": "Fashion-specific pre-training",
            "expected_gains": "Better fashion attribute understanding"
        },
        "training_time_minutes": training_time / 60,
        "notes": "v3 introduces FashionCLIP for improved fashion-specific image encoding"
    }
    
    # Save to file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "baseline_v3_fashionclip_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(v3_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {results_file}")
    print()
    
    # Print summary
    print("="*80)
    print("BASELINE V3 SUMMARY")
    print("="*80)
    print(f"Image Encoder: FashionCLIP (fashion-specific)")
    print(f"Top-1 Accuracy: {final_metrics.get('top1_accuracy', 0.0):.1%}")
    print(f"Top-5 Accuracy: {final_metrics.get('top5_accuracy', 0.0):.1%}")
    print(f"MRR: {final_metrics.get('mean_reciprocal_rank', 0.0):.4f}")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Positive Similarity: {final_metrics.get('avg_positive_similarity', 0.0):.4f}")
    print()
    
    # Comparison with v2
    print("Comparison with v2:")
    print(f"  v2 Top-1: 22.2% → v3 Top-1: {final_metrics.get('top1_accuracy', 0.0):.1%}")
    print(f"  v2 Top-5: 64.1% → v3 Top-5: {final_metrics.get('top5_accuracy', 0.0):.1%}")
    print(f"  v2 MRR: 0.407 → v3 MRR: {final_metrics.get('mean_reciprocal_rank', 0.0):.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
