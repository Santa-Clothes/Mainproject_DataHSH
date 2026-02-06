#!/usr/bin/env python3
"""
Baseline v4 Training Script - FashionCLIP Fine-tuning

This script fine-tunes the last 2 layers of FashionCLIP along with JSON Encoder.
Expected improvements over v3:
- Better domain adaptation to K-Fashion dataset
- Improved feature extraction for Korean fashion styles
- Higher Top-1 and Top-5 accuracy

Target Performance:
- Top-1: 35-40% (vs v3: 30.3%)
- Top-5: 73-75% (vs v3: 71.5%)
- MRR: 0.50-0.52 (vs v3: 0.484)
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
    print("BASELINE V4 TRAINING - FashionCLIP Fine-tuning")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,  # JSON Encoder LR
        temperature=0.1,
        max_epochs=15,  # Increased for fine-tuning
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout_rate=0.1,
        weight_decay=1e-5,
        target_categories=["레트로", "로맨틱", "리조트"]
    )
    
    print("Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate (JSON): {config.learning_rate}")
    print(f"  Learning rate (CLIP): {config.learning_rate * 0.1} (10x lower)")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Fine-tuning: Enabled (last 2 layers of FashionCLIP)")
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
    
    # Initialize trainer with fine-tuning enabled
    print("Initializing trainer with FashionCLIP fine-tuning...")
    trainer = FashionTrainer(
        config=config,
        vocab_sizes=vocab_sizes,
        device=device,
        checkpoint_dir='checkpoints',
        log_dir='logs/baseline_v4',
        finetune_clip=True,  # Enable fine-tuning
        finetune_layers=2     # Unfreeze last 2 layers
    )
    print()
    
    # Load v3 checkpoint as starting point
    v3_checkpoint = Path("checkpoints/best_model.pt")
    if v3_checkpoint.exists():
        print(f"Loading v3 checkpoint: {v3_checkpoint}")
        try:
            trainer.load_checkpoint(str(v3_checkpoint))
            print("✓ v3 checkpoint loaded successfully")
            print("  Starting from v3 performance as baseline")
        except Exception as e:
            print(f"Warning: Could not load v3 checkpoint ({e})")
            print("  Starting from scratch")
        print()
    
    # Training
    print("Starting fine-tuning...")
    print("-" * 80)
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    start_time = time.time()
    
    # Train with fine-tuning
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
    
    v4_results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "Fashion JSON Encoder Baseline v4",
        "version": "v4.0",
        "base_model": "baseline_v3_fashionclip (fine-tuned)",
        "key_change": "FashionCLIP fine-tuning (last 2 layers unfrozen)",
        "configuration": {
            "temperature": config.temperature,
            "batch_size": config.batch_size,
            "epochs": config.max_epochs,
            "learning_rate_json": config.learning_rate,
            "learning_rate_clip": config.learning_rate * 0.1,
            "image_encoder": "FashionCLIP (patrickjohncyh/fashion-clip) - Fine-tuned",
            "finetuned_layers": "Last 2 layers (layers 10-11)",
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
        "improvements_over_v3": {
            "approach": "Fine-tuning last 2 layers of FashionCLIP",
            "learning_strategy": "Differential learning rates (CLIP: 10x lower)",
            "expected_gains": "Better K-Fashion domain adaptation"
        },
        "training_time_minutes": training_time / 60,
        "notes": "v4 fine-tunes FashionCLIP for K-Fashion dataset adaptation"
    }
    
    # Save to file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "baseline_v4_finetuned_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(v4_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {results_file}")
    print()
    
    # Print summary
    print("="*80)
    print("BASELINE V4 SUMMARY")
    print("="*80)
    print(f"Approach: FashionCLIP Fine-tuning (last 2 layers)")
    print(f"Top-1 Accuracy: {final_metrics.get('top1_accuracy', 0.0):.1%}")
    print(f"Top-5 Accuracy: {final_metrics.get('top5_accuracy', 0.0):.1%}")
    print(f"MRR: {final_metrics.get('mean_reciprocal_rank', 0.0):.4f}")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Positive Similarity: {final_metrics.get('avg_positive_similarity', 0.0):.4f}")
    print()
    
    # Comparison with v3
    print("Comparison with v3:")
    print(f"  v3 Top-1: 30.3% → v4 Top-1: {final_metrics.get('top1_accuracy', 0.0):.1%}")
    print(f"  v3 Top-5: 71.5% → v4 Top-5: {final_metrics.get('top5_accuracy', 0.0):.1%}")
    print(f"  v3 MRR: 0.484 → v4 MRR: {final_metrics.get('mean_reciprocal_rank', 0.0):.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
