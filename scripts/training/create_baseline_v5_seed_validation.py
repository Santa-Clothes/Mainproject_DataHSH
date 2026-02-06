"""
Baseline v5: Seed Validation Experiment
========================================

목적: 다양한 random seed로 v5 성능 재현성 검증

실험:
- Seed 42 (기존)
- Seed 123
- Seed 456
- Seed 789
- Seed 2024

각 seed별로 5 epoch만 학습하여 빠르게 검증
"""

import os
import sys
import json
import torch
from pathlib import Path
import random
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.fashion_dataset import FashionDataModule, create_validation_transforms
from training.trainer import FashionTrainer
from utils.config import TrainingConfig
from torchvision import transforms
import torch.nn as nn


def set_all_seeds(seed: int):
    """모든 random seed 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic 설정 (재현성 보장)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_enhanced_augmented_transforms(image_size: int = 224) -> transforms.Compose:
    """v5용 강화된 이미지 증강"""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(
            p=0.2,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3)
        )
    ])


def train_with_seed(seed: int, num_epochs: int = 5):
    """특정 seed로 학습 실행"""
    
    print(f"\n{'='*80}")
    print(f"Training with Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Seed 설정
    set_all_seeds(seed)
    
    # Device 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 데이터 경로
    dataset_path = "C:/sample/라벨링데이터"
    
    # Training configuration
    config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        max_epochs=num_epochs,
        temperature=0.1,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout_rate=0.1,
        weight_decay=1e-5
    )
    
    # Data module 설정
    data_module = FashionDataModule(
        dataset_path=dataset_path,
        target_categories=['레트로', '로맨틱', '리조트'],
        batch_size=config.batch_size,
        num_workers=0,
        train_split=0.8,
        image_size=224,
        augment_prob=0.5
    )
    
    data_module.setup()
    
    # 강화된 augmentation 적용
    enhanced_transforms = create_enhanced_augmented_transforms(image_size=224)
    data_module.train_dataset.image_transforms = enhanced_transforms
    
    # 클래스 밸런싱 DataLoader
    train_loader = data_module.train_dataloader(use_class_balanced=True)
    val_loader = data_module.val_dataloader()
    
    # Trainer 초기화
    vocab_sizes = data_module.get_vocab_sizes()
    
    trainer = FashionTrainer(
        config=config,
        vocab_sizes=vocab_sizes,
        device=device,
        checkpoint_dir=f'checkpoints/seed_{seed}',
        log_dir=f'logs/baseline_v5_seed_{seed}',
        finetune_clip=False
    )
    
    # Training
    results = trainer.train_contrastive_learning(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )
    
    # 결과 추출
    final_metrics = results['final_metrics']
    
    result_summary = {
        'seed': seed,
        'num_epochs': num_epochs,
        'top1_accuracy': final_metrics.get('top1_accuracy', 0),
        'top5_accuracy': final_metrics.get('top5_accuracy', 0),
        'mean_reciprocal_rank': final_metrics.get('mean_reciprocal_rank', 0),
        'best_val_loss': results['best_val_loss'],
        'train_losses': results['train_losses'],
        'val_losses': results['val_losses']
    }
    
    # Cleanup
    trainer.close()
    
    return result_summary


def main():
    print("="*80)
    print("Fashion JSON Encoder - v5 Seed Validation")
    print("다양한 Random Seed로 성능 재현성 검증")
    print("="*80)
    
    # 테스트할 seed 값들
    seeds = [42, 123, 456, 789, 2024]
    num_epochs = 5  # 빠른 검증을 위해 5 epoch만
    
    print(f"\n📊 실험 설정:")
    print(f"   Seeds: {seeds}")
    print(f"   Epochs per seed: {num_epochs}")
    print(f"   Total experiments: {len(seeds)}")
    
    # 각 seed로 학습
    all_results = []
    
    for seed in seeds:
        try:
            result = train_with_seed(seed, num_epochs)
            all_results.append(result)
            
            print(f"\n✅ Seed {seed} 완료:")
            print(f"   Top-1: {result['top1_accuracy']*100:.1f}%")
            print(f"   Top-5: {result['top5_accuracy']*100:.1f}%")
            print(f"   MRR: {result['mean_reciprocal_rank']:.3f}")
            
        except Exception as e:
            print(f"\n❌ Seed {seed} 실패: {e}")
            continue
    
    # 통계 분석
    if all_results:
        top1_values = [r['top1_accuracy'] for r in all_results]
        top5_values = [r['top5_accuracy'] for r in all_results]
        mrr_values = [r['mean_reciprocal_rank'] for r in all_results]
        
        statistics = {
            'num_experiments': len(all_results),
            'seeds_tested': [r['seed'] for r in all_results],
            'top1_accuracy': {
                'mean': float(np.mean(top1_values)),
                'std': float(np.std(top1_values)),
                'min': float(np.min(top1_values)),
                'max': float(np.max(top1_values)),
                'values': top1_values
            },
            'top5_accuracy': {
                'mean': float(np.mean(top5_values)),
                'std': float(np.std(top5_values)),
                'min': float(np.min(top5_values)),
                'max': float(np.max(top5_values)),
                'values': top5_values
            },
            'mean_reciprocal_rank': {
                'mean': float(np.mean(mrr_values)),
                'std': float(np.std(mrr_values)),
                'min': float(np.min(mrr_values)),
                'max': float(np.max(mrr_values)),
                'values': mrr_values
            },
            'detailed_results': all_results
        }
        
        # 결과 저장
        results_path = "results/baseline_v5_seed_validation.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print("📊 Seed Validation 결과 요약")
        print(f"{'='*80}\n")
        
        print(f"실험 횟수: {len(all_results)}/{len(seeds)}")
        print(f"테스트된 Seeds: {statistics['seeds_tested']}\n")
        
        print(f"Top-1 Accuracy:")
        print(f"   평균: {statistics['top1_accuracy']['mean']*100:.1f}%")
        print(f"   표준편차: {statistics['top1_accuracy']['std']*100:.1f}%")
        print(f"   범위: {statistics['top1_accuracy']['min']*100:.1f}% ~ {statistics['top1_accuracy']['max']*100:.1f}%")
        
        print(f"\nTop-5 Accuracy:")
        print(f"   평균: {statistics['top5_accuracy']['mean']*100:.1f}%")
        print(f"   표준편차: {statistics['top5_accuracy']['std']*100:.1f}%")
        print(f"   범위: {statistics['top5_accuracy']['min']*100:.1f}% ~ {statistics['top5_accuracy']['max']*100:.1f}%")
        
        print(f"\nMean Reciprocal Rank:")
        print(f"   평균: {statistics['mean_reciprocal_rank']['mean']:.3f}")
        print(f"   표준편차: {statistics['mean_reciprocal_rank']['std']:.3f}")
        print(f"   범위: {statistics['mean_reciprocal_rank']['min']:.3f} ~ {statistics['mean_reciprocal_rank']['max']:.3f}")
        
        # 재현성 평가
        top1_cv = statistics['top1_accuracy']['std'] / statistics['top1_accuracy']['mean']
        top5_cv = statistics['top5_accuracy']['std'] / statistics['top5_accuracy']['mean']
        
        print(f"\n재현성 평가 (변동계수):")
        print(f"   Top-1 CV: {top1_cv*100:.1f}%")
        print(f"   Top-5 CV: {top5_cv*100:.1f}%")
        
        if top1_cv < 0.05 and top5_cv < 0.05:
            print(f"   ✅ 매우 안정적 (CV < 5%)")
        elif top1_cv < 0.10 and top5_cv < 0.10:
            print(f"   ✅ 안정적 (CV < 10%)")
        else:
            print(f"   ⚠️  변동성 있음 (CV ≥ 10%)")
        
        # v5 원본 결과와 비교
        v5_original = {
            'top1': 0.471,
            'top5': 0.871,
            'mrr': 0.638
        }
        
        print(f"\n{'='*80}")
        print("v5 원본 결과와 비교 (15 epochs)")
        print(f"{'='*80}\n")
        
        print(f"Top-1:")
        print(f"   원본 (seed 42, 15 epochs): {v5_original['top1']*100:.1f}%")
        print(f"   검증 평균 (5 epochs): {statistics['top1_accuracy']['mean']*100:.1f}%")
        print(f"   차이: {(v5_original['top1'] - statistics['top1_accuracy']['mean'])*100:+.1f}%p")
        
        print(f"\nTop-5:")
        print(f"   원본 (seed 42, 15 epochs): {v5_original['top5']*100:.1f}%")
        print(f"   검증 평균 (5 epochs): {statistics['top5_accuracy']['mean']*100:.1f}%")
        print(f"   차이: {(v5_original['top5'] - statistics['top5_accuracy']['mean'])*100:+.1f}%p")
        
        print(f"\n💡 분석:")
        print(f"   - 5 epoch 결과가 15 epoch보다 낮은 것은 정상")
        print(f"   - Seed 간 변동성이 작으면 재현성 확보")
        print(f"   - 추세가 일관되면 v5 성능 신뢰 가능")
        
        print(f"\n✅ 결과 저장: {results_path}")
        print(f"\n{'='*80}\n")
    
    else:
        print("\n❌ 모든 실험 실패")


if __name__ == "__main__":
    main()
