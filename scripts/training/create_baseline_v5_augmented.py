"""
Baseline v5: Data Augmentation + Class Balancing
=================================================

개선 사항:
1. 강화된 이미지 증강 (RandomCrop, RandomResizedCrop, GaussianNoise)
2. 클래스 밸런싱 샘플러 (레트로 카테고리 보강)
3. v3 체크포인트에서 시작 (transfer learning)

목표:
- Top-1: 32-33% (v3: 30.3% 대비 +2-3%)
- Top-5: 73-74% (v3: 71.5% 대비 +2-3%)
- 레트로 카테고리 성능 개선
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.fashion_dataset import FashionDataModule, create_augmented_transforms, create_validation_transforms
from training.trainer import FashionTrainer
from utils.config import TrainingConfig
from torchvision import transforms
import torch.nn as nn


class GaussianNoise(nn.Module):
    """가우시안 노이즈 추가 (augmentation)"""
    
    def __init__(self, mean=0.0, std=0.01):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, tensor):
        if self.training:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor


def create_enhanced_augmented_transforms(image_size: int = 224) -> transforms.Compose:
    """
    v5용 강화된 이미지 증강
    
    추가된 augmentation:
    - RandomResizedCrop: 스케일 변화 학습
    - RandomAffine: 약간의 변형
    - GaussianBlur: 블러 처리
    - RandomErasing: 일부 영역 마스킹
    """
    return transforms.Compose([
        # 랜덤 크롭 및 리사이즈 (스케일 변화)
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),  # 80-100% 크기
            ratio=(0.9, 1.1)   # 약간의 비율 변화
        ),
        
        # 기본 augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 색상 변화 (약간 강화)
        transforms.ColorJitter(
            brightness=0.2,  # 0.1 -> 0.2
            contrast=0.2,    # 0.1 -> 0.2
            saturation=0.2,  # 0.1 -> 0.2
            hue=0.1          # 0.05 -> 0.1
        ),
        
        # 회전 (약간 증가)
        transforms.RandomRotation(degrees=10),  # 5 -> 10
        
        # 약간의 변형
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),  # 5% 이동
            scale=(0.95, 1.05)       # 5% 스케일 변화
        ),
        
        # 가우시안 블러 (50% 확률)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        
        # 텐서 변환
        transforms.ToTensor(),
        
        # ImageNet 정규화
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # 랜덤 이레이징 (일부 영역 마스킹, 20% 확률)
        transforms.RandomErasing(
            p=0.2,
            scale=(0.02, 0.1),  # 2-10% 영역
            ratio=(0.3, 3.3)
        )
    ])


def main():
    print("="*80)
    print("Fashion JSON Encoder - Baseline v5 Training")
    print("Data Augmentation + Class Balancing")
    print("="*80)
    
    # Device 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 데이터 경로
    dataset_path = "C:/sample/라벨링데이터"
    
    # Training configuration (v3와 동일, epoch만 조정)
    config = TrainingConfig(
        batch_size=16,           # 기존 유지
        learning_rate=1e-4,
        max_epochs=15,           # 2-5 epoch 실험 후 조정
        temperature=0.1,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout_rate=0.1,
        weight_decay=1e-5
    )
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Max epochs: {config.max_epochs}")
    print(f"   Temperature: {config.temperature}")
    
    # Data module 설정 (강화된 augmentation)
    print(f"\n📂 Loading dataset from: {dataset_path}")
    
    data_module = FashionDataModule(
        dataset_path=dataset_path,
        target_categories=['레트로', '로맨틱', '리조트'],
        batch_size=config.batch_size,
        num_workers=0,
        train_split=0.8,
        image_size=224,
        augment_prob=0.5  # 기본값 유지
    )
    
    data_module.setup()
    
    # 강화된 augmentation으로 train dataset 재생성
    print(f"\n🎨 Applying enhanced augmentation...")
    enhanced_transforms = create_enhanced_augmented_transforms(image_size=224)
    data_module.train_dataset.image_transforms = enhanced_transforms
    
    print(f"   ✓ RandomResizedCrop (scale 0.8-1.0)")
    print(f"   ✓ ColorJitter (brightness/contrast/saturation 0.2)")
    print(f"   ✓ RandomRotation (10 degrees)")
    print(f"   ✓ RandomAffine (translate 5%, scale 5%)")
    print(f"   ✓ GaussianBlur (30% probability)")
    print(f"   ✓ RandomErasing (20% probability)")
    
    # 클래스 밸런싱 DataLoader 생성
    print(f"\n⚖️  Creating class-balanced DataLoader...")
    train_loader = data_module.train_dataloader(use_class_balanced=True)
    val_loader = data_module.val_dataloader()
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Trainer 초기화
    print(f"\n🤖 Initializing trainer...")
    vocab_sizes = data_module.get_vocab_sizes()
    
    trainer = FashionTrainer(
        config=config,
        vocab_sizes=vocab_sizes,
        device=device,
        checkpoint_dir='checkpoints',
        log_dir='logs/baseline_v5',
        finetune_clip=False  # v3처럼 frozen 유지
    )
    
    # v3 체크포인트 로드 (transfer learning)
    v3_checkpoint_path = "checkpoints/baseline_v3_best_model.pt"
    if os.path.exists(v3_checkpoint_path):
        print(f"\n📥 Loading v3 checkpoint for transfer learning...")
        print(f"   Path: {v3_checkpoint_path}")
        try:
            trainer.load_checkpoint(v3_checkpoint_path)
            print(f"   ✓ v3 checkpoint loaded successfully")
            print(f"   Starting from epoch {trainer.current_epoch}")
        except Exception as e:
            print(f"   ⚠️  Could not load v3 checkpoint: {e}")
            print(f"   Starting from scratch...")
    else:
        print(f"\n⚠️  v3 checkpoint not found: {v3_checkpoint_path}")
        print(f"   Starting from scratch...")
    
    # Training 시작
    print(f"\n{'='*80}")
    print("🚀 Starting Training...")
    print(f"{'='*80}\n")
    
    results = trainer.train_contrastive_learning(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.max_epochs
    )
    
    # 결과 저장
    print(f"\n💾 Saving results...")
    
    # 최종 메트릭
    final_metrics = results['final_metrics']
    
    # v3와 비교
    v3_metrics = {
        'top1_accuracy': 0.303,
        'top5_accuracy': 0.715,
        'mean_reciprocal_rank': 0.484
    }
    
    comparison = {
        'v5_results': {
            'top1_accuracy': final_metrics.get('top1_accuracy', 0),
            'top5_accuracy': final_metrics.get('top5_accuracy', 0),
            'mean_reciprocal_rank': final_metrics.get('mean_reciprocal_rank', 0),
            'best_val_loss': results['best_val_loss'],
            'total_epochs': results['total_epochs']
        },
        'v3_baseline': v3_metrics,
        'improvements': {
            'top1_delta': final_metrics.get('top1_accuracy', 0) - v3_metrics['top1_accuracy'],
            'top5_delta': final_metrics.get('top5_accuracy', 0) - v3_metrics['top5_accuracy'],
            'mrr_delta': final_metrics.get('mean_reciprocal_rank', 0) - v3_metrics['mean_reciprocal_rank']
        },
        'configuration': {
            'augmentation': 'enhanced (RandomResizedCrop, GaussianBlur, RandomErasing)',
            'class_balancing': True,
            'batch_size': config.batch_size,
            'epochs': config.max_epochs,
            'transfer_learning': 'v3 checkpoint' if os.path.exists(v3_checkpoint_path) else 'from scratch'
        }
    }
    
    # JSON 저장
    results_path = "results/baseline_v5_augmented_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Results saved to: {results_path}")
    
    # 체크포인트 복사
    import shutil
    best_checkpoint = "checkpoints/best_model.pt"
    v5_checkpoint = "checkpoints/baseline_v5_best_model.pt"
    
    if os.path.exists(best_checkpoint):
        shutil.copy(best_checkpoint, v5_checkpoint)
        print(f"   ✓ Best model saved to: {v5_checkpoint}")
    
    # 최종 리포트 출력
    print(f"\n{'='*80}")
    print("📊 Training Complete - v5 Results")
    print(f"{'='*80}")
    
    print(f"\n🎯 Final Metrics:")
    print(f"   Top-1 Accuracy: {final_metrics.get('top1_accuracy', 0)*100:.1f}%")
    print(f"   Top-5 Accuracy: {final_metrics.get('top5_accuracy', 0)*100:.1f}%")
    print(f"   MRR: {final_metrics.get('mean_reciprocal_rank', 0):.3f}")
    print(f"   Best Val Loss: {results['best_val_loss']:.3f}")
    
    print(f"\n📈 Improvement vs v3:")
    print(f"   Top-1: {comparison['improvements']['top1_delta']*100:+.1f}%")
    print(f"   Top-5: {comparison['improvements']['top5_delta']*100:+.1f}%")
    print(f"   MRR: {comparison['improvements']['mrr_delta']:+.3f}")
    
    # 목표 달성 여부
    top1_target = 0.32  # 32%
    top5_target = 0.73  # 73%
    
    top1_achieved = final_metrics.get('top1_accuracy', 0) >= top1_target
    top5_achieved = final_metrics.get('top5_accuracy', 0) >= top5_target
    
    print(f"\n🎯 Target Achievement:")
    print(f"   Top-1 ≥ 32%: {'✅ ACHIEVED' if top1_achieved else '❌ NOT YET'}")
    print(f"   Top-5 ≥ 73%: {'✅ ACHIEVED' if top5_achieved else '❌ NOT YET'}")
    
    if top1_achieved and top5_achieved:
        print(f"\n🎉 SUCCESS! v5 달성!")
    else:
        print(f"\n💡 추가 개선 필요:")
        if not top1_achieved:
            print(f"   - Top-1: {(top1_target - final_metrics.get('top1_accuracy', 0))*100:.1f}% 더 필요")
        if not top5_achieved:
            print(f"   - Top-5: {(top5_target - final_metrics.get('top5_accuracy', 0))*100:.1f}% 더 필요")
    
    print(f"\n{'='*80}\n")
    
    # Cleanup
    trainer.close()


if __name__ == "__main__":
    main()
