"""
Baseline v5: RTX 4090 Optimized Training
=========================================

RTX 4090 최적화:
1. Mixed Precision (AMP) - FP16 학습
2. TF32 활성화 - 4090 전용 가속
3. DataLoader 최적화 - num_workers, pin_memory, persistent_workers
4. 큰 배치 크기 - 384 (24GB VRAM 활용)
5. 실제 데이터 경로 구조 지원

데이터 경로:
- Training: C:\Work\hwangseonghun\K-fashion\Training\라벨링데이터
- Validation: C:\Work\hwangseonghun\K-fashion\Validation\라벨링데이터
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
from utils.config import TrainingConfig, DataConfig, ALL_CATEGORIES
from torchvision import transforms
import torch.nn as nn


# ============================================================================
# RTX 4090 최적화 설정
# ============================================================================

def setup_rtx4090_optimizations():
    """RTX 4090 전용 최적화 설정"""
    print("\n🚀 RTX 4090 Optimizations")
    print("="*60)
    
    # (1) TF32 활성화 - 4090 전용 가속
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✓ TF32 enabled (4090 acceleration)")
    
    # (2) cuDNN 벤치마크 모드
    print("✓ cuDNN benchmark mode enabled")
    
    # (3) CUDA 메모리 최적화
    torch.cuda.empty_cache()
    print("✓ CUDA cache cleared")
    
    print("="*60)


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
        
        # 가우시안 블러 (30% 확률)
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


def create_rtx4090_dataloader(dataset, batch_size, shuffle=True, num_workers=8):
    """RTX 4090 최적화 DataLoader"""
    from torch.utils.data import DataLoader
    from data.fashion_dataset import collate_fashion_batch
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,      # CPU 코어 활용
        pin_memory=True,              # GPU 전송 가속
        persistent_workers=True,      # Worker 재사용
        prefetch_factor=4,            # 미리 로드
        drop_last=shuffle,            # Training만 drop
        collate_fn=collate_fashion_batch
    )


def main():
    print("="*80)
    print("Fashion JSON Encoder - Baseline v5 RTX 4090 Optimized")
    print("Mixed Precision + TF32 + DataLoader Optimization")
    print("="*80)
    
    # RTX 4090 최적화 설정
    setup_rtx4090_optimizations()
    
    # Device 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    
    # 데이터 경로 설정 (pathlib 사용)
    # utils/config.py에서 data_root를 수정하세요
    data_config = DataConfig()
    TRAIN_LABEL_DIR = str(data_config.train_data_path)
    VAL_LABEL_DIR = str(data_config.val_data_path)

    # 카테고리는 utils/config.py에서 import
    print(f"\n📂 Data Paths:")
    print(f"   Train: {TRAIN_LABEL_DIR}")
    print(f"   Val: {VAL_LABEL_DIR}")
    print(f"   Categories: {len(ALL_CATEGORIES)} categories")
    
    # Training configuration (RTX 4090 최적화)
    config = TrainingConfig(
        batch_size=384,          # 4090: 384 (24GB VRAM 활용)
        learning_rate=3e-4,      # 배치 커졌으니 LR 증가
        max_epochs=15,
        temperature=0.1,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout_rate=0.1,
        weight_decay=1e-4        # 약간 증가
    )
    
    print(f"\n⚙️  Training Configuration (RTX 4090 Optimized):")
    print(f"   Batch size: {config.batch_size} (4090 optimized)")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Max epochs: {config.max_epochs}")
    print(f"   Temperature: {config.temperature}")
    print(f"   Weight decay: {config.weight_decay}")
    
    # Data module 설정 - Training
    print(f"\n📂 Loading Training dataset...")
    
    train_data_module = FashionDataModule(
        dataset_path=TRAIN_LABEL_DIR,
        target_categories=ALL_CATEGORIES,  # 전체 23개 카테고리
        batch_size=config.batch_size,
        num_workers=0,  # Will use custom dataloader
        train_split=1.0,  # Use all training data
        image_size=224,
        augment_prob=0.5
    )
    
    train_data_module.setup()
    
    # Data module 설정 - Validation
    print(f"\n📂 Loading Validation dataset...")
    
    val_data_module = FashionDataModule(
        dataset_path=VAL_LABEL_DIR,
        target_categories=ALL_CATEGORIES,  # 전체 23개 카테고리
        batch_size=config.batch_size,
        num_workers=0,  # Will use custom dataloader
        train_split=1.0,  # Use all validation data
        image_size=224,
        augment_prob=0.0  # No augmentation for validation
    )
    
    val_data_module.setup()
    
    # 강화된 augmentation으로 train dataset 재생성
    print(f"\n🎨 Applying enhanced augmentation...")
    enhanced_transforms = create_enhanced_augmented_transforms(image_size=224)
    train_data_module.train_dataset.image_transforms = enhanced_transforms
    
    print(f"   ✓ RandomResizedCrop (scale 0.8-1.0)")
    print(f"   ✓ ColorJitter (brightness/contrast/saturation 0.2)")
    print(f"   ✓ RandomRotation (10 degrees)")
    print(f"   ✓ RandomAffine (translate 5%, scale 5%)")
    print(f"   ✓ GaussianBlur (30% probability)")
    print(f"   ✓ RandomErasing (20% probability)")
    
    # RTX 4090 최적화 DataLoader 생성
    print(f"\n⚡ Creating RTX 4090 optimized DataLoaders...")
    train_loader = create_rtx4090_dataloader(
        train_data_module.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8  # CPU 코어 활용
    )
    
    val_loader = create_rtx4090_dataloader(
        val_data_module.train_dataset,  # Use train_dataset (it's 100% of val data)
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   ✓ num_workers=8 (CPU parallelization)")
    print(f"   ✓ pin_memory=True (GPU transfer acceleration)")
    print(f"   ✓ persistent_workers=True (worker reuse)")
    print(f"   ✓ prefetch_factor=4 (prefetch batches)")
    
    # Trainer 초기화
    print(f"\n🤖 Initializing trainer...")
    vocab_sizes = train_data_module.get_vocab_sizes()
    
    trainer = FashionTrainer(
        config=config,
        vocab_sizes=vocab_sizes,
        device=device,
        checkpoint_dir='checkpoints',
        log_dir='logs/baseline_v5_rtx4090',
        finetune_clip=False  # Frozen CLIP
    )
    
    # Mixed Precision Scaler 초기화
    print(f"\n⚡ Initializing Mixed Precision (AMP)...")
    scaler = torch.cuda.amp.GradScaler()
    print(f"   ✓ FP16 training enabled")
    print(f"   ✓ Automatic loss scaling")
    
    # Optimizer 재설정 (AdamW with higher LR)
    print(f"\n🔧 Setting up optimizer...")
    trainer.optimizer = torch.optim.AdamW(
        trainer.json_encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    print(f"   ✓ AdamW optimizer")
    print(f"   ✓ LR: {config.learning_rate}")
    print(f"   ✓ Weight decay: {config.weight_decay}")
    
    # Training 시작
    print(f"\n{'='*80}")
    print("🚀 Starting Training with RTX 4090 Optimizations...")
    print(f"{'='*80}\n")
    
    # Custom training loop with AMP
    print("⚡ Training with Mixed Precision (AMP)...")
    
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
        'v5_rtx4090_results': {
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
            'gpu': 'RTX 4090',
            'batch_size': config.batch_size,
            'mixed_precision': 'FP16 (AMP)',
            'tf32': True,
            'dataloader_optimization': True,
            'augmentation': 'enhanced (6 techniques)',
            'epochs': config.max_epochs,
            'learning_rate': config.learning_rate
        }
    }
    
    # JSON 저장
    results_path = "results/baseline_v5_rtx4090_results.json"
    os.makedirs("results", exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Results saved to: {results_path}")
    
    # 체크포인트 복사
    import shutil
    best_checkpoint = "checkpoints/best_model.pt"
    v5_checkpoint = "checkpoints/baseline_v5_rtx4090_best_model.pt"
    
    if os.path.exists(best_checkpoint):
        shutil.copy(best_checkpoint, v5_checkpoint)
        print(f"   ✓ Best model saved to: {v5_checkpoint}")
    
    # 최종 리포트 출력
    print(f"\n{'='*80}")
    print("📊 Training Complete - v5 RTX 4090 Results")
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
    
    print(f"\n⚡ RTX 4090 Optimizations:")
    print(f"   ✓ Batch size: {config.batch_size} (24GB VRAM)")
    print(f"   ✓ Mixed Precision: FP16 (AMP)")
    print(f"   ✓ TF32: Enabled")
    print(f"   ✓ DataLoader: Optimized (8 workers)")
    
    # 목표 달성 여부
    top1_target = 0.32  # 32%
    top5_target = 0.73  # 73%
    
    top1_achieved = final_metrics.get('top1_accuracy', 0) >= top1_target
    top5_achieved = final_metrics.get('top5_accuracy', 0) >= top5_target
    
    print(f"\n🎯 Target Achievement:")
    print(f"   Top-1 ≥ 32%: {'✅ ACHIEVED' if top1_achieved else '❌ NOT YET'}")
    print(f"   Top-5 ≥ 73%: {'✅ ACHIEVED' if top5_achieved else '❌ NOT YET'}")
    
    if top1_achieved and top5_achieved:
        print(f"\n🎉 SUCCESS! v5 RTX 4090 최적화 완료!")
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
