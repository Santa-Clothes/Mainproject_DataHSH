#!/usr/bin/env python3
"""
K-Fashion 10% 샘플 학습 스크립트
빠른 학습을 위해 데이터의 10%만 사용
"""

import random
import torch
from pathlib import Path

# 시드 고정
random.seed(42)
torch.manual_seed(42)

from data.fashion_dataset import FashionDataModule
from training.trainer import FashionTrainer, create_trainer_from_data_module
from utils.config import TrainingConfig

def main():
    print("="*60)
    print("K-Fashion 10% 샘플 학습")
    print("="*60)
    
    # 설정
    config = TrainingConfig()
    config.batch_size = 64  # 배치 사이즈 증가
    config.max_epochs = 5   # Stage 2 epoch 수
    
    dataset_path = "D:/K-Fashion/Training/라벨링데이터"
    
    # 23개 카테고리 자동 감지
    dataset_path_obj = Path(dataset_path)
    target_categories = [d.name for d in dataset_path_obj.iterdir() if d.is_dir()]
    
    print(f"\n1️⃣ 데이터 모듈 설정")
    print(f"   - 카테고리: {len(target_categories)}개")
    print(f"   - 배치 사이즈: {config.batch_size}")
    
    # 데이터 모듈 생성
    data_module = FashionDataModule(
        dataset_path=dataset_path,
        target_categories=target_categories,
        batch_size=config.batch_size,
        num_workers=0,
        train_split=0.8,
        image_size=224
    )
    
    # 데이터 로드
    data_module.setup()
    
    # 2️⃣ 10% 샘플링
    print(f"\n2️⃣ 데이터 샘플링 (10%)")
    original_train_size = len(data_module.train_dataset.file_paths)
    original_val_size = len(data_module.val_dataset.file_paths)
    
    sample_ratio = 0.1
    train_sample_size = int(original_train_size * sample_ratio)
    val_sample_size = int(original_val_size * sample_ratio)
    
    # 랜덤 샘플링
    train_sampled_paths = random.sample(
        data_module.train_dataset.file_paths, 
        train_sample_size
    )
    val_sampled_paths = random.sample(
        data_module.val_dataset.file_paths, 
        val_sample_size
    )
    
    # 샘플링된 경로로 교체
    data_module.train_dataset.file_paths = train_sampled_paths
    data_module.val_dataset.file_paths = val_sampled_paths
    
    print(f"   - Train: {original_train_size} → {train_sample_size}")
    print(f"   - Val: {original_val_size} → {val_sample_size}")
    
    # 3️⃣ 트레이너 설정
    print(f"\n3️⃣ 트레이너 설정")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   - Device: {device}")
    
    trainer = create_trainer_from_data_module(
        data_module=data_module,
        config=config,
        device=device,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )
    
    # 4️⃣ 학습 시작
    print(f"\n4️⃣ 학습 시작")
    print(f"   - Stage 1 (JSON Encoder): 2 epochs")
    print(f"   - Stage 2 (Contrastive): 5 epochs")
    
    # 데이터 로더
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"\n   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Stage 1: JSON Encoder Standalone
    print(f"\n{'='*60}")
    print("Stage 1: JSON Encoder Standalone Training")
    print(f"{'='*60}")
    
    stage1_results = trainer.train_json_encoder_standalone(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2
    )
    
    # Stage 2: Contrastive Learning
    print(f"\n{'='*60}")
    print("Stage 2: Contrastive Learning Training")
    print(f"{'='*60}")
    
    stage2_results = trainer.train_contrastive_learning(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5
    )
    
    # 결과 출력
    print(f"\n{'='*60}")
    print("학습 완료!")
    print(f"{'='*60}")
    print(f"Stage 1 최종 Loss: {stage1_results['val_losses'][-1]:.4f}")
    print(f"Stage 2 최고 Loss: {stage2_results['best_val_loss']:.4f}")
    print(f"체크포인트 저장 위치: checkpoints/")
    
    # 정리
    trainer.close()

if __name__ == "__main__":
    main()
