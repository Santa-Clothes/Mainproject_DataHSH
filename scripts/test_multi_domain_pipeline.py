"""
Multi-Domain Pipeline 테스트

MultiDomainFashionDataset과 학습 파이프라인이 제대로 작동하는지 테스트합니다.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

# Direct import to avoid __init__.py issues with Python 3.13
import importlib.util
spec = importlib.util.spec_from_file_location(
    "multi_domain_dataset",
    project_root / "src" / "data" / "multi_domain_dataset.py"
)
multi_domain_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_domain_module)

MultiDomainFashionDataset = multi_domain_module.MultiDomainFashionDataset
create_multi_domain_dataloader = multi_domain_module.create_multi_domain_dataloader


def test_dataset_loading():
    """데이터셋 로딩 테스트"""
    print("\n" + "="*80)
    print("Test 1: Dataset Loading")
    print("="*80)

    try:
        dataset = MultiDomainFashionDataset(
            csv_path='data/multi_domain/sample_dataset.csv',
            domain_augment=False,  # No augmentation for testing
            max_items=10,  # Small subset for quick test
        )

        print(f"\n[OK] Dataset loaded successfully")
        print(f"  Total items: {len(dataset)}")

        return dataset

    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_dataloader():
    """DataLoader 테스트"""
    print("\n" + "="*80)
    print("Test 2: DataLoader")
    print("="*80)

    try:
        dataset = MultiDomainFashionDataset(
            csv_path='data/multi_domain/sample_dataset.csv',
            domain_augment=False,
            max_items=20,
        )

        dataloader = create_multi_domain_dataloader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Single process for testing
        )

        print(f"\n[OK] DataLoader created successfully")
        print(f"  Batches: {len(dataloader)}")

        # Get one batch
        print("\nFetching first batch...")
        batch = next(iter(dataloader))

        print(f"\n[OK] Batch loaded successfully")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Num categories: {len(batch['categories'])}")
        print(f"  Num domains: {len(batch['domains'])}")
        print(f"  Domain types: {set(batch['domains'])}")
        print(f"  Categories: {set(batch['categories'])}")

        return True

    except Exception as e:
        print(f"\n[ERROR] DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Loss 계산 테스트"""
    print("\n" + "="*80)
    print("Test 3: Loss Computation")
    print("="*80)

    try:
        from models.multi_view_contrastive_loss import MultiViewContrastiveLoss
        import torch.nn.functional as F

        # Dummy embeddings
        batch_size = 8
        embed_dim = 512

        image_embeddings = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=-1)
        json_embeddings = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=-1)

        domains = ['flat_product'] * 4 + ['model_wearing'] * 4
        categories = ['상의', '하의', '상의', '하의'] * 2

        # Loss
        loss_fn = MultiViewContrastiveLoss(
            temperature=0.07,
            cross_domain_weight=0.3,
        )

        losses = loss_fn(image_embeddings, json_embeddings, domains, categories)

        print(f"\n[OK] Loss computed successfully")
        print(f"  Total Loss: {losses['total_loss'].item():.4f}")
        print(f"  Image-JSON Loss: {losses['image_json_loss'].item():.4f}")
        print(f"  Cross-Domain Loss: {losses['cross_domain_loss'].item():.4f}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """전체 파이프라인 테스트 (DataLoader + Loss)"""
    print("\n" + "="*80)
    print("Test 4: Full Pipeline (DataLoader + Loss)")
    print("="*80)

    try:
        # Dataset
        dataset = MultiDomainFashionDataset(
            csv_path='data/multi_domain/sample_dataset.csv',
            domain_augment=False,
            max_items=20,
        )

        dataloader = create_multi_domain_dataloader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )

        # Get batch
        batch = next(iter(dataloader))

        # Simulate CLIP embeddings (normally from model)
        from models.multi_view_contrastive_loss import MultiViewContrastiveLoss
        import torch.nn.functional as F

        batch_size = batch['images'].shape[0]
        embed_dim = 512

        # Dummy embeddings (normally from CLIP encoder)
        image_embeddings = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=-1)
        json_embeddings = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=-1)

        # Loss
        loss_fn = MultiViewContrastiveLoss(
            temperature=0.07,
            cross_domain_weight=0.3,
        )

        losses = loss_fn(
            image_embeddings,
            json_embeddings,
            domains=batch['domains'],
            categories=batch['categories'],
        )

        print(f"\n[OK] Full pipeline works!")
        print(f"  Batch size: {batch_size}")
        print(f"  Total Loss: {losses['total_loss'].item():.4f}")
        print(f"  Cross-Domain Loss: {losses['cross_domain_loss'].item():.4f}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """모든 테스트 실행"""
    print("\n" + "="*80)
    print("Multi-Domain Pipeline Tests")
    print("="*80)

    results = {}

    # Test 1: Dataset loading
    dataset = test_dataset_loading()
    results['dataset_loading'] = dataset is not None

    # Test 2: DataLoader
    results['dataloader'] = test_dataloader()

    # Test 3: Loss computation
    results['loss_computation'] = test_loss_computation()

    # Test 4: Full pipeline
    results['full_pipeline'] = test_full_pipeline()

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        print("\nNext steps:")
        print("1. Download DeepFashion2 dataset")
        print("2. Run: python scripts/data_collection/parse_deepfashion2.py --create_multi_domain")
        print("3. Run: python scripts/training/train_multi_domain.py")
    else:
        print("\n[FAILURE] Some tests failed. Please fix the issues above.")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
