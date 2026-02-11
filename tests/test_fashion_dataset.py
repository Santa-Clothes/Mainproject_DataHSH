"""
Unit tests for Fashion Dataset and DataLoader functionality.

Tests the PyTorch Dataset implementation, batch processing, padding,
data augmentation, and DataLoader integration.
"""

import pytest
import tempfile
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np

from src.data.fashion_dataset import (
    FashionDataset,
    collate_fashion_batch,
    create_fashion_dataloader,
    create_augmented_transforms,
    create_validation_transforms,
    FashionDataModule,
    _pad_sequences
)
from src.data.dataset_loader import KFashionDatasetLoader
from src.data.data_models import FashionItem, ProcessedBatch


class TestFashionDataset:
    """Test cases for FashionDataset class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_loader = self._create_test_dataset_loader()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_dataset_loader(self) -> KFashionDatasetLoader:
        """Create a test dataset loader with sample data."""
        # Create directory structure
        json_dir = Path(self.temp_dir) / "annotations"
        image_dir = Path(self.temp_dir) / "images"
        json_dir.mkdir()
        image_dir.mkdir()
        
        # Create test images
        test_items = []
        for i in range(5):
            # Create test image
            image_path = image_dir / f"test_{i}.jpg"
            image = Image.new('RGB', (200, 200), color='red')
            image.save(image_path)
            
            # Create test annotation
            annotation = {
                'image_name': f"test_{i}.jpg",
                'category': ['상의', '하의', '아우터'][i % 3],
                'style': [['캐주얼'], ['정장', '스포츠'], ['아웃도어']][i % 3],
                'silhouette': ['슬림', '와이드', '오버사이즈'][i % 3],
                'material': [['면'], ['울', '폴리에스터'], ['나일론']][i % 3],
                'detail': [['프린트'], ['무지', '스트라이프'], ['지퍼', '후드']][i % 3],
                'polygon': [20, 30, 80, 30, 80, 90, 20, 90]  # Rectangle
            }
            
            json_path = json_dir / f"test_{i}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False)
        
        # Create and setup dataset loader
        loader = KFashionDatasetLoader(
            dataset_path=self.temp_dir,
            target_categories=['상의', '하의', '아우터']
        )
        
        # Load dataset and build vocabularies
        items = loader.load_dataset()
        vocabularies = loader.build_vocabularies()
        
        return loader
    
    def test_initialization_success(self):
        """Test successful FashionDataset initialization."""
        dataset = FashionDataset(dataset_loader=self.dataset_loader)
        
        assert len(dataset) > 0
        assert dataset.image_transforms is not None
        assert dataset.max_sequence_lengths is not None
        assert 'style' in dataset.max_sequence_lengths
        assert 'material' in dataset.max_sequence_lengths
        assert 'detail' in dataset.max_sequence_lengths
    
    def test_initialization_no_items(self):
        """Test FashionDataset initialization with no items."""
        empty_loader = KFashionDatasetLoader(dataset_path=self.temp_dir)
        empty_loader._fashion_items = []
        
        with pytest.raises(ValueError, match="no fashion items"):
            FashionDataset(dataset_loader=empty_loader)
    
    def test_initialization_no_vocabularies(self):
        """Test FashionDataset initialization without vocabularies."""
        loader = KFashionDatasetLoader(dataset_path=self.temp_dir)
        loader._fashion_items = [FashionItem(
            image_path="test.jpg",
            bbox=(10, 20, 30, 40),
            category="상의",
            style=[],
            silhouette="",
            material=[],
            detail=[]
        )]
        loader._vocabularies_built = False
        
        with pytest.raises(ValueError, match="Vocabularies not built"):
            FashionDataset(dataset_loader=loader)
    
    def test_getitem_basic(self):
        """Test basic __getitem__ functionality."""
        dataset = FashionDataset(dataset_loader=self.dataset_loader)
        
        item = dataset[0]
        
        # Check return structure
        assert 'image' in item
        assert 'category' in item
        assert 'style' in item
        assert 'silhouette' in item
        assert 'material' in item
        assert 'detail' in item
        
        # Check tensor types and shapes
        assert isinstance(item['image'], torch.Tensor)
        assert item['image'].shape == (3, 224, 224)  # Default image size
        assert isinstance(item['category'], int)
        assert isinstance(item['style'], list)
        assert isinstance(item['silhouette'], int)
        assert isinstance(item['material'], list)
        assert isinstance(item['detail'], list)
    
    def test_getitem_different_indices(self):
        """Test __getitem__ with different indices."""
        dataset = FashionDataset(dataset_loader=self.dataset_loader)
        
        # Test multiple indices
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            assert item['image'].shape == (3, 224, 224)
            assert isinstance(item['category'], int)
    
    def test_custom_transforms(self):
        """Test FashionDataset with custom transforms."""
        custom_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        dataset = FashionDataset(
            dataset_loader=self.dataset_loader,
            image_transforms=custom_transforms
        )
        
        item = dataset[0]
        assert item['image'].shape == (3, 128, 128)  # Custom size
    
    def test_custom_max_lengths(self):
        """Test FashionDataset with custom max sequence lengths."""
        custom_lengths = {
            'style': 5,
            'material': 3,
            'detail': 7
        }
        
        dataset = FashionDataset(
            dataset_loader=self.dataset_loader,
            max_sequence_lengths=custom_lengths
        )
        
        assert dataset.max_sequence_lengths == custom_lengths
    
    def test_compute_max_lengths(self):
        """Test automatic computation of max sequence lengths."""
        dataset = FashionDataset(dataset_loader=self.dataset_loader)
        
        max_lengths = dataset.max_sequence_lengths
        
        # Should have all required fields
        assert 'style' in max_lengths
        assert 'material' in max_lengths
        assert 'detail' in max_lengths
        
        # Should be positive integers
        for length in max_lengths.values():
            assert isinstance(length, int)
            assert length > 0
    
    def test_get_vocab_sizes(self):
        """Test vocabulary size retrieval."""
        dataset = FashionDataset(dataset_loader=self.dataset_loader)
        
        vocab_sizes = dataset.get_vocab_sizes()
        
        # Should have all required fields
        required_fields = ['category', 'style', 'silhouette', 'material', 'detail']
        for field in required_fields:
            assert field in vocab_sizes
            assert isinstance(vocab_sizes[field], int)
            assert vocab_sizes[field] > 0
    
    def test_get_max_lengths(self):
        """Test max lengths retrieval."""
        dataset = FashionDataset(dataset_loader=self.dataset_loader)
        
        max_lengths = dataset.get_max_lengths()
        
        # Should return a copy
        assert max_lengths == dataset.max_sequence_lengths
        assert max_lengths is not dataset.max_sequence_lengths


class TestCollateFashionBatch:
    """Test cases for collate_fashion_batch function."""
    
    def _create_sample_batch_items(self, batch_size: int = 3) -> list:
        """Create sample batch items for testing."""
        items = []
        for i in range(batch_size):
            item = {
                'image': torch.randn(3, 224, 224),
                'category': i % 3,
                'style': [1, 2] if i % 2 == 0 else [3],
                'silhouette': i % 2,
                'material': [1, 2, 3] if i == 0 else [4, 5],
                'detail': [1] if i % 3 == 0 else [2, 3, 4, 5]
            }
            items.append(item)
        return items
    
    def test_collate_basic(self):
        """Test basic batch collation functionality."""
        batch_items = self._create_sample_batch_items(batch_size=3)
        
        batch = collate_fashion_batch(batch_items)
        
        # Check return type
        assert isinstance(batch, ProcessedBatch)
        
        # Check tensor shapes
        assert batch.images.shape == (3, 3, 224, 224)
        assert batch.category_ids.shape == (3,)
        assert batch.silhouette_ids.shape == (3,)
        
        # Check multi-categorical fields have proper shapes
        assert len(batch.style_ids.shape) == 2
        assert len(batch.material_ids.shape) == 2
        assert len(batch.detail_ids.shape) == 2
        
        # Check masks have same shapes as ids
        assert batch.style_mask.shape == batch.style_ids.shape
        assert batch.material_mask.shape == batch.material_ids.shape
        assert batch.detail_mask.shape == batch.detail_ids.shape
    
    def test_collate_padding(self):
        """Test that sequences are properly padded."""
        batch_items = self._create_sample_batch_items(batch_size=2)
        
        batch = collate_fashion_batch(batch_items)
        
        # All sequences in batch should have same length (padded)
        batch_size, max_style_len = batch.style_ids.shape
        batch_size2, max_material_len = batch.material_ids.shape
        batch_size3, max_detail_len = batch.detail_ids.shape
        
        assert batch_size == batch_size2 == batch_size3 == 2
        
        # Check that masks correctly indicate valid vs padded positions
        for i in range(batch_size):
            # Count valid positions in masks
            style_valid = batch.style_mask[i].sum().item()
            material_valid = batch.material_mask[i].sum().item()
            detail_valid = batch.detail_mask[i].sum().item()
            
            # Should match original sequence lengths
            assert style_valid == len(batch_items[i]['style'])
            assert material_valid == len(batch_items[i]['material'])
            assert detail_valid == len(batch_items[i]['detail'])
    
    def test_collate_empty_batch(self):
        """Test collation with empty batch."""
        batch = collate_fashion_batch([])
        
        assert isinstance(batch, ProcessedBatch)
        assert batch.images.shape[0] == 0
        assert batch.category_ids.shape[0] == 0
    
    def test_collate_single_item(self):
        """Test collation with single item."""
        batch_items = self._create_sample_batch_items(batch_size=1)
        
        batch = collate_fashion_batch(batch_items)
        
        assert batch.images.shape == (1, 3, 224, 224)
        assert batch.category_ids.shape == (1,)
        assert batch.silhouette_ids.shape == (1,)


class TestPadSequences:
    """Test cases for _pad_sequences function."""
    
    def test_pad_sequences_basic(self):
        """Test basic sequence padding."""
        sequences = [[1, 2, 3], [4, 5], [6]]
        
        result = _pad_sequences(sequences)
        
        assert 'ids' in result
        assert 'mask' in result
        
        # Check shapes
        assert result['ids'].shape == (3, 3)  # batch_size=3, max_len=3
        assert result['mask'].shape == (3, 3)
        
        # Check padding
        expected_ids = torch.tensor([
            [1, 2, 3],
            [4, 5, 0],  # Padded with 0
            [6, 0, 0]   # Padded with 0
        ], dtype=torch.long)
        
        expected_mask = torch.tensor([
            [1, 1, 1],
            [1, 1, 0],  # Last position masked
            [1, 0, 0]   # Last two positions masked
        ], dtype=torch.float)
        
        assert torch.equal(result['ids'], expected_ids)
        assert torch.equal(result['mask'], expected_mask)
    
    def test_pad_sequences_empty_input(self):
        """Test padding with empty input."""
        result = _pad_sequences([])
        
        assert result['ids'].shape == (0, 1)
        assert result['mask'].shape == (0, 1)
    
    def test_pad_sequences_empty_sequences(self):
        """Test padding with empty sequences."""
        sequences = [[], [1, 2], []]
        
        result = _pad_sequences(sequences)
        
        # Empty sequences should be replaced with [0]
        assert result['ids'].shape == (3, 2)  # max_len from [1, 2]
        
        expected_ids = torch.tensor([
            [0, 0],  # Empty -> [0], then padded
            [1, 2],  # Original sequence
            [0, 0]   # Empty -> [0], then padded
        ], dtype=torch.long)
        
        expected_mask = torch.tensor([
            [0, 0],  # All masked (was empty)
            [1, 1],  # All valid
            [0, 0]   # All masked (was empty)
        ], dtype=torch.float)
        
        assert torch.equal(result['ids'], expected_ids)
        assert torch.equal(result['mask'], expected_mask)
    
    def test_pad_sequences_single_sequence(self):
        """Test padding with single sequence."""
        sequences = [[1, 2, 3]]
        
        result = _pad_sequences(sequences)
        
        assert result['ids'].shape == (1, 3)
        assert result['mask'].shape == (1, 3)
        
        # No padding needed
        assert torch.equal(result['ids'], torch.tensor([[1, 2, 3]], dtype=torch.long))
        assert torch.equal(result['mask'], torch.tensor([[1, 1, 1]], dtype=torch.float))


class TestDataLoaderCreation:
    """Test cases for DataLoader creation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_minimal_dataset(self) -> FashionDataset:
        """Create a minimal dataset for testing."""
        # Create minimal test data
        json_dir = Path(self.temp_dir) / "annotations"
        image_dir = Path(self.temp_dir) / "images"
        json_dir.mkdir()
        image_dir.mkdir()
        
        # Create single test item
        image_path = image_dir / "test.jpg"
        image = Image.new('RGB', (100, 100), color='red')
        image.save(image_path)
        
        annotation = {
            'image_name': "test.jpg",
            'category': '상의',
            'style': ['캐주얼'],
            'silhouette': '슬림',
            'material': ['면'],
            'detail': ['프린트'],
            'polygon': [10, 20, 50, 60, 50, 80, 10, 80]
        }
        
        json_path = json_dir / "test.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False)
        
        # Create dataset
        loader = KFashionDatasetLoader(dataset_path=self.temp_dir)
        loader.load_dataset()
        loader.build_vocabularies()
        
        return FashionDataset(dataset_loader=loader)
    
    def test_create_fashion_dataloader_basic(self):
        """Test basic DataLoader creation."""
        dataset = self._create_minimal_dataset()
        
        dataloader = create_fashion_dataloader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 1
        assert dataloader.drop_last is True
        
        # Test iteration
        batch = next(iter(dataloader))
        assert isinstance(batch, ProcessedBatch)
    
    def test_create_augmented_transforms(self):
        """Test augmented transforms creation."""
        transforms_obj = create_augmented_transforms(
            image_size=128,
            augment_prob=0.5
        )
        
        assert isinstance(transforms_obj, transforms.Compose)
        
        # Test on sample image
        image = Image.new('RGB', (100, 100), color='red')
        tensor = transforms_obj(image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 128, 128)
    
    def test_create_validation_transforms(self):
        """Test validation transforms creation."""
        transforms_obj = create_validation_transforms(image_size=256)
        
        assert isinstance(transforms_obj, transforms.Compose)
        
        # Test on sample image
        image = Image.new('RGB', (100, 100), color='red')
        tensor = transforms_obj(image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 256, 256)


class TestFashionDataModule:
    """Test cases for FashionDataModule class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self._create_test_data()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self):
        """Create test data for data module."""
        json_dir = Path(self.temp_dir) / "annotations"
        image_dir = Path(self.temp_dir) / "images"
        json_dir.mkdir()
        image_dir.mkdir()
        
        # Create multiple test items for train/val split
        for i in range(10):
            # Create test image
            image_path = image_dir / f"test_{i}.jpg"
            image = Image.new('RGB', (100, 100), color='red')
            image.save(image_path)
            
            # Create test annotation
            annotation = {
                'image_name': f"test_{i}.jpg",
                'category': ['상의', '하의', '아우터'][i % 3],
                'style': ['캐주얼'],
                'silhouette': '슬림',
                'material': ['면'],
                'detail': ['프린트'],
                'polygon': [10, 20, 50, 60, 50, 80, 10, 80]
            }
            
            json_path = json_dir / f"test_{i}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False)
    
    def test_data_module_initialization(self):
        """Test FashionDataModule initialization."""
        data_module = FashionDataModule(
            dataset_path=self.temp_dir,
            batch_size=4,
            train_split=0.8
        )
        
        assert data_module.dataset_path == self.temp_dir
        assert data_module.batch_size == 4
        assert data_module.train_split == 0.8
        assert data_module.dataset_loader is None  # Not setup yet
    
    def test_data_module_setup(self):
        """Test FashionDataModule setup process."""
        data_module = FashionDataModule(
            dataset_path=self.temp_dir,
            batch_size=2,
            train_split=0.7
        )
        
        data_module.setup()
        
        # Check that components are initialized
        assert data_module.dataset_loader is not None
        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None
        
        # Check train/val split
        total_items = len(data_module.dataset_loader._fashion_items)
        expected_train_size = int(total_items * 0.7)
        
        assert len(data_module.train_dataset) == expected_train_size
        assert len(data_module.val_dataset) == total_items - expected_train_size
    
    def test_data_module_dataloaders(self):
        """Test DataLoader creation in data module."""
        data_module = FashionDataModule(
            dataset_path=self.temp_dir,
            batch_size=2
        )
        
        data_module.setup()
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 2
        
        # Test val dataloader
        val_loader = data_module.val_dataloader()
        assert isinstance(val_loader, DataLoader)
        assert val_loader.batch_size == 2
        
        # Test iteration
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        assert isinstance(train_batch, ProcessedBatch)
        assert isinstance(val_batch, ProcessedBatch)
    
    def test_data_module_vocab_sizes(self):
        """Test vocabulary size retrieval from data module."""
        data_module = FashionDataModule(dataset_path=self.temp_dir)
        data_module.setup()
        
        vocab_sizes = data_module.get_vocab_sizes()
        
        required_fields = ['category', 'style', 'silhouette', 'material', 'detail']
        for field in required_fields:
            assert field in vocab_sizes
            assert isinstance(vocab_sizes[field], int)
            assert vocab_sizes[field] > 0
    
    def test_data_module_sample_batch(self):
        """Test sample batch retrieval from data module."""
        data_module = FashionDataModule(dataset_path=self.temp_dir)
        data_module.setup()
        
        sample_batch = data_module.get_sample_batch()
        
        assert isinstance(sample_batch, ProcessedBatch)
        assert sample_batch.images.shape[0] <= 4  # Sample batch size
    
    def test_data_module_error_handling(self):
        """Test error handling in data module."""
        data_module = FashionDataModule(dataset_path=self.temp_dir)
        
        # Should raise error before setup
        with pytest.raises(ValueError, match="Call setup\\(\\) first"):
            data_module.train_dataloader()
        
        with pytest.raises(ValueError, match="Call setup\\(\\) first"):
            data_module.val_dataloader()
        
        with pytest.raises(ValueError, match="Call setup\\(\\) first"):
            data_module.get_vocab_sizes()
        
        with pytest.raises(ValueError, match="Call setup\\(\\) first"):
            data_module.get_sample_batch()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])