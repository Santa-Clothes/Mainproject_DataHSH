"""
PyTorch Dataset implementation for Fashion JSON Encoder training.

This module provides the FashionDataset class that integrates with PyTorch DataLoader
for efficient batch processing, padding, and data augmentation during training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
from PIL import Image

from .dataset_loader import KFashionDatasetLoader
from .data_models import FashionItem, ProcessedBatch


class FashionDataset(Dataset):
    """
    PyTorch Dataset for Fashion JSON Encoder training with LAZY LOADING.
    
    Loads JSON files on-demand in __getitem__ instead of loading all at initialization.
    """
    
    def __init__(self, 
                 dataset_loader: KFashionDatasetLoader,
                 file_paths: List[Tuple[Path, str]],
                 image_transforms: Optional[transforms.Compose] = None,
                 max_sequence_lengths: Optional[Dict[str, int]] = None):
        """
        Initialize the Fashion Dataset with lazy loading.
        
        Args:
            dataset_loader: KFashionDatasetLoader instance (for parsing and processing)
            file_paths: List of (json_path, category) tuples
            image_transforms: Optional torchvision transforms for image augmentation
            max_sequence_lengths: Maximum lengths for multi-categorical fields
        """
        self.dataset_loader = dataset_loader
        self.file_paths = file_paths
        
        if not file_paths:
            raise ValueError("No file paths provided. Call load_dataset_by_category() first.")
        
        if not dataset_loader._vocabularies_built:
            raise ValueError("Vocabularies not built. Call build_vocabularies() first.")
        
        # Set up image transforms
        if image_transforms is None:
            self.image_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.image_transforms = image_transforms
        
        # Set default max sequence lengths
        if max_sequence_lengths is None:
            self.max_sequence_lengths = {
                'style': 10,
                'material': 10,
                'detail': 15
            }
        else:
            self.max_sequence_lengths = max_sequence_lengths
    
    def __len__(self) -> int:
        """Return the number of file paths in the dataset."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, List[int]]]:
        """
        Get a single item from the dataset (LAZY LOADING).
        Parses JSON file on-demand.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing processed image and JSON metadata
        """
        # Get file path and category
        json_path, category = self.file_paths[idx]
        
        # Parse JSON file NOW (lazy loading)
        item = self.dataset_loader.parse_json_file(json_path, category)
        
        # If parsing failed, return a dummy item (will be filtered by collate_fn)
        if item is None:
            # Return dummy data - collate_fn should handle this
            return {
                'image': torch.zeros(3, 224, 224),
                'category': 0,
                'style': [0],
                'silhouette': 0,
                'material': [0],
                'detail': [0],
                'valid': False
            }
        
        try:
            # Load and preprocess image
            image = self.dataset_loader.get_cropped_image(item)
            image_tensor = self.image_transforms(image)
            
            # Process JSON metadata
            processed_json = self.dataset_loader.get_processed_json(item)
            
            return {
                'image': image_tensor,
                'category': processed_json['category'],
                'style': processed_json['style'],
                'silhouette': processed_json['silhouette'],
                'material': processed_json['material'],
                'detail': processed_json['detail'],
                'valid': True
            }
        except Exception as e:
            # If image loading fails, return dummy
            return {
                'image': torch.zeros(3, 224, 224),
                'category': 0,
                'style': [0],
                'silhouette': 0,
                'material': [0],
                'detail': [0],
                'valid': False
            }
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for model initialization."""
        return self.dataset_loader.get_vocab_sizes()
    
    def get_max_lengths(self) -> Dict[str, int]:
        """Get maximum sequence lengths for multi-categorical fields."""
        return self.max_sequence_lengths.copy()


def collate_fashion_batch(batch: List[Dict]) -> ProcessedBatch:
    """
    Custom collate function for Fashion Dataset batching.
    Filters out invalid items and handles padding.
    
    Args:
        batch: List of individual dataset items
        
    Returns:
        ProcessedBatch object with properly padded and masked tensors
    """
    # Filter out invalid items
    valid_batch = [item for item in batch if item.get('valid', True)]
    batch_size = len(valid_batch)
    
    # Handle empty batch
    if batch_size == 0:
        return ProcessedBatch(
            images=torch.empty(0, 3, 224, 224),
            category_ids=torch.empty(0, dtype=torch.long),
            style_ids=torch.empty(0, 1, dtype=torch.long),
            silhouette_ids=torch.empty(0, dtype=torch.long),
            material_ids=torch.empty(0, 1, dtype=torch.long),
            detail_ids=torch.empty(0, 1, dtype=torch.long),
            style_mask=torch.empty(0, 1, dtype=torch.float),
            material_mask=torch.empty(0, 1, dtype=torch.float),
            detail_mask=torch.empty(0, 1, dtype=torch.float)
        )
    
    # Stack images
    images = torch.stack([item['image'] for item in valid_batch])
    
    # Process single categorical fields
    category_ids = torch.tensor([item['category'] for item in valid_batch], dtype=torch.long)
    silhouette_ids = torch.tensor([item['silhouette'] for item in valid_batch], dtype=torch.long)
    
    # Process multi-categorical fields with padding
    style_data = _pad_sequences([item['style'] for item in valid_batch])
    material_data = _pad_sequences([item['material'] for item in valid_batch])
    detail_data = _pad_sequences([item['detail'] for item in valid_batch])
    
    return ProcessedBatch(
        images=images,
        category_ids=category_ids,
        style_ids=style_data['ids'],
        silhouette_ids=silhouette_ids,
        material_ids=material_data['ids'],
        detail_ids=detail_data['ids'],
        style_mask=style_data['mask'],
        material_mask=material_data['mask'],
        detail_mask=detail_data['mask']
    )


def _pad_sequences(sequences: List[List[int]]) -> Dict[str, torch.Tensor]:
    """
    Pad sequences to the same length and create attention masks.
    
    Args:
        sequences: List of variable-length sequences
        
    Returns:
        Dictionary containing:
            - 'ids': [batch_size, max_len] - padded sequences
            - 'mask': [batch_size, max_len] - attention mask (1 for valid, 0 for padding)
    """
    if not sequences:
        return {
            'ids': torch.zeros((0, 1), dtype=torch.long),
            'mask': torch.zeros((0, 1), dtype=torch.float)
        }
    
    batch_size = len(sequences)
    
    # Handle empty sequences - replace with [0] but mark as masked
    processed_sequences = []
    original_lengths = []
    
    for seq in sequences:
        if seq:  # Non-empty sequence
            processed_sequences.append(seq)
            original_lengths.append(len(seq))
        else:  # Empty sequence
            processed_sequences.append([0])  # Placeholder
            original_lengths.append(0)  # Remember it was originally empty
    
    max_len = max(len(seq) for seq in processed_sequences)
    max_len = max(max_len, 1)  # At least length 1
    
    # Create padded tensor
    padded_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float)
    
    for i, (seq, orig_len) in enumerate(zip(processed_sequences, original_lengths)):
        seq_len = len(seq)
        if seq_len > 0:
            padded_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            # Only mark as valid if the original sequence was non-empty
            if orig_len > 0:
                mask[i, :orig_len] = 1.0
    
    return {
        'ids': padded_ids,
        'mask': mask
    }


def create_fashion_dataloader(dataset: FashionDataset,
                            batch_size: int = 32,
                            shuffle: bool = True,
                            num_workers: int = 0,
                            pin_memory: bool = True,
                            drop_last: bool = None) -> DataLoader:
    """
    Create a DataLoader for the Fashion Dataset.
    
    Args:
        dataset: FashionDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch (default: True for training, False for validation)
        
    Returns:
        PyTorch DataLoader configured for fashion data
    """
    # Auto-determine drop_last based on shuffle if not specified
    if drop_last is None:
        drop_last = shuffle  # True for training (shuffle=True), False for validation (shuffle=False)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fashion_batch,
        drop_last=drop_last
    )


def create_augmented_transforms(image_size: int = 224,
                              augment_prob: float = 0.5) -> transforms.Compose:
    """
    Create image transforms with data augmentation for training.
    
    Args:
        image_size: Target image size (square)
        augment_prob: Probability of applying augmentation
        
    Returns:
        Composed transforms for training data augmentation
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=augment_prob),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        ),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_validation_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Create image transforms for validation (no augmentation).
    
    Args:
        image_size: Target image size (square)
        
    Returns:
        Composed transforms for validation data
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])


class FashionDataModule:
    """
    Data module that encapsulates dataset creation and DataLoader management.
    
    Provides a high-level interface for setting up training and validation
    data loaders with proper transforms and configurations.
    """
    
    def __init__(self,
                 dataset_path: str,
                 target_categories: List[str] = None,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 train_split: float = 0.8,
                 image_size: int = 224,
                 augment_prob: float = 0.5):
        """
        Initialize the Fashion Data Module.
        
        Args:
            dataset_path: Path to K-Fashion dataset
            target_categories: List of target categories to process
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            train_split: Fraction of data to use for training
            image_size: Target image size
            augment_prob: Probability of data augmentation
        """
        self.dataset_path = dataset_path
        self.target_categories = target_categories or ['레트로', '로맨틱', '리조트']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.image_size = image_size
        self.augment_prob = augment_prob
        
        # Will be initialized in setup()
        self.dataset_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self._train_dataloader = None
        self._val_dataloader = None
    
    def setup(self) -> None:
        """
        Set up datasets and data loaders with LAZY LOADING.
        
        Scans dataset, builds vocabularies, and creates train/val splits.
        """
        # Initialize dataset loader
        self.dataset_loader = KFashionDatasetLoader(
            dataset_path=self.dataset_path,
            target_categories=self.target_categories,
            image_size=(self.image_size, self.image_size)
        )
        
        # Scan dataset - FAST (only collects paths, no JSON parsing)
        print("\n=== PHASE 1: Scanning dataset (fast) ===")
        file_paths = self.dataset_loader.load_dataset_by_category()
        
        # Build vocabularies from sample
        print("\n=== PHASE 2: Building vocabularies ===")
        vocabularies = self.dataset_loader.build_vocabularies()
        
        print(f"\nTotal files: {len(file_paths)}")
        print(f"Vocabulary sizes: {self.dataset_loader.get_vocab_sizes()}")
        
        # Create train/val split with random shuffle
        import random
        random.seed(42)
        random.shuffle(file_paths)
        
        train_size = int(len(file_paths) * self.train_split)
        train_paths = file_paths[:train_size]
        val_paths = file_paths[train_size:]
        
        # Create separate dataset loaders for train/val
        train_loader = KFashionDatasetLoader(
            dataset_path=self.dataset_path,
            target_categories=self.target_categories,
            image_size=(self.image_size, self.image_size)
        )
        train_loader.processor.vocabularies = vocabularies
        train_loader._vocabularies_built = True
        
        val_loader = KFashionDatasetLoader(
            dataset_path=self.dataset_path,
            target_categories=self.target_categories,
            image_size=(self.image_size, self.image_size)
        )
        val_loader.processor.vocabularies = vocabularies
        val_loader._vocabularies_built = True
        
        # Create datasets with lazy loading
        train_transforms = create_augmented_transforms(
            image_size=self.image_size,
            augment_prob=self.augment_prob
        )
        val_transforms = create_validation_transforms(image_size=self.image_size)
        
        print("\n=== PHASE 3: Creating datasets (lazy loading) ===")
        self.train_dataset = FashionDataset(
            dataset_loader=train_loader,
            file_paths=train_paths,
            image_transforms=train_transforms
        )
        
        self.val_dataset = FashionDataset(
            dataset_loader=val_loader,
            file_paths=val_paths,
            image_transforms=val_transforms
        )
        
        print(f"Train dataset: {len(self.train_dataset)} files")
        print(f"Validation dataset: {len(self.val_dataset)} files")
        print("\n✓ Setup complete - JSON files will be loaded on-demand during training")
    
    def train_dataloader(self, use_class_balanced: bool = False) -> DataLoader:
        """Get training data loader."""
        if self.train_dataset is None:
            raise ValueError("Call setup() first")
        
        if self._train_dataloader is None or use_class_balanced:
            if use_class_balanced:
                # Use class-balanced sampler
                from .class_balanced_sampler import create_balanced_dataloader
                self._train_dataloader = create_balanced_dataloader(
                    dataset=self.train_dataset,
                    batch_size=self.batch_size,
                    oversample_minority=True,
                    min_samples_per_class=2,
                    num_workers=self.num_workers
                )
            else:
                # Use regular dataloader
                self._train_dataloader = create_fashion_dataloader(
                    dataset=self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True
                )
        
        return self._train_dataloader
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        if self.val_dataset is None:
            raise ValueError("Call setup() first")
        
        if self._val_dataloader is None:
            self._val_dataloader = create_fashion_dataloader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        return self._val_dataloader
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for model initialization."""
        if self.dataset_loader is None:
            raise ValueError("Call setup() first")
        return self.dataset_loader.get_vocab_sizes()
    
    def get_sample_batch(self) -> ProcessedBatch:
        """Get a sample batch for testing/debugging."""
        if self.train_dataset is None:
            raise ValueError("Call setup() first")
        
        # Get a small sample
        sample_loader = create_fashion_dataloader(
            dataset=self.train_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        
        return next(iter(sample_loader))