"""
Unit tests for data processing functionality.

Tests the core data processing components including polygon to bbox conversion,
image cropping, vocabulary building, and dataset loading.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np

from src.data.processor import FashionDataProcessor
from src.data.dataset_loader import KFashionDatasetLoader
from src.data.data_models import FashionItem


class TestFashionDataProcessor:
    """Test cases for FashionDataProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FashionDataProcessor(
            dataset_path="/tmp/test_dataset",
            target_categories=['상의', '하의', '아우터']
        )
    
    def test_polygon_to_bbox_basic(self):
        """Test basic polygon to bbox conversion."""
        # Simple rectangle polygon
        polygon = [(10, 20), (50, 20), (50, 80), (10, 80)]
        bbox = self.processor.polygon_to_bbox(polygon)
        
        expected = (10, 20, 40, 60)  # (x, y, width, height)
        assert bbox == expected
    
    def test_polygon_to_bbox_irregular(self):
        """Test irregular polygon to bbox conversion."""
        # Irregular polygon
        polygon = [(5, 10), (25, 5), (45, 15), (40, 35), (15, 30)]
        bbox = self.processor.polygon_to_bbox(polygon)
        
        expected = (5, 5, 40, 30)  # (min_x, min_y, width, height)
        assert bbox == expected
    
    def test_polygon_to_bbox_empty(self):
        """Test polygon to bbox with empty polygon."""
        with pytest.raises(ValueError, match="Empty polygon provided"):
            self.processor.polygon_to_bbox([])
    
    def test_crop_image_by_bbox_basic(self):
        """Test basic image cropping functionality."""
        # Create test image
        image = Image.new('RGB', (100, 100), color='red')
        bbox = (20, 30, 40, 30)  # (x, y, width, height)
        
        cropped = self.processor.crop_image_by_bbox(image, bbox, padding=0.0)
        
        # Should crop to exact bbox size when no padding
        assert cropped.size == (40, 30)
    
    def test_crop_image_by_bbox_with_padding(self):
        """Test image cropping with padding."""
        # Create test image
        image = Image.new('RGB', (100, 100), color='red')
        bbox = (20, 30, 20, 20)  # (x, y, width, height)
        
        cropped = self.processor.crop_image_by_bbox(image, bbox, padding=0.1)
        
        # With 10% padding, should be larger than original bbox
        assert cropped.size[0] > 20
        assert cropped.size[1] > 20
    
    def test_crop_image_edge_cases(self):
        """Test image cropping edge cases."""
        # Create test image
        image = Image.new('RGB', (50, 50), color='red')
        
        # Bbox at image edge
        bbox = (40, 40, 20, 20)  # Extends beyond image
        cropped = self.processor.crop_image_by_bbox(image, bbox, padding=0.0)
        
        # Should be clipped to image boundaries
        assert cropped.size == (10, 10)
    
    def test_build_vocabulary_basic(self):
        """Test vocabulary building from JSON files."""
        # Create temporary JSON files
        with tempfile.TemporaryDirectory() as temp_dir:
            json_data = [
                {
                    'category': '상의',
                    'style': ['캐주얼', '스포츠'],
                    'silhouette': '슬림',
                    'material': ['면', '폴리에스터'],
                    'detail': ['프린트']
                },
                {
                    'category': '하의',
                    'style': ['정장'],
                    'silhouette': '와이드',
                    'material': ['울'],
                    'detail': ['무지', '스트라이프']
                }
            ]
            
            json_files = []
            for i, data in enumerate(json_data):
                json_file = Path(temp_dir) / f"test_{i}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                json_files.append(str(json_file))
            
            vocabularies = self.processor.build_vocabulary(json_files)
            
            # Check vocabulary structure
            assert '<UNK>' in vocabularies['category']
            assert vocabularies['category']['<UNK>'] == 0
            assert '상의' in vocabularies['category']
            assert '하의' in vocabularies['category']
            
            # Check multi-categorical fields
            assert '캐주얼' in vocabularies['style']
            assert '스포츠' in vocabularies['style']
            assert '정장' in vocabularies['style']
    
    def test_process_json_fields_basic(self):
        """Test JSON field processing to vocabulary indices."""
        # Build vocabulary first
        self.processor.vocabularies = {
            'category': {'<UNK>': 0, '상의': 1, '하의': 2},
            'style': {'<UNK>': 0, '캐주얼': 1, '스포츠': 2},
            'silhouette': {'<UNK>': 0, '슬림': 1, '와이드': 2},
            'material': {'<UNK>': 0, '면': 1, '울': 2},
            'detail': {'<UNK>': 0, '프린트': 1, '무지': 2}
        }
        
        json_data = {
            'category': '상의',
            'style': ['캐주얼', '스포츠'],
            'silhouette': '슬림',
            'material': ['면'],
            'detail': ['프린트', '무지']
        }
        
        processed = self.processor.process_json_fields(json_data)
        
        assert processed['category'] == 1  # '상의'
        assert processed['style'] == [1, 2]  # ['캐주얼', '스포츠']
        assert processed['silhouette'] == 1  # '슬림'
        assert processed['material'] == [1]  # ['면']
        assert processed['detail'] == [1, 2]  # ['프린트', '무지']
    
    def test_process_json_fields_unknown_tokens(self):
        """Test JSON field processing with unknown tokens."""
        # Build vocabulary first
        self.processor.vocabularies = {
            'category': {'<UNK>': 0, '상의': 1},
            'style': {'<UNK>': 0, '캐주얼': 1},
            'silhouette': {'<UNK>': 0, '슬림': 1},
            'material': {'<UNK>': 0, '면': 1},
            'detail': {'<UNK>': 0, '프린트': 1}
        }
        
        json_data = {
            'category': '알 수 없는 카테고리',  # Unknown
            'style': ['알 수 없는 스타일'],  # Unknown
            'silhouette': '알 수 없는 실루엣',  # Unknown
            'material': ['알 수 없는 소재'],  # Unknown
            'detail': ['알 수 없는 디테일']  # Unknown
        }
        
        processed = self.processor.process_json_fields(json_data)
        
        # All should map to <UNK> (index 0)
        assert processed['category'] == 0
        assert processed['style'] == [0]
        assert processed['silhouette'] == 0
        assert processed['material'] == [0]
        assert processed['detail'] == [0]
    
    def test_get_vocab_sizes(self):
        """Test vocabulary size retrieval."""
        self.processor.vocabularies = {
            'category': {'<UNK>': 0, '상의': 1, '하의': 2},
            'style': {'<UNK>': 0, '캐주얼': 1, '스포츠': 2, '정장': 3},
            'silhouette': {'<UNK>': 0, '슬림': 1},
            'material': {'<UNK>': 0, '면': 1, '울': 2, '폴리에스터': 3, '나일론': 4},
            'detail': {'<UNK>': 0, '프린트': 1, '무지': 2}
        }
        
        sizes = self.processor.get_vocab_sizes()
        
        assert sizes['category'] == 3
        assert sizes['style'] == 4
        assert sizes['silhouette'] == 2
        assert sizes['material'] == 5
        assert sizes['detail'] == 3


class TestKFashionDatasetLoader:
    """Test cases for KFashionDatasetLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = KFashionDatasetLoader(
            dataset_path=self.temp_dir,
            target_categories=['상의', '하의', '아우터']
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_image(self, path: Path, size: tuple = (100, 100)):
        """Create a test image file."""
        image = Image.new('RGB', size, color='red')
        image.save(path)
        return image
    
    def _create_test_annotation(self, image_name: str, category: str = '상의'):
        """Create a test annotation dictionary."""
        return {
            'image_name': image_name,
            'category': category,
            'style': ['캐주얼', '스포츠'],
            'silhouette': '슬림',
            'material': ['면', '폴리에스터'],
            'detail': ['프린트'],
            'polygon': [10, 20, 50, 20, 50, 80, 10, 80]  # Rectangle
        }
    
    def test_create_fashion_item_basic(self):
        """Test basic FashionItem creation."""
        # Create test image
        image_dir = Path(self.temp_dir) / "images"
        image_dir.mkdir()
        image_path = image_dir / "test.jpg"
        self._create_test_image(image_path)
        
        # Create annotation
        annotation = self._create_test_annotation("test.jpg")
        
        item = self.loader._create_fashion_item(annotation, image_dir)
        
        assert item is not None
        assert item.category == '상의'
        assert item.style == ['캐주얼', '스포츠']
        assert item.silhouette == '슬림'
        assert item.material == ['면', '폴리에스터']
        assert item.detail == ['프린트']
        assert item.bbox == (10, 20, 40, 60)  # Converted from polygon
    
    def test_create_fashion_item_missing_image(self):
        """Test FashionItem creation with missing image."""
        image_dir = Path(self.temp_dir) / "images"
        image_dir.mkdir()
        
        annotation = self._create_test_annotation("nonexistent.jpg")
        
        item = self.loader._create_fashion_item(annotation, image_dir)
        
        assert item is None
    
    def test_create_fashion_item_invalid_polygon(self):
        """Test FashionItem creation with invalid polygon."""
        # Create test image
        image_dir = Path(self.temp_dir) / "images"
        image_dir.mkdir()
        image_path = image_dir / "test.jpg"
        self._create_test_image(image_path)
        
        # Create annotation with invalid polygon
        annotation = {
            'image_name': 'test.jpg',
            'category': '상의',
            'polygon': []  # Empty polygon
        }
        
        item = self.loader._create_fashion_item(annotation, image_dir)
        
        assert item is None
    
    def test_has_valid_bbox(self):
        """Test bounding box validation."""
        # Valid bbox
        item = FashionItem(
            image_path="test.jpg",
            bbox=(10, 20, 30, 40),
            category="상의",
            style=[],
            silhouette="",
            material=[],
            detail=[]
        )
        assert self.loader._has_valid_bbox(item) is True
        
        # Invalid bbox - zero width
        item.bbox = (10, 20, 0, 40)
        assert self.loader._has_valid_bbox(item) is False
        
        # Invalid bbox - negative coordinates
        item.bbox = (-10, 20, 30, 40)
        assert self.loader._has_valid_bbox(item) is False
    
    def test_load_dataset_basic(self):
        """Test basic dataset loading functionality."""
        # Create directory structure
        json_dir = Path(self.temp_dir) / "annotations"
        image_dir = Path(self.temp_dir) / "images"
        json_dir.mkdir()
        image_dir.mkdir()
        
        # Create test images
        for i in range(3):
            image_path = image_dir / f"test_{i}.jpg"
            self._create_test_image(image_path)
        
        # Create test annotations
        annotations = [
            self._create_test_annotation(f"test_{i}.jpg", category)
            for i, category in enumerate(['상의', '하의', '아우터'])
        ]
        
        for i, annotation in enumerate(annotations):
            json_path = json_dir / f"test_{i}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False)
        
        # Load dataset
        items = self.loader.load_dataset()
        
        assert len(items) == 3
        assert all(isinstance(item, FashionItem) for item in items)
        assert all(item.category in ['상의', '하의', '아우터'] for item in items)
    
    def test_build_vocabularies(self):
        """Test vocabulary building from loaded items."""
        # Create test items
        items = [
            FashionItem(
                image_path="test1.jpg",
                bbox=(10, 20, 30, 40),
                category="상의",
                style=['캐주얼', '스포츠'],
                silhouette="슬림",
                material=['면'],
                detail=['프린트']
            ),
            FashionItem(
                image_path="test2.jpg",
                bbox=(15, 25, 35, 45),
                category="하의",
                style=['정장'],
                silhouette="와이드",
                material=['울', '폴리에스터'],
                detail=['무지', '스트라이프']
            )
        ]
        
        self.loader._fashion_items = items
        vocabularies = self.loader.build_vocabularies()
        
        # Check vocabulary structure
        assert '<UNK>' in vocabularies['category']
        assert '상의' in vocabularies['category']
        assert '하의' in vocabularies['category']
        
        # Check multi-categorical fields
        assert '캐주얼' in vocabularies['style']
        assert '스포츠' in vocabularies['style']
        assert '정장' in vocabularies['style']
        
        assert '면' in vocabularies['material']
        assert '울' in vocabularies['material']
        assert '폴리에스터' in vocabularies['material']
    
    def test_get_cropped_image(self):
        """Test image cropping functionality."""
        # Create test image
        image_dir = Path(self.temp_dir) / "images"
        image_dir.mkdir()
        image_path = image_dir / "test.jpg"
        self._create_test_image(image_path, size=(200, 200))
        
        # Create test item
        item = FashionItem(
            image_path=str(image_path),
            bbox=(50, 60, 80, 70),
            category="상의",
            style=[],
            silhouette="",
            material=[],
            detail=[]
        )
        
        cropped = self.loader.get_cropped_image(item)
        
        # Should be resized to target size (224, 224)
        assert cropped.size == (224, 224)
        assert cropped.mode == 'RGB'
    
    def test_get_processed_json(self):
        """Test JSON processing functionality."""
        # Set up vocabularies
        self.loader.processor.vocabularies = {
            'category': {'<UNK>': 0, '상의': 1, '하의': 2},
            'style': {'<UNK>': 0, '캐주얼': 1, '스포츠': 2},
            'silhouette': {'<UNK>': 0, '슬림': 1, '와이드': 2},
            'material': {'<UNK>': 0, '면': 1, '울': 2},
            'detail': {'<UNK>': 0, '프린트': 1, '무지': 2}
        }
        self.loader._vocabularies_built = True
        
        # Create test item
        item = FashionItem(
            image_path="test.jpg",
            bbox=(10, 20, 30, 40),
            category="상의",
            style=['캐주얼', '스포츠'],
            silhouette="슬림",
            material=['면'],
            detail=['프린트']
        )
        
        processed = self.loader.get_processed_json(item)
        
        assert processed['category'] == 1  # '상의'
        assert processed['style'] == [1, 2]  # ['캐주얼', '스포츠']
        assert processed['silhouette'] == 1  # '슬림'
        assert processed['material'] == [1]  # ['면']
        assert processed['detail'] == [1]  # ['프린트']


if __name__ == "__main__":
    pytest.main([__file__])