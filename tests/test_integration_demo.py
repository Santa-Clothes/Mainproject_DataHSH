"""
Integration test demonstrating the complete K-Fashion dataset loading pipeline.

This test shows how all components work together to process K-Fashion data
from raw JSON annotations to processed fashion items ready for training.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from PIL import Image

from src.data.dataset_loader import KFashionDatasetLoader
from src.data.data_models import FashionItem


class TestKFashionIntegration:
    """Integration test for the complete K-Fashion data processing pipeline."""
    
    def test_complete_pipeline_demo(self):
        """
        Demonstrate the complete pipeline from raw K-Fashion data to processed items.
        
        This test shows:
        1. Loading K-Fashion dataset from JSON annotations and images
        2. Filtering by target categories (상의, 하의, 아우터)
        3. Converting polygons to bounding boxes
        4. Building vocabularies from metadata
        5. Processing JSON fields to vocabulary indices
        6. Cropping and resizing images
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock K-Fashion dataset structure
            dataset_path = Path(temp_dir)
            json_dir = dataset_path / "annotations"
            image_dir = dataset_path / "images"
            json_dir.mkdir()
            image_dir.mkdir()
            
            # Create sample fashion items with Korean metadata
            sample_items = [
                {
                    "image_name": "top_001.jpg",
                    "category": "상의",
                    "style": ["캐주얼", "스포츠"],
                    "silhouette": "슬림",
                    "material": ["면", "폴리에스터"],
                    "detail": ["프린트", "로고"],
                    "polygon": [20, 30, 80, 30, 80, 120, 20, 120]  # Rectangle
                },
                {
                    "image_name": "bottom_001.jpg", 
                    "category": "하의",
                    "style": ["정장"],
                    "silhouette": "와이드",
                    "material": ["울"],
                    "detail": ["무지"],
                    "polygon": [15, 25, 75, 25, 75, 100, 15, 100]  # Rectangle
                },
                {
                    "image_name": "outer_001.jpg",
                    "category": "아우터", 
                    "style": ["캐주얼", "빈티지"],
                    "silhouette": "오버사이즈",
                    "material": ["데님", "면"],
                    "detail": ["워싱", "스티치"],
                    "polygon": [10, 20, 90, 20, 90, 130, 10, 130]  # Rectangle
                },
                {
                    "image_name": "shoes_001.jpg",
                    "category": "신발",  # This should be filtered out
                    "style": ["스니커즈"],
                    "silhouette": "로우탑",
                    "material": ["가죽"],
                    "detail": ["레이스업"],
                    "polygon": [25, 35, 65, 35, 65, 85, 25, 85]  # Rectangle
                }
            ]
            
            # Create test images and annotations
            for i, item in enumerate(sample_items):
                # Create test image (200x200 RGB)
                image_path = image_dir / item["image_name"]
                image = Image.new('RGB', (200, 200), color=['red', 'green', 'blue', 'yellow'][i])
                image.save(image_path)
                
                # Create JSON annotation
                json_path = json_dir / f"annotation_{i:03d}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
            
            # Initialize dataset loader
            loader = KFashionDatasetLoader(
                dataset_path=str(dataset_path),
                target_categories=['상의', '하의', '아우터'],
                image_size=(224, 224),
                crop_padding=0.1
            )
            
            # Step 1: Load dataset (should filter out 신발 category)
            fashion_items = loader.load_dataset()
            
            assert len(fashion_items) == 3  # 신발 should be filtered out
            assert all(isinstance(item, FashionItem) for item in fashion_items)
            assert all(item.category in ['상의', '하의', '아우터'] for item in fashion_items)
            
            # Verify polygon to bbox conversion
            for item in fashion_items:
                x, y, width, height = item.bbox
                assert width > 0 and height > 0
                assert x >= 0 and y >= 0
            
            # Step 2: Build vocabularies from loaded items
            vocabularies = loader.build_vocabularies()
            
            # Verify vocabulary structure
            expected_fields = ['category', 'style', 'silhouette', 'material', 'detail']
            for field in expected_fields:
                assert field in vocabularies
                assert '<UNK>' in vocabularies[field]
                assert vocabularies[field]['<UNK>'] == 0
            
            # Verify Korean terms are in vocabularies
            assert '상의' in vocabularies['category']
            assert '하의' in vocabularies['category']
            assert '아우터' in vocabularies['category']
            assert '캐주얼' in vocabularies['style']
            assert '스포츠' in vocabularies['style']
            
            # Step 3: Process individual items
            for item in fashion_items:
                # Get processed JSON with vocabulary indices
                processed_json = loader.get_processed_json(item)
                
                # Verify structure
                assert 'category' in processed_json
                assert 'style' in processed_json
                assert 'silhouette' in processed_json
                assert 'material' in processed_json
                assert 'detail' in processed_json
                
                # Verify single categorical fields are integers
                assert isinstance(processed_json['category'], int)
                assert isinstance(processed_json['silhouette'], int)
                
                # Verify multi-categorical fields are lists
                assert isinstance(processed_json['style'], list)
                assert isinstance(processed_json['material'], list)
                assert isinstance(processed_json['detail'], list)
                
                # Get cropped and resized image
                cropped_image = loader.get_cropped_image(item)
                
                # Verify image processing
                assert cropped_image.size == (224, 224)
                assert cropped_image.mode == 'RGB'
            
            # Step 4: Verify vocabulary sizes
            vocab_sizes = loader.get_vocab_sizes()
            
            # Should have reasonable vocabulary sizes
            assert vocab_sizes['category'] >= 4  # <UNK> + 3 categories
            assert vocab_sizes['style'] >= 5     # <UNK> + various styles
            assert vocab_sizes['material'] >= 4  # <UNK> + various materials
            
            print("✅ Complete K-Fashion pipeline test passed!")
            print(f"   - Loaded {len(fashion_items)} fashion items")
            print(f"   - Built vocabularies with sizes: {vocab_sizes}")
            print(f"   - Successfully processed images and JSON metadata")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])