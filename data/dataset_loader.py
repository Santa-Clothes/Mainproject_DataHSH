"""
K-Fashion Dataset Loader for Fashion JSON Encoder.

This module provides the KFashionDatasetLoader class that integrates all preprocessing
components to load and process K-Fashion dataset for training the JSON encoder.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
import PIL.Image
from PIL import Image
import torch

from .processor import FashionDataProcessor
from .data_models import FashionItem


class KFashionDatasetLoader:
    """
    K-Fashion dataset loader that handles JSON metadata parsing, filtering,
    polygon to bbox conversion, and image cropping.
    
    This class integrates all preprocessing components to provide a complete
    data loading pipeline for the Fashion JSON Encoder system.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 target_categories: List[str] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 crop_padding: float = 0.1):
        """
        Initialize the K-Fashion dataset loader.
        
        Args:
            dataset_path: Path to the K-Fashion dataset root directory
            target_categories: List of target categories to process (default: ['레트로', '로맨틱', '리조트'])
            image_size: Target image size for resizing cropped images
            crop_padding: Padding ratio to add around bounding boxes
        """
        self.dataset_path = Path(dataset_path)
        self.target_categories = target_categories or ['레트로', '로맨틱', '리조트']
        self.image_size = image_size
        self.crop_padding = crop_padding
        
        # Initialize data processor
        self.processor = FashionDataProcessor(
            dataset_path=str(self.dataset_path),
            target_categories=self.target_categories
        )
        
        # Cache for loaded data
        self._file_paths: List[Tuple[Path, str]] = []  # (json_path, category) tuples
        self._fashion_items: List[FashionItem] = []
        self._vocabularies_built = False
        
    def load_dataset(self, 
                    json_dir: str = "annotations",
                    image_dir: str = "images") -> List[FashionItem]:
        """
        Load the complete K-Fashion dataset (legacy format).
        
        Args:
            json_dir: Subdirectory containing JSON annotation files
            image_dir: Subdirectory containing image files
            
        Returns:
            List of FashionItem objects with valid bounding boxes
        """
        json_path = self.dataset_path / json_dir
        image_path = self.dataset_path / image_dir
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_path}")
        
        fashion_items = []
        
        # Process all JSON files
        json_files = list(json_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                items = self._process_json_file(json_file, image_path)
                fashion_items.extend(items)
            except Exception as e:
                print(f"Warning: Failed to process {json_file}: {e}")
                continue
        
        # Filter items with valid bounding boxes
        valid_items = [item for item in fashion_items if self._has_valid_bbox(item)]
        
        self._fashion_items = valid_items
        return valid_items
    
    def load_dataset_by_category(self) -> List[Tuple[Path, str]]:
        """
        Scan K-Fashion dataset and collect JSON file paths (LAZY LOADING).
        Does NOT parse JSON files - only collects paths for lazy loading.
        
        Expected structure: D:/K-Fashion/Training/라벨링데이터/{카테고리}/{파일번호}.json
        
        Returns:
            List of (json_path, category) tuples
            
        Raises:
            FileNotFoundError: If dataset path or category folders don't exist
            ValueError: If no valid data is found
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        file_paths = []
        category_stats = {}
        
        print(f"Scanning dataset path: {self.dataset_path}")
        print(f"Target categories: {self.target_categories}")
        
        # Scan each target category - ONLY collect paths, NO parsing
        for category in self.target_categories:
            category_path = self.dataset_path / category
            
            if not category_path.exists():
                print(f"Warning: Category folder not found: {category_path}")
                continue
            
            print(f"Scanning category: {category}")
            
            # Find all JSON files in category folder - FAST
            json_files = list(category_path.glob("*.json"))
            print(f"Found {len(json_files)} JSON files in {category}")
            
            # Store paths only - NO json.load() here!
            for json_file in json_files:
                file_paths.append((json_file, category))
            
            category_stats[category] = len(json_files)
        
        # Print scanning statistics
        total_files = len(file_paths)
        print(f"\n=== Dataset Scan Summary ===")
        print(f"Total JSON files found: {total_files}")
        print("Category distribution:")
        for category, count in category_stats.items():
            print(f"  {category}: {count} files")
        
        if total_files == 0:
            available_folders = [f.name for f in self.dataset_path.iterdir() if f.is_dir()]
            raise ValueError(
                f"No JSON files found in {self.dataset_path}. "
                f"Available folders: {available_folders}. "
                f"Expected categories: {self.target_categories}"
            )
        
        # Store file paths for lazy loading
        self._file_paths = file_paths
        return file_paths
    
    def parse_json_file(self, json_path: Path, category: str) -> Optional[FashionItem]:
        """
        Parse a single JSON file and create FashionItem (LAZY - called on demand).
        
        Args:
            json_path: Path to JSON file
            category: Category name
            
        Returns:
            FashionItem or None if invalid
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                annotations = data
            elif 'annotations' in data:
                annotations = data['annotations']
            else:
                annotations = [data]
            
            # Process first valid annotation
            for annotation in annotations:
                annotation['category'] = category
                item = self._create_fashion_item_from_category(
                    annotation, json_path.parent, json_path.stem
                )
                if item and self._has_valid_bbox(item):
                    return item
            
            return None
            
        except Exception as e:
            # Silently skip invalid files
            return None
    
    def _create_fashion_item_from_category(self, 
                                         annotation: Dict, 
                                         category_path: Path,
                                         json_stem: str) -> Optional[FashionItem]:
        """
        Create a FashionItem from JSON annotation in category-based structure.
        
        Args:
            annotation: Single item annotation dictionary
            category_path: Path to category folder
            json_stem: JSON filename without extension (e.g., "1016530")
            
        Returns:
            FashionItem object or None if invalid
        """
        # Extract image filename from 이미지 정보
        image_info = annotation.get('이미지 정보', {})
        image_filename = image_info.get('이미지 파일명')
        
        if not image_filename:
            return None
        
        # Try to find image file in various locations
        full_image_path = None
        
        # For K-Fashion dataset, try JSON filename with .jpg extension
        simple_filename = f"{json_stem}.jpg"
        
        # Get the base dataset path (parent of 라벨링데이터)
        base_path = category_path.parent.parent  # D:/K-Fashion/Training
        
        possible_paths = [
            # In 원천데이터 folder (most likely location)
            base_path / "원천데이터" / category_path.name / simple_filename,
            base_path / "원천데이터" / category_path.name / image_filename,
            # Original filename in category folder
            category_path / image_filename,
            # Simple filename based on JSON name
            category_path / simple_filename,
            # In parent directory
            category_path.parent / image_filename,
            category_path.parent / simple_filename,
            # In images folder
            category_path.parent / "images" / image_filename,
            category_path.parent / "images" / simple_filename,
            category_path / "images" / image_filename,
            category_path / "images" / simple_filename,
        ]
        
        for path in possible_paths:
            if path.exists():
                full_image_path = path
                break
        
        if not full_image_path:
            return None
        
        # Extract polygon coordinates from 데이터셋 정보 → 데이터셋 상세설명 → 폴리곤좌표
        dataset_info = annotation.get('데이터셋 정보', {})
        dataset_detail = dataset_info.get('데이터셋 상세설명', {})
        polygon_data_dict = dataset_detail.get('폴리곤좌표', {})
        
        if not polygon_data_dict:
            print(f"Debug: No 폴리곤좌표 found. Keys: {list(dataset_detail.keys())}")
            return None
        
        # Find the first non-empty polygon from any clothing category
        polygon_coords = None
        for clothing_type in ['상의', '하의', '아우터', '원피스']:
            items = polygon_data_dict.get(clothing_type, [])
            if items and len(items) > 0 and items[0]:  # Non-empty item
                polygon_item = items[0]
                # Extract X, Y coordinates
                coords = []
                i = 1
                while f'X좌표{i}' in polygon_item and f'Y좌표{i}' in polygon_item:
                    x = polygon_item[f'X좌표{i}']
                    y = polygon_item[f'Y좌표{i}']
                    coords.append((int(x), int(y)))
                    i += 1
                
                if coords:
                    polygon_coords = coords
                    break
        
        if not polygon_coords:
            return None
        
        # Convert polygon to bbox
        try:
            bbox = self.processor.polygon_to_bbox(polygon_coords)
        except (ValueError, IndexError, TypeError):
            return None
        
        # Extract metadata from 데이터셋 정보 → 데이터셋 상세설명 → 라벨링
        dataset_info = annotation.get('데이터셋 정보', {})
        dataset_detail = dataset_info.get('데이터셋 상세설명', {})
        labeling = dataset_detail.get('라벨링', {})
        
        # Extract category from folder name (already set)
        category = annotation.get('category', '')
        
        # Extract style from 스타일 field
        style_list = labeling.get('스타일', [])
        style = []
        for style_item in style_list:
            if isinstance(style_item, dict) and '스타일' in style_item:
                style.append(style_item['스타일'])
        
        # Extract other attributes from clothing items
        silhouette = ''
        material = []
        detail = []
        
        # Look through all clothing types for attributes
        for clothing_type in ['상의', '하의', '아우터', '원피스']:
            items = labeling.get(clothing_type, [])
            for item in items:
                if not item or not isinstance(item, dict):  # Skip empty items
                    continue
                
                # Extract silhouette (핏) - use first non-empty one found
                if not silhouette and '핏' in item and item['핏']:
                    silhouette = item['핏']
                
                # Extract materials (소재)
                if '소재' in item and isinstance(item['소재'], list):
                    material.extend([m for m in item['소재'] if m])  # Filter out empty strings
                
                # Extract details (디테일)
                if '디테일' in item and isinstance(item['디테일'], list):
                    detail.extend([d for d in item['디테일'] if d])  # Filter out empty strings
        
        # Remove duplicates while preserving order
        material = list(dict.fromkeys(material))
        detail = list(dict.fromkeys(detail))
        
        return FashionItem(
            image_path=str(full_image_path),
            bbox=bbox,
            category=category,
            style=style,
            silhouette=silhouette,
            material=material,
            detail=detail
        )

    def _process_json_file(self, 
                          json_file: Path, 
                          image_path: Path) -> List[FashionItem]:
        """
        Process a single JSON annotation file.
        
        Args:
            json_file: Path to JSON annotation file
            image_path: Path to image directory
            
        Returns:
            List of FashionItem objects from this file
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        
        # Handle different JSON structures (single item vs multiple items)
        if isinstance(data, list):
            annotations = data
        elif 'annotations' in data:
            annotations = data['annotations']
        else:
            # Single item format
            annotations = [data]
        
        for annotation in annotations:
            try:
                item = self._create_fashion_item(annotation, image_path)
                if item and item.category in self.target_categories:
                    items.append(item)
            except Exception as e:
                print(f"Warning: Failed to process annotation: {e}")
                continue
        
        return items
    
    def _create_fashion_item(self, 
                           annotation: Dict, 
                           image_path: Path) -> Optional[FashionItem]:
        """
        Create a FashionItem from JSON annotation.
        
        Args:
            annotation: Single item annotation dictionary
            image_path: Path to image directory
            
        Returns:
            FashionItem object or None if invalid
        """
        # Extract required fields
        image_filename = annotation.get('image_name') or annotation.get('file_name')
        if not image_filename:
            return None
        
        full_image_path = image_path / image_filename
        if not full_image_path.exists():
            return None
        
        # Extract polygon coordinates
        polygon_data = annotation.get('polygon') or annotation.get('segmentation')
        if not polygon_data:
            return None
        
        # Convert polygon to bbox
        try:
            if isinstance(polygon_data[0], list):
                # Multiple polygons - use the first one
                polygon = [(int(polygon_data[0][i]), int(polygon_data[0][i+1])) 
                          for i in range(0, len(polygon_data[0]), 2)]
            else:
                # Single polygon
                polygon = [(int(polygon_data[i]), int(polygon_data[i+1])) 
                          for i in range(0, len(polygon_data), 2)]
            
            bbox = self.processor.polygon_to_bbox(polygon)
        except (ValueError, IndexError, TypeError):
            return None
        
        # Extract metadata fields
        category = annotation.get('category', '')
        style = annotation.get('style', [])
        silhouette = annotation.get('silhouette', '')
        material = annotation.get('material', [])
        detail = annotation.get('detail', [])
        
        # Ensure list fields are actually lists
        if not isinstance(style, list):
            style = [style] if style else []
        if not isinstance(material, list):
            material = [material] if material else []
        if not isinstance(detail, list):
            detail = [detail] if detail else []
        
        return FashionItem(
            image_path=str(full_image_path),
            bbox=bbox,
            category=category,
            style=style,
            silhouette=silhouette,
            material=material,
            detail=detail
        )
    
    def _has_valid_bbox(self, item: FashionItem) -> bool:
        """
        Check if a FashionItem has a valid bounding box.
        
        Args:
            item: FashionItem to check
            
        Returns:
            True if bbox is valid, False otherwise
        """
        x, y, width, height = item.bbox
        return width > 0 and height > 0 and x >= 0 and y >= 0
    
    def build_vocabularies(self) -> Dict[str, Dict[str, int]]:
        """
        Build vocabularies from file paths (samples subset for efficiency).
        
        Returns:
            Dictionary mapping field names to vocabularies
        """
        if not self._file_paths:
            raise ValueError("No file paths loaded. Call load_dataset_by_category() first.")
        
        print(f"\nBuilding vocabularies from {len(self._file_paths)} files...")
        
        # Sample files for vocabulary building (max 10000 files for speed)
        import random
        sample_size = min(10000, len(self._file_paths))
        sampled_paths = random.sample(self._file_paths, sample_size)
        
        print(f"Sampling {sample_size} files for vocabulary building...")
        
        # Collect tokens from sampled files
        field_tokens = {
            'category': set(),
            'style': set(),
            'silhouette': set(),
            'material': set(),
            'detail': set()
        }
        
        valid_count = 0
        for json_path, category in sampled_paths:
            item = self.parse_json_file(json_path, category)
            if item:
                field_tokens['category'].add(item.category)
                if item.silhouette:
                    field_tokens['silhouette'].add(item.silhouette)
                for s in item.style:
                    if s:
                        field_tokens['style'].add(s)
                for m in item.material:
                    if m:
                        field_tokens['material'].add(m)
                for d in item.detail:
                    if d:
                        field_tokens['detail'].add(d)
                valid_count += 1
        
        print(f"Processed {valid_count} valid items for vocabulary")
        
        # Build vocabularies with <UNK> token
        vocabularies = {}
        for field, tokens in field_tokens.items():
            vocab = {'<UNK>': 0}
            for i, token in enumerate(sorted(tokens), 1):
                vocab[token] = i
            vocabularies[field] = vocab
            print(f"  {field}: {len(vocab)} tokens")
        
        self.processor.vocabularies = vocabularies
        self._vocabularies_built = True
        return vocabularies
    
    def get_cropped_image(self, item: FashionItem) -> PIL.Image.Image:
        """
        Get cropped and resized image for a fashion item.
        
        Args:
            item: FashionItem to process
            
        Returns:
            Cropped and resized PIL Image
        """
        # Load original image
        image = Image.open(item.image_path).convert('RGB')
        
        # Crop using bbox with padding
        cropped = self.processor.crop_image_by_bbox(
            image, item.bbox, padding=self.crop_padding
        )
        
        # Resize to target size
        resized = cropped.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return resized
    
    def get_processed_json(self, item: FashionItem) -> Dict[str, any]:
        """
        Get processed JSON data with vocabulary indices.
        
        Args:
            item: FashionItem to process
            
        Returns:
            Dictionary with fields converted to vocabulary indices
        """
        if not self._vocabularies_built:
            raise ValueError("Vocabularies not built. Call build_vocabularies() first.")
        
        json_data = {
            'category': item.category,
            'style': item.style,
            'silhouette': item.silhouette,
            'material': item.material,
            'detail': item.detail
        }
        
        return self.processor.process_json_fields(json_data)
    
    def get_vocab_sizes(self) -> Dict[str, int]:
        """
        Get vocabulary sizes for each field.
        
        Returns:
            Dictionary mapping field names to vocabulary sizes
        """
        return self.processor.get_vocab_sizes()
    
    def __len__(self) -> int:
        """Return number of loaded fashion items."""
        return len(self._fashion_items)
    
    def __getitem__(self, idx: int) -> FashionItem:
        """Get fashion item by index."""
        return self._fashion_items[idx]
    
    def __iter__(self) -> Iterator[FashionItem]:
        """Iterate over fashion items."""
        return iter(self._fashion_items)
    
    def process_json_for_inference(self, style_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Process JSON style description for API inference.
        
        Args:
            style_dict: Dictionary containing style description
            
        Returns:
            Dictionary with tensors ready for model inference
        """
        if not self._vocabularies_built:
            raise ValueError("Vocabularies not built. Call build_vocabularies() first.")
        
        # Process the JSON data
        processed = self.processor.process_json_fields(style_dict)
        
        # Convert to tensors with batch dimension
        batch = {}
        
        # Single categorical fields
        batch['category'] = torch.tensor([processed['category']], dtype=torch.long)
        batch['silhouette'] = torch.tensor([processed['silhouette']], dtype=torch.long)
        
        # Multi-categorical fields with padding
        max_lengths = {'style': 10, 'material': 10, 'detail': 15}
        
        for field in ['style', 'material', 'detail']:
            ids = processed[field]
            max_len = max_lengths[field]
            
            # Pad or truncate to max length
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [0] * (max_len - len(ids))  # Pad with 0 (UNK token)
            
            batch[field] = torch.tensor([ids], dtype=torch.long)
            
            # Create mask (1 for valid tokens, 0 for padding)
            mask = [1 if i < len(processed[field]) else 0 for i in range(max_len)]
            batch[f'{field}_mask'] = torch.tensor([mask], dtype=torch.long)
        
        return batch