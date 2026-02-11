"""
Data processing utilities for the Fashion JSON Encoder system.

This module provides the FashionDataProcessor class for handling K-Fashion dataset
preprocessing, including polygon to bbox conversion, image cropping, and vocabulary building.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import PIL.Image


class FashionDataProcessor:
    """
    Data processor for K-Fashion dataset preprocessing.

    Handles polygon to bbox conversion, image cropping, vocabulary building,
    and JSON field processing for the fashion JSON encoder system.
    """

    def __init__(self, dataset_path: str, target_categories: List[str] = None):
        """
        Initialize the data processor.

        Args:
            dataset_path: Path to the K-Fashion dataset
            target_categories: List of target categories to process
        """
        self.dataset_path = Path(dataset_path)
        self.target_categories = target_categories  # None = all categories
        self.vocabularies = {}

    def polygon_to_bbox(self, polygon: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """
        Convert polygon coordinates to bounding box.

        Args:
            polygon: List of (x, y) coordinate tuples

        Returns:
            Tuple of (x, y, width, height) for bounding box
        """
        if not polygon:
            raise ValueError("Empty polygon provided")

        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        width = max_x - min_x
        height = max_y - min_y

        return (min_x, min_y, width, height)

    def crop_image_by_bbox(
        self, image: PIL.Image.Image, bbox: Tuple[int, int, int, int], padding: float = 0.1
    ) -> PIL.Image.Image:
        """
        Crop image using bounding box with optional padding.

        Args:
            image: PIL Image to crop
            bbox: (x, y, width, height) bounding box
            padding: Padding ratio to add around bbox

        Returns:
            Cropped PIL Image
        """
        x, y, width, height = bbox

        # Add padding
        pad_x = int(width * padding)
        pad_y = int(height * padding)

        # Calculate crop coordinates with padding
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(image.width, x + width + pad_x)
        bottom = min(image.height, y + height + pad_y)

        # Crop image
        cropped = image.crop((left, top, right, bottom))

        return cropped

    def build_vocabulary(self, json_files: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Build vocabulary for each JSON field from training data.

        Args:
            json_files: List of paths to JSON metadata files

        Returns:
            Dictionary mapping field names to {token: index} vocabularies
        """
        field_tokens = {
            "category": set(),
            "style": set(),
            "silhouette": set(),
            "material": set(),
            "detail": set(),
        }

        # Collect all unique tokens from JSON files
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Filter by target categories
            if data.get("category") not in self.target_categories:
                continue

            # Collect tokens for each field
            field_tokens["category"].add(data.get("category", ""))
            field_tokens["silhouette"].add(data.get("silhouette", ""))

            # Multi-categorical fields
            for style in data.get("style", []):
                field_tokens["style"].add(style)
            for material in data.get("material", []):
                field_tokens["material"].add(material)
            for detail in data.get("detail", []):
                field_tokens["detail"].add(detail)

        # Build vocabularies with <UNK> token
        vocabularies = {}
        for field, tokens in field_tokens.items():
            vocab = {"<UNK>": 0}  # OOV token at index 0
            for i, token in enumerate(sorted(tokens), 1):
                if token:  # Skip empty strings
                    vocab[token] = i
            vocabularies[field] = vocab

        self.vocabularies = vocabularies
        return vocabularies

    def process_json_fields(self, json_data: Dict) -> Dict[str, Union[int, List[int]]]:
        """
        Convert JSON fields to vocabulary indices.

        Args:
            json_data: Raw JSON metadata dictionary

        Returns:
            Dictionary with fields converted to vocabulary indices
        """
        if not self.vocabularies:
            raise ValueError("Vocabularies not built. Call build_vocabulary() first.")

        processed = {}

        # Single categorical fields
        processed["category"] = self.vocabularies["category"].get(
            json_data.get("category", ""), 0  # Default to <UNK>
        )
        processed["silhouette"] = self.vocabularies["silhouette"].get(
            json_data.get("silhouette", ""), 0  # Default to <UNK>
        )

        # Multi categorical fields
        processed["style"] = [
            self.vocabularies["style"].get(style, 0) for style in json_data.get("style", [])
        ]
        processed["material"] = [
            self.vocabularies["material"].get(material, 0)
            for material in json_data.get("material", [])
        ]
        processed["detail"] = [
            self.vocabularies["detail"].get(detail, 0) for detail in json_data.get("detail", [])
        ]

        return processed

    def get_vocab_sizes(self) -> Dict[str, int]:
        """
        Get vocabulary sizes for each field.

        Returns:
            Dictionary mapping field names to vocabulary sizes
        """
        if not self.vocabularies:
            raise ValueError("Vocabularies not built. Call build_vocabulary() first.")

        return {field: len(vocab) for field, vocab in self.vocabularies.items()}
