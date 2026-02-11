"""Tests for FashionMetadataProcessor."""

import pytest

from src.data.processor import FashionMetadataProcessor


class TestFashionMetadataProcessor:
    """Test suite for FashionMetadataProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance with sample categories."""
        return FashionMetadataProcessor(target_categories=None)  # All categories

    @pytest.fixture
    def sample_json_data(self):
        """Sample JSON data from K-Fashion dataset."""
        return {
            "데이터셋 정보": {
                "데이터셋 상세설명": {
                    "라벨링": {
                        "스타일": [{"스타일": "레트로", "서브스타일": "리조트"}],
                        "상의": [
                            {
                                "카테고리": "셔츠/블라우스",
                                "색상": "블루",
                                "소재": ["우븐"],
                                "넥라인": "스탠드칼라",
                            }
                        ],
                        "하의": [{}],
                        "아우터": [{}],
                        "원피스": [{}],
                    }
                }
            }
        }

    def test_processor_initialization(self, processor):
        """Test processor initializes with vocabularies."""
        assert hasattr(processor, "vocabularies")
        assert "category" in processor.vocabularies
        assert "style" in processor.vocabularies
        assert "silhouette" in processor.vocabularies

    def test_process_json(self, processor, sample_json_data):
        """Test JSON processing returns correct structure."""
        result = processor.process_json(sample_json_data)

        assert "category_id" in result
        assert "style_ids" in result
        assert "silhouette_id" in result
        assert "material_ids" in result
        assert "detail_ids" in result

    def test_category_encoding(self, processor):
        """Test category ID encoding."""
        # Should encode known categories
        assert processor.vocabularies["category"]["셔츠/블라우스"] >= 0

    def test_style_encoding(self, processor):
        """Test style ID encoding."""
        # Should encode known styles
        assert processor.vocabularies["style"]["레트로"] >= 0
        assert processor.vocabularies["style"]["로맨틱"] >= 0

    def test_material_encoding(self, processor):
        """Test material ID encoding."""
        # Should encode known materials
        assert processor.vocabularies["material"]["우븐"] >= 0

    def test_unknown_category_handling(self, processor):
        """Test handling of unknown categories."""
        unknown_json = {
            "데이터셋 정보": {
                "데이터셋 상세설명": {
                    "라벨링": {
                        "스타일": [{"스타일": "UnknownStyle"}],
                        "상의": [{"카테고리": "UnknownCategory"}],
                        "하의": [{}],
                        "아우터": [{}],
                        "원피스": [{}],
                    }
                }
            }
        }
        result = processor.process_json(unknown_json)
        # Should handle gracefully (return UNK or skip)
        assert result is not None

    def test_vocabulary_sizes(self, processor):
        """Test vocabulary sizes are reasonable."""
        vocab_sizes = processor.get_vocab_sizes()

        assert vocab_sizes["category"] == 23  # Total K-Fashion categories
        assert vocab_sizes["style"] > 0
        assert vocab_sizes["silhouette"] > 0
        assert vocab_sizes["material"] > 0
        assert vocab_sizes["detail"] > 0

    def test_empty_fields_handling(self, processor):
        """Test handling of empty optional fields."""
        minimal_json = {
            "데이터셋 정보": {
                "데이터셋 상세설명": {
                    "라벨링": {
                        "스타일": [{}],
                        "상의": [{}],
                        "하의": [{}],
                        "아우터": [{}],
                        "원피스": [{}],
                    }
                }
            }
        }
        result = processor.process_json(minimal_json)
        # Should return default values for missing fields
        assert result is not None
