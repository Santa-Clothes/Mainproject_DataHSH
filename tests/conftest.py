"""Shared pytest fixtures for all tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def sample_vocab_sizes():
    """Standard vocabulary sizes for testing."""
    return {
        "category": 23,
        "style": 4,
        "silhouette": 8,
        "material": 25,
        "detail": 41,
    }


@pytest.fixture
def small_batch_size():
    """Small batch size for quick tests."""
    return 4


@pytest.fixture
def medium_batch_size():
    """Medium batch size for tests."""
    return 16
