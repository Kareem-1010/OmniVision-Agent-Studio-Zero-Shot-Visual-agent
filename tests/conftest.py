"""
Pytest configuration and shared fixtures for OmniVision Agent tests.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator
import pytest

from src.models.cache import ContentHashCache
from src.models.memory_manager import GPUMemoryManager
from src.models.registry import ModelRegistry
from src.utils.tracker import MetricsTracker


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for cache isolation during tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="omnivision_test_cache_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def isolated_tracker() -> MetricsTracker:
    """Provide a fresh, isolated metrics tracker."""
    return MetricsTracker()


@pytest.fixture
def isolated_cache(temp_cache_dir: Path) -> ContentHashCache:
    """Provide a cache instance bound to an isolated temporary directory."""
    return ContentHashCache(cache_dir=temp_cache_dir, l1_max_items=10)


@pytest.fixture
def isolated_memory_manager() -> GPUMemoryManager:
    """Provide a memory manager with limited capacity for offloader testing."""
    return GPUMemoryManager(max_active_models=2)


@pytest.fixture
def isolated_registry(
    isolated_memory_manager: GPUMemoryManager,
    isolated_cache: ContentHashCache,
) -> ModelRegistry:
    """Provide an isolated model registry."""
    return ModelRegistry(memory_manager=isolated_memory_manager, cache=isolated_cache)
