"""
Unit tests for ContentHashCache and artifact hashing.
"""

import numpy as np
import torch
from PIL import Image
from src.models.cache import ContentHashCache, hash_artifact


def test_deterministic_hashing():
    # Numpy array
    arr1 = np.ones((10, 10), dtype=np.uint8)
    arr2 = np.ones((10, 10), dtype=np.uint8)
    assert hash_artifact(arr1) == hash_artifact(arr2)

    # PIL Image
    img1 = Image.new("RGB", (32, 32), color="red")
    img2 = Image.new("RGB", (32, 32), color="red")
    assert hash_artifact(img1) == hash_artifact(img2)

    # Torch Tensor
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([1.0, 2.0, 3.0])
    assert hash_artifact(t1) == hash_artifact(t2)


def test_cache_tier_promotion(isolated_cache: ContentHashCache):
    key = "test_item"
    val = {"predictions": [1, 2, 3]}

    isolated_cache.set(key, val, ttl_seconds=3600)

    # First fetch from L1
    fetched1 = isolated_cache.get(key)
    assert fetched1 == val

    # Clear L1 to test L2 disk fallback
    isolated_cache._l1.clear()
    fetched2 = isolated_cache.get(key)
    assert fetched2 == val

    # Verify item was promoted back into L1
    assert key in isolated_cache._l1
