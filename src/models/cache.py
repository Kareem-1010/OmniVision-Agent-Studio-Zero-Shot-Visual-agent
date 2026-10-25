"""
Content-Addressed Multi-Tier Cache Subsystem.
Provides deterministic cryptographic hashing (SHA-256/BLAKE2b) and caching across:
- L1: In-memory LRU cache
- L2: Persistent disk cache
- L3: Optional distributed Redis cache
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger("OmniVision.Cache")


def hash_artifact(data: Any) -> str:
    """
    Produce a deterministic, collision-resistant content hash for diverse vision artifacts.
    Handles PIL Images, Torch Tensors, NumPy arrays, dicts, strings, and primitives.
    """
    hasher = hashlib.sha256()

    if isinstance(data, (str, bytes)):
        if isinstance(data, str) and len(data) < 1024 and os.path.isfile(data):
            try:
                stat = os.stat(data)
                hasher.update(f"file:{data}:{stat.st_mtime}:{stat.st_size}:".encode("utf-8"))
                with open(data, "rb") as f:
                    hasher.update(f.read())
                return hasher.hexdigest()
            except Exception:
                pass
        b = data.encode("utf-8") if isinstance(data, str) else data
        hasher.update(b"str_bytes:")
        hasher.update(b)
    elif isinstance(data, (int, float, bool)):
        hasher.update(f"primitive:{type(data)}:{data}".encode("utf-8"))
    elif isinstance(data, Image.Image):
        hasher.update(f"pil_image:{data.size}:{data.mode}:".encode("utf-8"))
        buf = io.BytesIO()
        data.save(buf, format="PNG")
        hasher.update(buf.getvalue())
    elif isinstance(data, np.ndarray):
        hasher.update(f"numpy:{data.dtype}:{data.shape}:".encode("utf-8"))
        hasher.update(data.tobytes())
    elif isinstance(data, torch.Tensor):
        cpu_t = data.detach().cpu().contiguous()
        hasher.update(f"torch:{cpu_t.dtype}:{tuple(cpu_t.shape)}:".encode("utf-8"))
        hasher.update(cpu_t.numpy().tobytes())
    elif isinstance(data, dict):
        hasher.update(b"dict:")
        for k in sorted(data.keys()):
            hasher.update(str(k).encode("utf-8"))
            hasher.update(hash_artifact(data[k]).encode("utf-8"))
    elif isinstance(data, (list, tuple)):
        hasher.update(b"seq:")
        for item in data:
            hasher.update(hash_artifact(item).encode("utf-8"))
    else:
        # Fallback to pickle byte representation
        hasher.update(b"pickle:")
        hasher.update(pickle.dumps(data))

    return hasher.hexdigest()


class ContentHashCache:
    """Multi-level content-addressed cache (L1 Memory, L2 Disk, optional L3 Redis)."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = ".cache/omnivision",
        l1_max_items: int = 256,
        default_ttl_seconds: int = 86400,
        enable_redis: bool = False,
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.l1_max_items = l1_max_items
        self.default_ttl_seconds = default_ttl_seconds

        # L1 In-Memory LRU (key -> (value, expiry_timestamp))
        self._l1: collections.OrderedDict[str, Tuple[Any, float]] = collections.OrderedDict()

        # Telemetry
        self._hits_l1 = 0
        self._hits_l2 = 0
        self._hits_l3 = 0
        self._misses = 0

        # L3 Redis Client setup (lazy / optional)
        self.enable_redis = enable_redis
        self.redis_url = redis_url
        self._redis_client = None
        if self.enable_redis:
            try:
                import redis

                self._redis_client = redis.from_url(self.redis_url)
                self._redis_client.ping()
                logger.info(f"Connected to Redis cache at {self.redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis cache: {e}. Falling back to L1+L2.")
                self.enable_redis = False
                self._redis_client = None

    def compute_key(self, namespace: str, *args: Any, **kwargs: Any) -> str:
        """Derive a deterministic content hash key from arguments."""
        h = hashlib.sha256()
        h.update(namespace.encode("utf-8"))
        for arg in args:
            h.update(hash_artifact(arg).encode("utf-8"))
        for k in sorted(kwargs.keys()):
            h.update(str(k).encode("utf-8"))
            h.update(hash_artifact(kwargs[k]).encode("utf-8"))
        return f"{namespace}:{h.hexdigest()}"

    def get(self, key: str) -> Optional[Any]:
        """Fetch item by key across L1 -> L2 -> L3 tiers."""
        now = time.time()

        # 1. L1 Memory Lookup
        if key in self._l1:
            val, expiry = self._l1[key]
            if expiry > now:
                self._l1.move_to_end(key)
                self._hits_l1 += 1
                return val
            else:
                del self._l1[key]

        # 2. L2 Disk Lookup
        disk_path = self._get_disk_path(key)
        if disk_path.exists():
            try:
                with open(disk_path, "rb") as f:
                    entry = pickle.load(f)
                val, expiry = entry["value"], entry["expiry"]
                if expiry > now:
                    self._hits_l2 += 1
                    # Promote to L1
                    self._set_l1(key, val, expiry)
                    return val
                else:
                    disk_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Corrupted disk cache entry for {key}: {e}")
                disk_path.unlink(missing_ok=True)

        # 3. L3 Redis Lookup
        if self.enable_redis and self._redis_client:
            try:
                raw_bytes = self._redis_client.get(key)
                if raw_bytes:
                    entry = pickle.loads(raw_bytes)
                    val, expiry = entry["value"], entry["expiry"]
                    if expiry > now:
                        self._hits_l3 += 1
                        self._set_l1(key, val, expiry)
                        self._set_l2(key, val, expiry)
                        return val
            except Exception as e:
                logger.warning(f"Redis get error for {key}: {e}")

        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store item into cache tiers with specified TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        expiry = time.time() + ttl

        # Save to L1
        self._set_l1(key, value, expiry)

        # Save to L2
        self._set_l2(key, value, expiry)

        # Save to L3
        if self.enable_redis and self._redis_client:
            try:
                payload = pickle.dumps({"value": value, "expiry": expiry})
                self._redis_client.setex(key, ttl, payload)
            except Exception as e:
                logger.warning(f"Redis set error for {key}: {e}")

    def _set_l1(self, key: str, value: Any, expiry: float) -> None:
        """Insert or update entry in L1 memory cache with LRU eviction."""
        if key in self._l1:
            self._l1.move_to_end(key)
        self._l1[key] = (value, expiry)
        if len(self._l1) > self.l1_max_items:
            self._l1.popitem(last=False)

    def _set_l2(self, key: str, value: Any, expiry: float) -> None:
        """Persist entry to L2 disk cache."""
        disk_path = self._get_disk_path(key)
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            temp_path = disk_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                pickle.dump({"value": value, "expiry": expiry}, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.replace(disk_path)
        except Exception as e:
            logger.warning(f"Failed to write disk cache for {key}: {e}")

    def _get_disk_path(self, key: str) -> Path:
        """Compute disk filesystem path for a cache key."""
        safe_name = hashlib.sha256(key.encode("utf-8")).hexdigest()
        # 2-level directory sharding
        return self.cache_dir / safe_name[:2] / safe_name[2:4] / f"{safe_name}.cache"

    def clear(self) -> None:
        """Flush L1 memory and L2 disk caches."""
        self._l1.clear()
        import shutil

        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits_l1 = 0
        self._hits_l2 = 0
        self._hits_l3 = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total_requests = self._hits_l1 + self._hits_l2 + self._hits_l3 + self._misses
        hit_rate = (
            ((self._hits_l1 + self._hits_l2 + self._hits_l3) / total_requests)
            if total_requests > 0
            else 0.0
        )
        return {
            "l1_items": len(self._l1),
            "l1_hits": self._hits_l1,
            "l2_hits": self._hits_l2,
            "l3_hits": self._hits_l3,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 4),
        }
