"""
GPU & Host Memory Manager with LRU Offloader and Device Auto-Selection.
Governs GPU VRAM allocation, dynamic device placement, and automatic offloading.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
import torch

logger = logging.getLogger("OmniVision.MemoryManager")


def auto_select_device() -> torch.device:
    """
    Automatically select optimal compute device:
    1. CUDA if available and operational
    2. MPS (Apple Silicon Metal Performance Shaders) if available
    3. CPU fallback
    """
    override = os.getenv("OMNIVISION_DEVICE", "auto").lower().strip()
    if override != "auto":
        return torch.device(override)

    if torch.cuda.is_available():
        try:
            # Probe device with a lightweight tensor allocation
            t = torch.zeros((1,), device="cuda:0")
            del t
            return torch.device("cuda:0")
        except Exception as e:
            logger.warning(f"CUDA device detected but initialization failed: {e}. Falling back.")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class GPUMemoryManager:
    """
    Resource arbiter maintaining model residency within strict VRAM thresholds.
    Implements LRU offloading of PyTorch models from GPU to Host CPU RAM.
    """

    def __init__(
        self,
        target_device: Optional[torch.device] = None,
        max_vram_fraction: float = 0.85,
        max_active_models: int = 4,
    ) -> None:
        self.device = target_device or auto_select_device()
        self.max_vram_fraction = max_vram_fraction
        self.max_active_models = max_active_models

        # Active resident models in VRAM: model_id -> (model_instance, last_access_timestamp)
        self._active_models: collections.OrderedDict[str, Tuple[torch.nn.Module, float]] = (
            collections.OrderedDict()
        )
        # Offloaded models residing in CPU memory: model_id -> model_instance
        self._offloaded_models: Dict[str, torch.nn.Module] = {}

        logger.info(f"Initialized GPUMemoryManager on device={self.device} (VRAM cap={max_vram_fraction*100:.0f}%)")

    def register_model(self, model_id: str, model: torch.nn.Module) -> None:
        """Register a loaded model into the memory manager."""
        self.touch_model(model_id, model)

    def touch_model(self, model_id: str, model: Optional[torch.nn.Module] = None) -> None:
        """Mark model as recently accessed and move to most-recently-used position."""
        if model_id in self._offloaded_models and model is None:
            model = self._offloaded_models.pop(model_id)

        if model is not None:
            # Ensure model is on the target device
            self.ensure_capacity(new_model_id=model_id)
            try:
                model.to(self.device)
            except Exception as e:
                logger.warning(f"Failed to place model '{model_id}' on {self.device}: {e}")

            if model_id in self._active_models:
                self._active_models.move_to_end(model_id)
            self._active_models[model_id] = (model, time.time())

    def ensure_capacity(self, new_model_id: str) -> None:
        """
        Check VRAM utilization and model count.
        Evict/offload least-recently-used models if limits are exceeded.
        """
        # Count-based constraint
        while len(self._active_models) >= self.max_active_models:
            # Pop least-recently-used model (first item in OrderedDict)
            oldest_id, (oldest_model, _) = self._active_models.popitem(last=False)
            if oldest_id == new_model_id:
                continue
            self._offload_model_to_cpu(oldest_id, oldest_model)

        # VRAM capacity constraint (if running on CUDA)
        if self.device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(self.device).total_memory
            allocated = torch.cuda.memory_allocated(self.device)
            fraction = allocated / total_vram

            while fraction > self.max_vram_fraction and self._active_models:
                oldest_id, (oldest_model, _) = self._active_models.popitem(last=False)
                if oldest_id == new_model_id:
                    break
                self._offload_model_to_cpu(oldest_id, oldest_model)
                allocated = torch.cuda.memory_allocated(self.device)
                fraction = allocated / total_vram

    def _offload_model_to_cpu(self, model_id: str, model: torch.nn.Module) -> None:
        """Move PyTorch model to host CPU RAM and reclaim GPU cache."""
        logger.info(f"LRU Offloader: Transferring model '{model_id}' from {self.device} to CPU RAM.")
        try:
            model.to("cpu")
            self._offloaded_models[model_id] = model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during offload of '{model_id}': {e}")

    def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """Fetch model by ID, restoring it to GPU if previously offloaded."""
        if model_id in self._active_models:
            model, _ = self._active_models[model_id]
            self.touch_model(model_id, model)
            return model
        elif model_id in self._offloaded_models:
            model = self._offloaded_models[model_id]
            self.touch_model(model_id, model)
            return model
        return None

    def release_all(self) -> None:
        """Evict and deallocate all resident and offloaded models."""
        self._active_models.clear()
        self._offloaded_models.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @contextlib.contextmanager
    def scoped_allocation(self) -> Generator[None, None, None]:
        """Context manager guaranteeing CUDA cache flushing upon block exit."""
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def get_status(self) -> Dict[str, Any]:
        """Telemetry snapshot of memory allocations and active models."""
        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0
        vram_total_mb = 0.0

        if self.device.type == "cuda":
            vram_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            vram_reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            vram_total_mb = (
                torch.cuda.get_device_properties(self.device).total_memory / (1024 * 1024)
            )

        return {
            "device": str(self.device),
            "active_models_gpu": list(self._active_models.keys()),
            "offloaded_models_cpu": list(self._offloaded_models.keys()),
            "vram_allocated_mb": round(vram_allocated_mb, 2),
            "vram_reserved_mb": round(vram_reserved_mb, 2),
            "vram_total_mb": round(vram_total_mb, 2),
        }
