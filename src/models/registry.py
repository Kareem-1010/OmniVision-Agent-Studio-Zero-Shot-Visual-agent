"""
Centralized Model Registry and Lazy-Loading Engine.
Coordinates on-demand model instantiation, precision casting (FP16/INT8/BF16),
optional torch.compile graph optimization, and GPU memory lifecycle management.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union
import torch

from src.models.cache import ContentHashCache
from src.models.memory_manager import GPUMemoryManager, auto_select_device

logger = logging.getLogger("OmniVision.ModelRegistry")


@dataclass
class ModelSpec:
    """Specification defining loading and execution constraints for a neural model."""
    model_id: str
    loader: Callable[..., Any]
    precision: str = "fp16"  # 'fp32', 'fp16', 'bf16', 'int8'
    enable_compile: bool = False
    compile_mode: str = "reduce-overhead"
    device: Optional[Union[str, torch.device]] = None
    description: str = ""
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Thread-safe catalog and lazy-instantiation manager for neural vision models."""

    def __init__(
        self,
        memory_manager: Optional[GPUMemoryManager] = None,
        cache: Optional[ContentHashCache] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.memory_manager = memory_manager or GPUMemoryManager()
        self.cache = cache or ContentHashCache()
        self._specs: Dict[str, ModelSpec] = {}
        self._instances: Dict[str, Any] = {}

    def register(self, spec: ModelSpec) -> None:
        """Register a model specification into the catalog."""
        with self._lock:
            self._specs[spec.model_id] = spec
            logger.debug(f"Registered model specification: {spec.model_id} (precision={spec.precision})")

    def get(self, model_id: str, **override_kwargs: Any) -> Any:
        """
        Lazily load or retrieve an already instantiated model.
        Restores offloaded model to GPU if managed by GPUMemoryManager.
        """
        with self._lock:
            if model_id not in self._specs:
                raise KeyError(
                    f"Model '{model_id}' is not registered. Available: {list(self._specs.keys())}"
                )

            # Check if managed by GPUMemoryManager
            managed_instance = self.memory_manager.get_model(model_id)
            if managed_instance is not None:
                return managed_instance

            # Check local instance cache
            if model_id in self._instances:
                instance = self._instances[model_id]
                if isinstance(instance, torch.nn.Module):
                    self.memory_manager.touch_model(model_id, instance)
                return instance

            # Lazy load model
            spec = self._specs[model_id]
            logger.info(f"Lazy-loading model '{model_id}' ({spec.description or 'No description'})...")

            load_args = {**spec.extra_kwargs, **override_kwargs}
            instance = spec.loader(**load_args)

            # Apply precision and optimization if PyTorch Module
            if isinstance(instance, torch.nn.Module):
                instance = self._optimize_module(instance, spec)
                self.memory_manager.register_model(model_id, instance)

            self._instances[model_id] = instance
            return instance

    def _optimize_module(self, module: torch.nn.Module, spec: ModelSpec) -> torch.nn.Module:
        """Apply precision casting, device placement, and torch.compile."""
        device = spec.device or self.memory_manager.device
        if isinstance(device, str):
            device = torch.device(device)

        # Precision casting
        # Note: Half/Bfloat16 precision is applied only if device is CUDA/MPS to prevent CPU slowdown
        if spec.precision == "fp16" and device.type in ("cuda", "mps"):
            module = module.half()
        elif spec.precision == "bf16" and device.type in ("cuda", "mps"):
            module = module.bfloat16()
        elif spec.precision == "int8":
            try:
                # Dynamic quantization for linear layers
                module = torch.ao.quantization.quantize_dynamic(
                    module, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                logger.warning(f"INT8 quantization skipped for {spec.model_id}: {e}")

        # Device placement
        module = module.to(device)

        # Graph compilation (torch.compile)
        if spec.enable_compile and hasattr(torch, "compile") and device.type == "cuda":
            try:
                logger.info(f"Applying torch.compile(mode='{spec.compile_mode}') to {spec.model_id}")
                module = torch.compile(module, mode=spec.compile_mode)  # type: ignore
            except Exception as e:
                logger.warning(f"torch.compile failed for {spec.model_id}: {e}. Proceeding eager.")

        module.eval()
        return module

    def unload(self, model_id: str) -> bool:
        """Explicitly unload an instantiated model and release memory."""
        with self._lock:
            if model_id in self._instances:
                instance = self._instances.pop(model_id)
                del instance
                if model_id in self.memory_manager._active_models:
                    del self.memory_manager._active_models[model_id]
                if model_id in self.memory_manager._offloaded_models:
                    del self.memory_manager._offloaded_models[model_id]
                if self.memory_manager.device.type == "cuda":
                    torch.cuda.empty_cache()
                logger.info(f"Unloaded model '{model_id}' successfully.")
                return True
            return False

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models, their status, and metadata."""
        with self._lock:
            out = {}
            for mid, spec in self._specs.items():
                is_loaded = mid in self._instances
                out[mid] = {
                    "is_loaded": is_loaded,
                    "precision": spec.precision,
                    "device": str(spec.device or self.memory_manager.device),
                    "compiled": spec.enable_compile,
                    "description": spec.description,
                }
            return out


_GLOBAL_MODEL_REGISTRY = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Access global singleton model registry."""
    return _GLOBAL_MODEL_REGISTRY
