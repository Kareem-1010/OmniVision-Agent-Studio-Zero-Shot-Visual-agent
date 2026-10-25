"""
Unit tests for Model Registry.
"""

import torch
from src.models.registry import ModelRegistry, ModelSpec


class SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_lazy_loading_and_optimization(isolated_registry: ModelRegistry):
    load_count = 0

    def mock_loader():
        nonlocal load_count
        load_count += 1
        return SampleModel()

    spec = ModelSpec(
        model_id="sample_detector",
        loader=mock_loader,
        precision="fp32",
        description="Sample test detector",
    )
    isolated_registry.register(spec)

    # Model should not be loaded yet
    models_status = isolated_registry.list_models()
    assert models_status["sample_detector"]["is_loaded"] is False
    assert load_count == 0

    # Retrieve model (triggers lazy load)
    instance1 = isolated_registry.get("sample_detector")
    assert isinstance(instance1, SampleModel)
    assert load_count == 1

    # Second retrieval should use instance cache without re-invoking loader
    instance2 = isolated_registry.get("sample_detector")
    assert instance2 is instance1
    assert load_count == 1

    # Test explicit unloader
    unloaded = isolated_registry.unload("sample_detector")
    assert unloaded is True
    assert isolated_registry.list_models()["sample_detector"]["is_loaded"] is False
