"""
Unit tests for GPU Memory Manager and LRU Model Offloader.
"""

import torch
from src.models.memory_manager import GPUMemoryManager


class DummyModule(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.fc = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.fc(x)


def test_lru_model_offload(isolated_memory_manager: GPUMemoryManager):
    # Manager max_active_models is 2
    m1 = DummyModule("model_1")
    m2 = DummyModule("model_2")
    m3 = DummyModule("model_3")

    isolated_memory_manager.register_model("m1", m1)
    isolated_memory_manager.register_model("m2", m2)

    status = isolated_memory_manager.get_status()
    assert len(status["active_models_gpu"]) == 2

    # Registering 3rd model should offload oldest (m1) to CPU
    isolated_memory_manager.register_model("m3", m3)

    status = isolated_memory_manager.get_status()
    assert len(status["active_models_gpu"]) == 2
    assert "m1" in status["offloaded_models_cpu"]

    # Accessing m1 should restore it and offload the next oldest
    retrieved_m1 = isolated_memory_manager.get_model("m1")
    assert retrieved_m1 is not None

    status = isolated_memory_manager.get_status()
    assert "m1" in status["active_models_gpu"]
    assert "m2" in status["offloaded_models_cpu"]
