"""
Unit and Integration Tests for Phase 8 Interfaces and Performance.
Covers:
- ONNX Model Export and ONNX Runtime validation
- INT8 Dynamic Quantization
- Dynamic Batching Queue
- Async Tool Dispatcher
- Performance Profiler & Bottleneck Analysis
- FastAPI REST endpoints, Async Job Queue & Authentication
- Gradio UI construction and callbacks
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.api.service import AgentRunRequest, app, job_manager
from src.models.optimization import (
    AsyncToolDispatcher,
    DynamicBatcher,
    ModelQuantizer,
    ONNXExporter,
    PerformanceProfiler,
    ProfilingReport,
    QuantizationResult,
)
from src.ui.app import _compare_baseline, _run_agent_pipeline, build_demo


# ---------------------------------------------------------------------------
# Test ONNX Export and Quantization
# ---------------------------------------------------------------------------

class SampleVisionHead(nn.Module):
    """Simple test module for export and quantization."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_onnx_exporter():
    model = SampleVisionHead()
    sample_input = torch.randn(2, 32)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "test_model.onnx"
        exported_path = ONNXExporter.export_to_onnx(
            model=model,
            sample_input=sample_input,
            output_path=str(onnx_file),
            input_names=["input_feats"],
            output_names=["output_coords"],
        )

        assert Path(exported_path).exists()
        assert Path(exported_path).stat().st_size > 500


def test_model_quantizer():
    model = SampleVisionHead()
    quantized_model, metrics = ModelQuantizer.quantize_dynamic_int8(model)

    assert isinstance(metrics, QuantizationResult)
    assert metrics.original_size_bytes > 0
    assert metrics.quantized_size_bytes > 0
    assert metrics.original_latency_ms > 0.0
    assert metrics.quantized_latency_ms > 0.0


def test_dynamic_batcher():
    async def _run():
        call_count = 0

        def mock_batch_runner(items: list[int]) -> list[int]:
            nonlocal call_count
            call_count += 1
            return [x * 2 for x in items]

        batcher = DynamicBatcher(batch_fn=mock_batch_runner, max_batch_size=4, timeout_ms=10.0)

        # Launch 4 requests concurrently
        tasks = [batcher.enqueue(i) for i in range(4)]
        results = await asyncio.gather(*tasks)

        assert results == [0, 2, 4, 6]
        assert call_count == 1

    asyncio.run(_run())


def test_async_tool_dispatcher():
    async def _run():
        class MockSyncDispatcher:
            def execute(self, tool_name: str, args: dict):
                return {"tool": tool_name, "success": True}

        sync_disp = MockSyncDispatcher()
        async_disp = AsyncToolDispatcher(sync_disp)

        calls = [
            ("detector_ensemble", {"conf": 0.5}),
            ("metric_depth", {"mode": "relative"}),
            ("scene_ocr", {}),
        ]

        results = await async_disp.execute_parallel(calls)
        assert len(results) == 3
        assert results[0]["tool"] == "detector_ensemble"
        assert results[1]["tool"] == "metric_depth"
        assert results[2]["tool"] == "scene_ocr"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test Performance Profiler
# ---------------------------------------------------------------------------

def test_performance_profiler():
    profiler = PerformanceProfiler()

    # Simulate realistic pipeline stage timings
    for _ in range(10):
        profiler.record("Planner", np.random.uniform(15.0, 30.0))
        profiler.record("detector_ensemble", np.random.uniform(50.0, 90.0))
        profiler.record("metric_depth", np.random.uniform(30.0, 60.0))
        profiler.record("Synthesizer", np.random.uniform(10.0, 20.0))

    report = profiler.generate_report()
    assert isinstance(report, ProfilingReport)
    assert report.total_calls == 40
    assert report.total_duration_ms > 0.0
    assert len(report.components) == 4
    # Highest latency stage should be detector_ensemble
    assert report.primary_bottleneck == "detector_ensemble"
    assert len(report.recommendations) > 0
    assert any("detector_ensemble" in r for r in report.recommendations)


# ---------------------------------------------------------------------------
# Test FastAPI REST & Async Job Queue
# ---------------------------------------------------------------------------

client = TestClient(app)


def test_api_health_endpoint():
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "retail" in data["registered_presets"]
    assert "gpu_status" in data


def test_api_presets_endpoint():
    resp = client.get("/v1/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert "retail" in data
    assert "accessibility" in data
    assert "hazard_safety" in data
    assert data["retail"]["name"] == "retail"


def test_api_sync_run_with_preset():
    payload = {
        "query": "Assess retail traffic and shelf status",
        "media_uri": "test_store.jpg",
        "media_type": "image",
        "preset_name": "retail",
    }
    resp = client.post("/v1/agent/run", json=payload, headers={"X-API-Key": "omni_test_token"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["calibrated_confidence"] >= 0.90


def test_api_async_job_submission_and_polling():
    payload = {
        "query": "Assess hazard risks",
        "media_uri": "test_plant.jpg",
        "media_type": "image",
        "preset_name": "hazard_safety",
    }
    # 1. Submit async job
    sub_resp = client.post("/v1/agent/submit", json=payload, headers={"X-API-Key": "omni_test_token"})
    assert sub_resp.status_code == 202
    sub_data = sub_resp.json()
    job_id = sub_data["job_id"]
    assert job_id.startswith("job_")

    # 2. Poll job status
    poll_resp = client.get(f"/v1/agent/jobs/{job_id}")
    assert poll_resp.status_code == 200
    poll_data = poll_resp.json()
    assert poll_data["job_id"] == job_id
    assert poll_data["status"] in ["queued", "running", "completed"]


# ---------------------------------------------------------------------------
# Test Gradio UI Construction & Callbacks
# ---------------------------------------------------------------------------

def test_gradio_ui_building():
    demo = build_demo()
    assert demo is not None
    assert hasattr(demo, "launch")


def test_gradio_callbacks():
    # Test agent execution callback with preset
    out = _run_agent_pipeline(
        image=None,
        query="Verify PPE compliance",
        preset_name="hazard_safety",
        budget_limit=0.35,
        max_iter=6,
        enable_tree_search=True,
        enable_active_perception=True,
    )
    assert len(out) == 8
    ans_text, conf, ci, claims, telem, dag, timings, depth = out
    assert "[HAZARD_SAFETY PRESET RESULT]" in ans_text
    assert conf >= 0.90

    # Test baseline comparison callback
    omni_ans, omni_met, base_ans, base_met = _compare_baseline("Check hard hats")
    assert "Verified Grounded Analysis" in omni_ans
    assert "Hallucination Risk: HIGH" in base_met
