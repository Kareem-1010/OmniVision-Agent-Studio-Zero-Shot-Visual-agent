#!/usr/bin/env python3
"""
OmniVision Agent - Phase 1 Foundations Demonstration.
Demonstrates:
- LLM Factory & prompt registry invocation
- Constrained JSON parsing & automated JSON repair
- Model registry lazy loading & GPU memory manager LRU offloading
- Content-addressed cache (L1/L2)
- Typed vision tool execution with observation IDs and uncertainty quantification
- Telemetry & cost reporting
"""

from __future__ import annotations

import logging
import pprint
import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Tuple, Type
import torch
from pydantic import BaseModel, Field

from src.models.cache import ContentHashCache
from src.models.memory_manager import GPUMemoryManager, auto_select_device
from src.models.registry import ModelRegistry, ModelSpec
from src.tools.base import (
    BaseVisionTool,
    ToolMetadata,
    UncertaintyEstimate,
    ToolResult,
)
from src.utils.json_repair import parse_json_safely, repair_json
from src.utils.llm_factory import LLMFactory
from src.utils.prompt_registry import get_prompt_registry
from src.utils.tracker import get_global_tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OmniVision.Phase1Demo")


# -------------------------------------------------------------------------
# 1. Define Dummy Vision Neural Module for Registry Verification
# -------------------------------------------------------------------------
class SampleVisionEncoder(torch.nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


# -------------------------------------------------------------------------
# 2. Define Concrete Typed Vision Tool
# -------------------------------------------------------------------------
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ObjectDetection(BaseModel):
    box: BoundingBox
    category: str
    confidence: float


class DetectionToolInput(BaseModel):
    image_uri: str
    target_queries: List[str]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class DetectionToolOutput(BaseModel):
    detections: List[ObjectDetection]
    total_found: int


class DemoDetectionTool(BaseVisionTool[DetectionToolInput, DetectionToolOutput]):
    @property
    def input_schema(self) -> Type[DetectionToolInput]:
        return DetectionToolInput

    @property
    def output_schema(self) -> Type[DetectionToolOutput]:
        return DetectionToolOutput

    def _run(self, input_data: DetectionToolInput) -> Tuple[DetectionToolOutput, UncertaintyEstimate]:
        # Synthetic deterministic perception output
        detections = [
            ObjectDetection(
                box=BoundingBox(x1=50.0, y1=80.0, x2=220.0, y2=400.0),
                category=input_data.target_queries[0] if input_data.target_queries else "object",
                confidence=0.94,
            )
        ]
        output = DetectionToolOutput(detections=detections, total_found=len(detections))
        uncertainty = UncertaintyEstimate(
            confidence=0.94,
            entropy=0.09,
            conformal_interval=(0.89, 0.99),
            calibration_method="conformal_prediction",
        )
        return output, uncertainty


# -------------------------------------------------------------------------
# 3. Execution Pipeline
# -------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print(" OmniVision Agent: Phase 1 Foundations Demonstration")
    print("=" * 80)

    tracker = get_global_tracker()
    tracker.reset()

    # Step 1: Prompt Registry & LLM Factory
    print("\n[1/5] Prompt Registry & LLM Factory Invocation...")
    prompt_reg = get_prompt_registry()
    rendered_prompt = prompt_reg.render(
        "query_analyst",
        query="Count all red sedans parked in front of the building",
        modality="image",
    )
    print(f"Rendered Prompt:\n{rendered_prompt}\n")

    class QueryIntentAnalysis(BaseModel):
        intent: str
        target: str
        attributes: List[str]
        spatial_scope: str

    llm = LLMFactory.create(
        provider="mock",
        model="omnivision-mock-v1",
        mock_structured_data={
            "intent": "count",
            "target": "sedan",
            "attributes": ["red", "parked"],
            "spatial_scope": "in front of the building",
        },
    )
    llm_resp = llm.generate(
        [{"role": "user", "content": rendered_prompt}],
        response_model=QueryIntentAnalysis,
    )
    print(f"LLM Response (Structured Data): {llm_resp.structured_data}")
    print(f"Latency: {llm_resp.latency_ms:.2f} ms | Tokens: {llm_resp.total_tokens}")

    # Step 2: Resilient JSON Repair Fallback
    print("\n[2/5] Testing JSON Repair Fallback Engine...")
    malformed_json = "```json\n{query_type: 'spatial_relation', 'target': 'car', 'valid': True, }\n```"
    repaired_dict = parse_json_safely(malformed_json)
    print(f"Input malformed: {repr(malformed_json)}")
    print(f"Repaired output: {repaired_dict}")

    # Step 3: Model Registry & GPU Memory Manager with LRU Offloader
    print("\n[3/5] Testing Model Registry & Memory Manager with LRU Offload...")
    device = auto_select_device()
    print(f"Auto-selected compute device: {device}")
    mem_mgr = GPUMemoryManager(target_device=device, max_active_models=2)
    cache = ContentHashCache(cache_dir=".cache/demo", l1_max_items=50)
    model_reg = ModelRegistry(memory_manager=mem_mgr, cache=cache)

    # Register two models
    model_reg.register(
        ModelSpec(
            model_id="encoder_primary",
            loader=lambda: SampleVisionEncoder(in_channels=3, feature_dim=64),
            precision="fp32",
            description="Primary visual feature backbone",
        )
    )
    model_reg.register(
        ModelSpec(
            model_id="encoder_secondary",
            loader=lambda: SampleVisionEncoder(in_channels=3, feature_dim=32),
            precision="fp32",
            description="Secondary auxiliary backbone",
        )
    )
    model_reg.register(
        ModelSpec(
            model_id="encoder_tertiary",
            loader=lambda: SampleVisionEncoder(in_channels=3, feature_dim=16),
            precision="fp32",
            description="Tertiary lightweight backbone",
        )
    )

    # Lazy load models 1 and 2
    m1 = model_reg.get("encoder_primary")
    m2 = model_reg.get("encoder_secondary")
    print(f"Active models on GPU: {mem_mgr.get_status()['active_models_gpu']}")

    # Loading 3rd model triggers LRU offload of model 1
    m3 = model_reg.get("encoder_tertiary")
    print(f"State after loading 3rd model (max_active=2):")
    print(f"  Active on GPU: {mem_mgr.get_status()['active_models_gpu']}")
    print(f"  Offloaded to CPU: {mem_mgr.get_status()['offloaded_models_cpu']}")

    # Step 4: Typed Vision Tool Execution & Content Cache
    print("\n[4/5] Testing Typed Vision Tool & Observation Caching...")
    tool_meta = ToolMetadata(
        name="zero_shot_detector",
        description="Detects candidate instances by text queries",
        estimated_cost_usd=0.005,
        estimated_latency_ms=25.0,
    )
    tool = DemoDetectionTool(metadata=tool_meta, cache=cache, tracker=tracker)

    req_payload = {
        "image_uri": "s3://dataset/sample_image.png",
        "target_queries": ["red sedan"],
        "threshold": 0.5,
    }
    result1 = tool.execute(req_payload)
    print(f"First Execution (cold):")
    print(f"  Observation ID: {result1.observation_id}")
    print(f"  Success: {result1.success} | Latency: {result1.latency_ms:.2f} ms")
    print(f"  Uncertainty: {result1.uncertainty.model_dump()}")
    print(f"  Provenance: {result1.provenance}")

    result2 = tool.execute(req_payload)
    print(f"Second Execution (cached):")
    print(f"  Observation ID: {result2.observation_id}")
    print(f"  Latency: {result2.latency_ms:.2f} ms | Provenance: {result2.provenance}")

    # Step 5: System Telemetry & Cost Ledger
    print("\n[5/5] Telemetry Ledger Summary...")
    summary = tracker.get_summary()
    print(f"Total Calls: {summary.total_calls}")
    print(f"Total Cost USD: ${summary.total_cost_usd:.6f}")
    print(f"Latency p50: {summary.latency_p50_ms:.2f} ms | p95: {summary.latency_p95_ms:.2f} ms")
    print("Breakdown by component:")
    pprint.pprint(summary.by_component)

    print("\n" + "=" * 80)
    print(" Phase 1 Foundations Successfully Verified!")
    print("=" * 80)


if __name__ == "__main__":
    main()
