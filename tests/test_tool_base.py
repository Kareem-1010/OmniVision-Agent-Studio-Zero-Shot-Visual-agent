"""
Unit tests for Typed Vision Tool Base Class.
"""

from typing import List, Tuple, Type
from pydantic import BaseModel, Field

from src.models.cache import ContentHashCache
from src.tools.base import (
    BaseVisionTool,
    ToolMetadata,
    UncertaintyEstimate,
    ToolResult,
)
from src.utils.tracker import MetricsTracker


class MockInput(BaseModel):
    image_id: str
    target_class: str
    confidence_threshold: float = 0.5


class DetectionItem(BaseModel):
    bbox: List[float]
    label: str
    score: float


class MockOutput(BaseModel):
    detections: List[DetectionItem]
    count: int


class MockDetectorTool(BaseVisionTool[MockInput, MockOutput]):
    @property
    def input_schema(self) -> Type[MockInput]:
        return MockInput

    @property
    def output_schema(self) -> Type[MockOutput]:
        return MockOutput

    def _run(self, input_data: MockInput) -> Tuple[MockOutput, UncertaintyEstimate]:
        detections = [
            DetectionItem(bbox=[10.0, 20.0, 50.0, 80.0], label=input_data.target_class, score=0.92)
        ]
        output = MockOutput(detections=detections, count=len(detections))
        uncertainty = UncertaintyEstimate(
            confidence=0.92,
            entropy=0.12,
            conformal_interval=(0.85, 0.98),
            calibration_method="conformal_prediction",
        )
        return output, uncertainty


def test_typed_tool_execution_and_observation_id(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    meta = ToolMetadata(
        name="mock_detector",
        description="Detects mock targets in synthetic visual tests",
        estimated_cost_usd=0.002,
        estimated_latency_ms=15.0,
    )
    tool = MockDetectorTool(metadata=meta, cache=isolated_cache, tracker=isolated_tracker)

    payload = {"image_id": "frame_001.png", "target_class": "person", "confidence_threshold": 0.5}
    result: ToolResult[MockOutput] = tool.execute(payload)

    assert result.success is True
    assert result.observation_id.startswith("obs_")
    assert result.data is not None
    assert result.data.count == 1
    assert result.data.detections[0].label == "person"
    assert result.uncertainty.confidence == 0.92
    assert result.uncertainty.conformal_interval == (0.85, 0.98)
    assert result.provenance["cached"] is False

    # Second execution with identical payload should be served from cache
    cached_result: ToolResult[MockOutput] = tool.execute(payload)
    assert cached_result.success is True
    assert cached_result.provenance["cached"] is True
    assert cached_result.data.count == 1


def test_typed_tool_validation_error(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    meta = ToolMetadata(
        name="mock_detector",
        description="Detects mock targets",
    )
    tool = MockDetectorTool(metadata=meta, cache=isolated_cache, tracker=isolated_tracker)

    # Missing required target_class field
    invalid_payload = {"image_id": "frame_001.png"}
    result = tool.execute(invalid_payload)

    assert result.success is False
    assert result.data is None
    assert "InputValidationException" in result.error_message
