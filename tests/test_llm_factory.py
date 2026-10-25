"""
Unit tests for LLM Factory and Client implementations.
"""

from pydantic import BaseModel
from src.utils.llm_factory import (
    LLMFactory,
    MockLLMClient,
    FunctionCall,
    ToolDefinition,
)


class SampleAnalysis(BaseModel):
    intent: str
    target: str
    confidence: float


def test_mock_llm_client_generation(isolated_tracker):
    client = MockLLMClient(
        model="omnivision-mock-v1",
        default_content="Test multi-agent reasoning message.",
        tracker=isolated_tracker,
    )
    resp = client.generate([{"role": "user", "content": "Analyze image"}])
    assert resp.content == "Test multi-agent reasoning message."
    assert resp.model == "omnivision-mock-v1"
    assert resp.total_tokens > 0

    summary = isolated_tracker.get_summary()
    assert summary.total_calls == 1


def test_mock_llm_structured_output():
    client = MockLLMClient(
        mock_structured_data={"intent": "count", "target": "pedestrians", "confidence": 0.98}
    )
    resp = client.generate(
        [{"role": "user", "content": "Count people"}],
        response_model=SampleAnalysis,
    )
    assert resp.structured_data is not None
    assert resp.structured_data["intent"] == "count"
    assert resp.structured_data["target"] == "pedestrians"
    assert resp.structured_data["confidence"] == 0.98


def test_llm_factory_instantiation():
    mock_client = LLMFactory.create(provider="mock")
    assert isinstance(mock_client, MockLLMClient)
