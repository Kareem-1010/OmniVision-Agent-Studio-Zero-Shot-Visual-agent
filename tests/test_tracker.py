"""
Unit tests for Token, Latency, and Cost Tracker.
"""

from src.utils.tracker import MetricsTracker, TokenUsage


def test_token_cost_calculation():
    tokens = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    cost = tokens.compute_cost("gpt-4o")
    # GPT-4o: $5/M in, $15/M out -> (1000/1M)*5 + (500/1M)*15 = 0.005 + 0.0075 = 0.0125
    assert round(cost, 5) == 0.0125


def test_metrics_tracker_accumulation(isolated_tracker: MetricsTracker):
    isolated_tracker.record_call(
        call_id="call-1",
        component="llm:mock",
        model_or_tool="omnivision-mock-v1",
        latency_ms=120.0,
        prompt_tokens=200,
        completion_tokens=50,
        success=True,
    )
    isolated_tracker.record_call(
        call_id="call-2",
        component="tool:detector",
        model_or_tool="yolo11",
        latency_ms=45.0,
        success=True,
        custom_cost_usd=0.001,
    )

    summary = isolated_tracker.get_summary()
    assert summary.total_calls == 2
    assert summary.successful_calls == 2
    assert summary.failed_calls == 0
    assert summary.total_tokens == 250
    assert summary.total_cost_usd > 0.0
    assert "llm:mock" in summary.by_component
    assert "tool:detector" in summary.by_component
