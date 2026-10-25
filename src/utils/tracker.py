"""
Token, Cost, and Latency Tracking Subsystem.
Provides thread-safe metrics recording and cost accounting across LLM providers.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field


# Pricing in USD per 1,000,000 tokens (input, output)
MODEL_PRICING_PER_MILLION: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Anthropic
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    # Groq
    "qwen/qwen3.8-27b": (0.20, 0.20),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    # Local / Free
    "qwen/qwen2.5-vl-7b-instruct": (0.0, 0.0),
    "omnivision-mock-v1": (0.0, 0.0),
}


class TokenUsage(BaseModel):
    """Pydantic model representing token consumption for an inference call."""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    def compute_cost(self, model_name: str) -> float:
        """Compute cost in USD based on model pricing table."""
        key = model_name.lower().strip()
        pricing = MODEL_PRICING_PER_MILLION.get(key, (1.00, 3.00))  # conservative default
        cost_in = (self.prompt_tokens / 1_000_000.0) * pricing[0]
        cost_out = (self.completion_tokens / 1_000_000.0) * pricing[1]
        return cost_in + cost_out


class InvocationMetric(BaseModel):
    """Metadata recorded for an individual LLM or tool execution."""
    call_id: str
    component: str  # e.g., 'llm:openai', 'tool:yolo11'
    model_or_tool: str
    latency_ms: float = Field(ge=0.0)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = Field(default=0.0, ge=0.0)
    success: bool = True
    error_type: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class TrackerSummary(BaseModel):
    """Aggregated summary of system telemetry."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    by_component: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class MetricsTracker:
    """Thread-safe, observable tracker for token consumption, costs, and latencies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[InvocationMetric] = []

    def record_call(
        self,
        call_id: str,
        component: str,
        model_or_tool: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        error_type: Optional[str] = None,
        custom_cost_usd: Optional[float] = None,
    ) -> InvocationMetric:
        """Record an invocation event into the global ledger."""
        tokens = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        if custom_cost_usd is not None:
            cost = custom_cost_usd
        else:
            cost = tokens.compute_cost(model_or_tool)

        record = InvocationMetric(
            call_id=call_id,
            component=component,
            model_or_tool=model_or_tool,
            latency_ms=latency_ms,
            tokens=tokens,
            cost_usd=cost,
            success=success,
            error_type=error_type,
            timestamp=time.time(),
        )

        with self._lock:
            self._records.append(record)

        return record

    def get_summary(self) -> TrackerSummary:
        """Calculate and return aggregate performance and cost metrics."""
        with self._lock:
            records = list(self._records)

        if not records:
            return TrackerSummary()

        total_calls = len(records)
        successful_calls = sum(1 for r in records if r.success)
        failed_calls = total_calls - successful_calls
        total_prompt = sum(r.tokens.prompt_tokens for r in records)
        total_comp = sum(r.tokens.completion_tokens for r in records)
        total_tokens = total_prompt + total_comp
        total_cost = sum(r.cost_usd for r in records)

        latencies = [r.latency_ms for r in records]
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        by_component: Dict[str, Dict[str, float]] = {}
        for r in records:
            if r.component not in by_component:
                by_component[r.component] = {
                    "calls": 0,
                    "cost_usd": 0.0,
                    "tokens": 0,
                    "avg_latency_ms": 0.0,
                }
            comp_entry = by_component[r.component]
            comp_entry["calls"] += 1
            comp_entry["cost_usd"] += r.cost_usd
            comp_entry["tokens"] += r.tokens.total_tokens

        for comp, metrics in by_component.items():
            comp_lats = [r.latency_ms for r in records if r.component == comp]
            metrics["avg_latency_ms"] = float(np.mean(comp_lats)) if comp_lats else 0.0

        return TrackerSummary(
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_comp,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 6),
            latency_p50_ms=round(p50, 2),
            latency_p95_ms=round(p95, 2),
            latency_p99_ms=round(p99, 2),
            by_component=by_component,
        )

    def reset(self) -> None:
        """Clear all metrics logs."""
        with self._lock:
            self._records.clear()


_GLOBAL_TRACKER = MetricsTracker()


def get_global_tracker() -> MetricsTracker:
    """Access the singleton metrics tracker."""
    return _GLOBAL_TRACKER
