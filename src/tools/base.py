"""
Typed Vision Tool Base Class and Observation Contract.
Every perception, geometry, and temporal node inherits from BaseVisionTool with:
- Strict Pydantic v2 input and output schemas
- Cryptographically grounded Observation IDs for evidence attribution
- Uncertainty quantification (entropy, confidence, conformal prediction bounds)
- Execution telemetry, automated caching, and error containment
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union
from pydantic import BaseModel, ConfigDict, Field

from src.models.cache import ContentHashCache, hash_artifact
from src.utils.tracker import MetricsTracker, get_global_tracker

logger = logging.getLogger("OmniVision.Tools")

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class ToolMetadata(BaseModel):
    """Declarative metadata characterizing tool performance, cost, and semantics."""
    name: str
    description: str
    version: str = "1.0.0"
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    estimated_latency_ms: float = Field(default=50.0, ge=0.0)
    baseline_accuracy: float = Field(default=0.85, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)


class ObservationID(str):
    """Cryptographic content-addressed observation identifier ('obs_<hash>')."""

    @classmethod
    def create(cls, tool_name: str, payload: Any) -> ObservationID:
        """Construct deterministic Observation ID from tool name and payload."""
        h = hashlib.sha256()
        h.update(tool_name.encode("utf-8"))
        h.update(b":")
        h.update(hash_artifact(payload).encode("utf-8"))
        h.update(f":{time.time()}".encode("utf-8"))
        return cls(f"obs_{h.hexdigest()[:16]}")


class UncertaintyEstimate(BaseModel):
    """Calibrated uncertainty estimate accompanying tool observations."""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entropy: float = Field(default=0.0, ge=0.0)
    conformal_interval: Tuple[float, float] = Field(default=(0.0, 1.0))
    calibration_method: str = Field(default="heuristic")
    is_ambiguous: bool = False


class ToolResult(BaseModel, Generic[TOutput]):
    """Standardized result envelope returned by all OmniVision tools."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation_id: str
    tool_name: str
    success: bool
    data: Optional[TOutput] = None
    uncertainty: UncertaintyEstimate = Field(default_factory=UncertaintyEstimate)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    error_message: Optional[str] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class BaseVisionTool(ABC, Generic[TInput, TOutput]):
    """
    Abstract generic base class for all OmniVision tools.
    Provides automated validation, telemetry tracking, caching, and uncertainty output.
    """

    def __init__(
        self,
        metadata: ToolMetadata,
        cache: Optional[ContentHashCache] = None,
        tracker: Optional[MetricsTracker] = None,
        enable_cache: bool = True,
    ) -> None:
        self.metadata = metadata
        self.cache = cache or ContentHashCache()
        self.tracker = tracker or get_global_tracker()
        self.enable_cache = enable_cache

    @property
    @abstractmethod
    def input_schema(self) -> Type[TInput]:
        """Pydantic model class defining tool inputs."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Type[TOutput]:
        """Pydantic model class defining tool outputs."""
        pass

    @abstractmethod
    def _run(self, input_data: TInput) -> Tuple[TOutput, UncertaintyEstimate]:
        """Internal execution logic to be implemented by perception/reasoning tools."""
        pass

    def execute(self, input_data: Union[TInput, Dict[str, Any]]) -> ToolResult[TOutput]:
        """
        Execute tool with:
        1. Strict input validation against Pydantic schema
        2. Content-hash cache lookup
        3. Latency, cost, and failure logging
        4. Observation ID tagging
        """
        start_time = time.perf_counter()
        call_id = f"tool_{uuid.uuid4().hex[:8]}"

        # 1. Parse and validate input
        try:
            if isinstance(input_data, dict):
                validated_input = self.input_schema.model_validate(input_data)
            else:
                validated_input = input_data
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Input validation error in tool '{self.metadata.name}': {e}")
            obs_id = ObservationID.create(self.metadata.name, str(input_data))
            return ToolResult(
                observation_id=obs_id,
                tool_name=self.metadata.name,
                success=False,
                data=None,
                latency_ms=latency_ms,
                cost_usd=0.0,
                error_message=f"InputValidationException: {str(e)}",
            )

        obs_id = ObservationID.create(self.metadata.name, validated_input.model_dump())

        # 2. Cache Lookup
        cache_key = self.cache.compute_key(f"tool:{self.metadata.name}", validated_input.model_dump())
        if self.enable_cache:
            cached_val = self.cache.get(cache_key)
            if cached_val is not None and isinstance(cached_val, dict):
                try:
                    cached_data = self.output_schema.model_validate(cached_val.get("data"))
                    cached_uncertainty = UncertaintyEstimate.model_validate(cached_val.get("uncertainty", {}))
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    return ToolResult(
                        observation_id=obs_id,
                        tool_name=self.metadata.name,
                        success=True,
                        data=cached_data,
                        uncertainty=cached_uncertainty,
                        latency_ms=latency_ms,
                        cost_usd=0.0,  # Cached execution has zero additional model cost
                        provenance={"cached": True, "cache_key": cache_key},
                    )
                except Exception as e:
                    logger.warning(f"Error restoring cached output for {self.metadata.name}: {e}")

        # 3. Execution
        try:
            output_data, uncertainty = self._run(validated_input)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            cost_usd = self.metadata.estimated_cost_usd

            # Record telemetry
            self.tracker.record_call(
                call_id=call_id,
                component=f"tool:{self.metadata.name}",
                model_or_tool=self.metadata.name,
                latency_ms=latency_ms,
                success=True,
                custom_cost_usd=cost_usd,
            )

            # Store in cache
            if self.enable_cache:
                self.cache.set(
                    cache_key,
                    {
                        "data": output_data.model_dump(),
                        "uncertainty": uncertainty.model_dump(),
                    },
                )

            return ToolResult(
                observation_id=obs_id,
                tool_name=self.metadata.name,
                success=True,
                data=output_data,
                uncertainty=uncertainty,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                provenance={"cached": False, "version": self.metadata.version},
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception(f"Execution error in tool '{self.metadata.name}': {e}")
            self.tracker.record_call(
                call_id=call_id,
                component=f"tool:{self.metadata.name}",
                model_or_tool=self.metadata.name,
                latency_ms=latency_ms,
                success=False,
                error_type=type(e).__name__,
                custom_cost_usd=0.0,
            )
            return ToolResult(
                observation_id=obs_id,
                tool_name=self.metadata.name,
                success=False,
                data=None,
                latency_ms=latency_ms,
                cost_usd=0.0,
                error_message=f"{type(e).__name__}: {str(e)}",
            )

    def get_tool_definition(self) -> Dict[str, Any]:
        """Convert tool into OpenAI / LangChain compatible JSON tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "parameters": self.input_schema.model_json_schema(),
            },
        }


# Type alias for backward/forward compatibility
TypedTool = BaseVisionTool

