"""
Unified LLM Client Factory and Inference Layer.
Supports OpenAI, Anthropic, Groq, vLLM/Ollama, and Mock providers with:
- Native function calling & structured output schemas
- Constrained JSON decoding with automated JSON repair fallback
- Built-in exponential backoff retry with jitter
- Token, latency, and cost telemetry
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar
import httpx
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.utils.json_repair import parse_json_safely
from src.utils.tracker import MetricsTracker, get_global_tracker

logger = logging.getLogger("OmniVision.LLMFactory")
T = TypeVar("T", bound=BaseModel)


class ToolDefinition(BaseModel):
    """Specification of an executable function/tool passed to an LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema specification


class FunctionCall(BaseModel):
    """Normalized function call request emitted by an LLM."""
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = ""


class LLMResponse(BaseModel):
    """Standardized multi-agent LLM response container."""
    content: str = ""
    function_calls: List[FunctionCall] = Field(default_factory=list)
    structured_data: Optional[Dict[str, Any]] = None
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str = "stop"


class BaseLLMClient(ABC):
    """Abstract base class governing all LLM client implementations."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        enable_json_repair: bool = True,
        tracker: Optional[MetricsTracker] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.enable_json_repair = enable_json_repair
        self.tracker = tracker or get_global_tracker()

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[ToolDefinition]] = None,
        response_model: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronously execute inference."""
        pass

    def _parse_constrained_json(
        self,
        raw_text: str,
        response_model: Optional[Type[T]],
    ) -> Optional[Dict[str, Any]]:
        """Parse text into validated structured data with fallback repair."""
        if not response_model:
            return None

        # Attempt safe parse
        parsed_dict = parse_json_safely(raw_text, default=None)
        if parsed_dict is not None and isinstance(parsed_dict, dict):
            try:
                # Validate against target Pydantic model
                validated = response_model.model_validate(parsed_dict)
                return validated.model_dump()
            except Exception as e:
                logger.warning(f"Schema validation warning for {response_model.__name__}: {e}")
                return parsed_dict
        return None


class OpenAIClient(BaseLLMClient):
    """Direct HTTP client for OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[ToolDefinition]] = None,
        response_model: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        call_id = f"oai_{uuid.uuid4().hex[:10]}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        if response_model:
            payload["response_format"] = {"type": "json_object"}

        start_time = time.perf_counter()
        response_data: Dict[str, Any] = {}
        error_type: Optional[str] = None

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _invoke() -> httpx.Response:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                res = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                return res

        try:
            resp = _invoke()
            response_data = resp.json()
        except Exception as e:
            error_type = type(e).__name__
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.tracker.record_call(
                call_id=call_id,
                component="llm:openai",
                model_or_tool=self.model,
                latency_ms=latency_ms,
                success=False,
                error_type=error_type,
            )
            raise e

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        choice = response_data["choices"][0]
        message = choice.get("message", {})
        raw_content = message.get("content") or ""

        function_calls: List[FunctionCall] = []
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                parsed_args = parse_json_safely(raw_args, default={})
                function_calls.append(
                    FunctionCall(
                        call_id=tc.get("id", f"call_{uuid.uuid4().hex[:6]}"),
                        name=fn_name,
                        arguments=parsed_args if isinstance(parsed_args, dict) else {},
                        raw_arguments=raw_args,
                    )
                )

        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        structured = self._parse_constrained_json(raw_content, response_model)

        metric = self.tracker.record_call(
            call_id=call_id,
            component="llm:openai",
            model_or_tool=self.model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=True,
        )

        return LLMResponse(
            content=raw_content,
            function_calls=function_calls,
            structured_data=structured,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            cost_usd=metric.cost_usd,
            finish_reason=choice.get("finish_reason", "stop"),
        )


class AnthropicClient(BaseLLMClient):
    """Direct HTTP client for Anthropic Messages API."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = (base_url or "https://api.anthropic.com/v1").rstrip("/")

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[ToolDefinition]] = None,
        response_model: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        call_id = f"ant_{uuid.uuid4().hex[:10]}"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Separate system message if provided
        system_prompt = ""
        anthropic_msgs = []
        for m in messages:
            if m.get("role") == "system":
                system_prompt = m.get("content", "")
            else:
                anthropic_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_msgs,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        start_time = time.perf_counter()
        error_type: Optional[str] = None

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _invoke() -> httpx.Response:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                res = client.post(f"{self.base_url}/messages", headers=headers, json=payload)
                res.raise_for_status()
                return res

        try:
            resp = _invoke()
            response_data = resp.json()
        except Exception as e:
            error_type = type(e).__name__
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.tracker.record_call(
                call_id=call_id,
                component="llm:anthropic",
                model_or_tool=self.model,
                latency_ms=latency_ms,
                success=False,
                error_type=error_type,
            )
            raise e

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        content_blocks = response_data.get("content", [])
        text_content = ""
        function_calls: List[FunctionCall] = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                function_calls.append(
                    FunctionCall(
                        call_id=block.get("id", f"call_{uuid.uuid4().hex[:6]}"),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                        raw_arguments=json.dumps(block.get("input", {})),
                    )
                )

        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        structured = self._parse_constrained_json(text_content, response_model)

        metric = self.tracker.record_call(
            call_id=call_id,
            component="llm:anthropic",
            model_or_tool=self.model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=True,
        )

        return LLMResponse(
            content=text_content,
            function_calls=function_calls,
            structured_data=structured,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            cost_usd=metric.cost_usd,
            finish_reason=response_data.get("stop_reason", "stop"),
        )


class GroqClient(OpenAIClient):
    """Client for Groq fast LPU inference, compatible with OpenAI protocol."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.getenv("GROQ_API_KEY", "")
        chosen_model = model or os.getenv("OMNIVISION_LLM_MODEL", "openai/gpt-oss-120b")
        super().__init__(
            model=chosen_model,
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )


class VLLMClient(OpenAIClient):
    """Client for locally hosted vLLM or Ollama OpenAI-compatible endpoints."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        super().__init__(
            model=model,
            api_key="EMPTY",
            base_url=url,
            **kwargs,
        )


class MockLLMClient(BaseLLMClient):
    """Deterministic, mock LLM client for unit tests and local reproducible evaluation."""

    def __init__(
        self,
        model: str = "omnivision-mock-v1",
        default_content: str = "Mock response from OmniVision Agent.",
        mock_function_calls: Optional[List[FunctionCall]] = None,
        mock_structured_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.default_content = default_content
        self.mock_function_calls = mock_function_calls or []
        self.mock_structured_data = mock_structured_data

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[ToolDefinition]] = None,
        response_model: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        call_id = f"mock_{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        structured = self.mock_structured_data
        content = self.default_content

        if response_model and not structured:
            # Generate minimal valid payload compliant with response_model if not provided
            sample_data: Dict[str, Any] = {}
            for field_name, field_info in response_model.model_fields.items():
                annotation = field_info.annotation
                if annotation in (str, Optional[str]):
                    sample_data[field_name] = "mock_value"
                elif annotation in (int, Optional[int]):
                    sample_data[field_name] = 1
                elif annotation in (float, Optional[float]):
                    sample_data[field_name] = 0.95
                elif annotation in (bool, Optional[bool]):
                    sample_data[field_name] = True
                elif annotation in (list, List[str]):
                    sample_data[field_name] = ["mock_item"]
                else:
                    sample_data[field_name] = None
            structured = sample_data
            content = json.dumps(sample_data)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages) * 2
        completion_tokens = len(content.split()) * 2

        self.tracker.record_call(
            call_id=call_id,
            component="llm:mock",
            model_or_tool=self.model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=True,
        )

        return LLMResponse(
            content=content,
            function_calls=self.mock_function_calls,
            structured_data=structured,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,
            finish_reason="stop",
        )


class LLMFactory:
    """Factory creating configured LLM provider instances."""

    _PROVIDERS: Dict[str, Type[BaseLLMClient]] = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "groq": GroqClient,
        "vllm": VLLMClient,
        "ollama": VLLMClient,
        "mock": MockLLMClient,
    }

    @classmethod
    def create(
        cls,
        provider: str = "mock",
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseLLMClient:
        """Instantiate an LLM client by provider name."""
        provider_key = provider.lower().strip()
        if provider_key not in cls._PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider '{provider}'. Supported providers: {list(cls._PROVIDERS.keys())}"
            )
        client_cls = cls._PROVIDERS[provider_key]
        if model:
            kwargs["model"] = model
        return client_cls(**kwargs)
