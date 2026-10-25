"""
Performance Optimization, Model Export, Quantization, and Profiling Engine.
Implements:
1. ONNX Model Export and ONNX Runtime inference verification.
2. PyTorch Post-Training Dynamic INT8 Quantization.
3. Micro-Batch Dynamic Batching Queue with timeout-based dispatch.
4. Asynchronous Concurrent Tool Dispatcher.
5. Performance Profiler with P50/P90/P99 latency percentiles and bottleneck diagnostics.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar, Union
import numpy as np
from pydantic import BaseModel, Field
import torch
import torch.nn as nn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger("OmniVision.Performance.Optimization")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# 1. ONNX Exporter & Runtime Verification
# ---------------------------------------------------------------------------

class ONNXExporter:
    """Exports PyTorch vision models to optimized ONNX computational graphs."""

    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 17,
    ) -> str:
        """
        Export a PyTorch module to ONNX format with graph validation.
        """
        import onnx
        import onnxruntime as ort

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        in_names = input_names or ["input_tensor"]
        out_names = output_names or ["output_tensor"]
        dyn_axes = dynamic_axes or {
            in_names[0]: {0: "batch_size"},
            out_names[0]: {0: "batch_size"},
        }

        model.eval()
        torch.onnx.export(
            model,
            sample_input,
            str(out_file),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=in_names,
            output_names=out_names,
            dynamic_axes=dyn_axes,
        )

        # Validate ONNX graph structure
        onnx_model = onnx.load(str(out_file))
        onnx.checker.check_model(onnx_model)

        # Validate with test inference via ONNX Runtime
        session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
        ort_inputs = {in_names[0]: sample_input.detach().cpu().numpy()}
        ort_outputs = session.run(None, ort_inputs)

        logger.info(f"Successfully exported and verified ONNX model: {output_path} (Output shape: {ort_outputs[0].shape})")
        return str(out_file.resolve())


# ---------------------------------------------------------------------------
# 2. INT8 Dynamic Quantization
# ---------------------------------------------------------------------------

class QuantizationResult(BaseModel):
    """Model compression and speedup metrics post-quantization."""
    original_size_bytes: int
    quantized_size_bytes: int
    compression_ratio: float
    original_latency_ms: float
    quantized_latency_ms: float
    speedup_factor: float


class ModelQuantizer:
    """Post-Training Dynamic INT8 Quantization Engine."""

    @staticmethod
    def quantize_dynamic_int8(
        model: nn.Module,
        qconfig_layers: Optional[Tuple[Type[nn.Module], ...]] = None,
    ) -> Tuple[nn.Module, QuantizationResult]:
        """
        Apply PyTorch dynamic INT8 quantization to linear/conv modules.
        Returns quantized module and performance comparison metrics.
        """
        layers = qconfig_layers or (nn.Linear,)

        # 1. Benchmark unquantized model
        model.eval()
        sample = torch.randn(8, next(model.parameters()).shape[-1])
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                _ = model(sample)
        orig_lat = ((time.perf_counter() - t0) / 50.0) * 1000.0

        # Measure original size
        buf_orig = io.BytesIO()
        torch.save(model.state_dict(), buf_orig)
        orig_size = buf_orig.tell()

        # 2. Apply quantization
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            qconfig_spec=set(layers),
            dtype=torch.qint8,
        )
        quantized_model.eval()

        # 3. Benchmark quantized model
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                _ = quantized_model(sample)
        quant_lat = ((time.perf_counter() - t0) / 50.0) * 1000.0

        # Measure quantized size
        buf_quant = io.BytesIO()
        torch.save(quantized_model.state_dict(), buf_quant)
        quant_size = buf_quant.tell()

        comp_ratio = orig_size / max(1, quant_size)
        speedup = orig_lat / max(0.001, quant_lat)

        metrics = QuantizationResult(
            original_size_bytes=orig_size,
            quantized_size_bytes=quant_size,
            compression_ratio=round(comp_ratio, 2),
            original_latency_ms=round(orig_lat, 3),
            quantized_latency_ms=round(quant_lat, 3),
            speedup_factor=round(speedup, 2),
        )
        logger.info(f"INT8 Quantization: Compression {comp_ratio:.1f}x, Latency {orig_lat:.2f}ms -> {quant_lat:.2f}ms")
        return quantized_model, metrics


# ---------------------------------------------------------------------------
# 3. Micro-Batch Dynamic Batching Queue
# ---------------------------------------------------------------------------

class DynamicBatcher:
    """
    Micro-Batch Aggregator for high-throughput perception serving.
    Gathers asynchronous single-item inference requests into dynamic micro-batches.
    """

    def __init__(
        self,
        batch_fn: Callable[[List[Any]], Any],
        max_batch_size: int = 16,
        timeout_ms: float = 8.0,
    ) -> None:
        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self.timeout_sec = timeout_ms / 1000.0
        self._queue: List[Tuple[Any, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._dispatch_task: Optional[asyncio.Task] = None

    async def enqueue(self, item: Any) -> Any:
        """Enqueue an item and await its batched execution result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        async with self._lock:
            self._queue.append((item, future))
            if len(self._queue) >= self.max_batch_size:
                # Trigger immediate dispatch
                self._flush_batch()
            elif self._dispatch_task is None or self._dispatch_task.done():
                self._dispatch_task = asyncio.create_task(self._scheduled_flush())

        return await future

    async def _scheduled_flush(self) -> None:
        """Wait for timeout window then flush whatever is buffered."""
        await asyncio.sleep(self.timeout_sec)
        async with self._lock:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Process buffered queue items as a single batch."""
        if not self._queue:
            return

        batch_items = [item for item, _ in self._queue]
        batch_futures = [fut for _, fut in self._queue]
        self._queue = []

        try:
            results = self.batch_fn(batch_items)
            for fut, res in zip(batch_futures, results):
                if not fut.done():
                    fut.set_result(res)
        except Exception as e:
            for fut in batch_futures:
                if not fut.done():
                    fut.set_exception(e)


# ---------------------------------------------------------------------------
# 4. Asynchronous Concurrent Tool Dispatcher
# ---------------------------------------------------------------------------

class AsyncToolDispatcher:
    """Executes independent perception tools concurrently using asyncio."""

    def __init__(self, sync_dispatcher: Any) -> None:
        self.dispatcher = sync_dispatcher

    async def execute_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Run tool in worker thread pool without blocking event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.dispatcher.execute, tool_name, arguments)

    async def execute_parallel(self, calls: List[Tuple[str, Dict[str, Any]]]) -> List[Any]:
        """Execute multiple vision tools simultaneously."""
        tasks = [self.execute_tool_async(t_name, t_args) for t_name, t_args in calls]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# 5. Performance Profiler & Bottleneck Analysis
# ---------------------------------------------------------------------------

class ComponentLatency(BaseModel):
    """Latency distribution for a single architectural stage."""
    component: str
    sample_count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    percentage_of_total: float


class ProfilingReport(BaseModel):
    """Comprehensive system profiling report with bottleneck diagnostic."""
    total_calls: int
    total_duration_ms: float
    components: List[ComponentLatency]
    primary_bottleneck: str
    recommendations: List[str]


class PerformanceProfiler:
    """Telemetry collector and latency distribution analyzer."""

    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = {}

    def record(self, component: str, latency_ms: float) -> None:
        """Record an execution timing sample."""
        self._timings.setdefault(component, []).append(float(latency_ms))

    def generate_report(self) -> ProfilingReport:
        """Calculate percentiles and generate bottleneck analysis."""
        total_time = sum(sum(v) for v in self._timings.values())
        components_list: List[ComponentLatency] = []

        for comp, vals in self._timings.items():
            if not vals:
                continue
            arr = np.array(vals)
            c_sum = float(arr.sum())
            pct = (c_sum / max(0.001, total_time)) * 100.0

            components_list.append(
                ComponentLatency(
                    component=comp,
                    sample_count=len(vals),
                    mean_ms=round(float(arr.mean()), 2),
                    p50_ms=round(float(np.percentile(arr, 50)), 2),
                    p90_ms=round(float(np.percentile(arr, 90)), 2),
                    p99_ms=round(float(np.percentile(arr, 99)), 2),
                    percentage_of_total=round(pct, 1),
                )
            )

        # Sort descending by total time
        components_list.sort(key=lambda x: x.percentage_of_total, reverse=True)
        primary = components_list[0].component if components_list else "None"

        # Automated optimization recommendations
        recs: List[str] = []
        if primary in ["detector_ensemble", "segmentation", "metric_depth"]:
            recs.append(f"Export {primary} to ONNX and enable INT8 quantization.")
            recs.append("Enable Dynamic Batching with max_batch_size=8.")
        elif primary in ["Planner", "Synthesizer"]:
            recs.append("Enable KV-cache reuse in LLM inference engine.")
            recs.append("Quantize planner LLM with AWQ or FP8 weights.")
        else:
            recs.append("Pipeline execution is balanced; scale parallel tool workers.")

        return ProfilingReport(
            total_calls=sum(len(v) for v in self._timings.values()),
            total_duration_ms=round(total_time, 2),
            components=components_list,
            primary_bottleneck=primary,
            recommendations=recs,
        )
