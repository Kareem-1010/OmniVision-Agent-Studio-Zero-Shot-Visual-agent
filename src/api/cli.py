"""
Command-Line Interface (CLI) for OmniVision Agent.
Commands:
- `serve`: Launches FastAPI REST and WebSocket server.
- `run`: Executes a visual query from the terminal.
- `optimize`: Runs ONNX export and INT8 quantization benchmarks.
- `ui`: Launches interactive Gradio dashboard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import click
import torch

from src.agent.orchestrator import OmniVisionAgent
from src.models.optimization import ModelQuantizer, ONNXExporter, PerformanceProfiler
from src.presets import get_preset_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OmniVision.CLI")


@click.group()
@click.version_option(version="3.0.0")
def main():
    """OmniVision Agent: Research-Grade Multi-Agent Vision Reasoning CLI."""
    pass


@main.command()
@click.option("--host", default="127.0.0.1", help="Server host binding address.")
@click.option("--port", default=8000, type=int, help="Server port number.")
@click.option("--reload", is_flag=True, help="Enable auto-reload on code change.")
def serve(host: str, port: int, reload: bool):
    """Launch the FastAPI REST and WebSocket API service."""
    import uvicorn
    click.echo(f"Starting OmniVision API server at http://{host}:{port} ...")
    uvicorn.run("src.api.service:app", host=host, port=port, reload=reload)


@main.command()
@click.option("--query", "-q", required=True, help="User question or reasoning goal.")
@click.option("--media", "-m", required=True, help="Path or URI to input image or video file.")
@click.option("--media-type", default="image", type=click.Choice(["image", "video"]), help="Media modality.")
@click.option("--preset", "-p", default=None, help="Optional domain preset: retail, accessibility, hazard_safety.")
@click.option("--max-iter", default=8, type=int, help="Maximum agent planning iterations.")
@click.option("--budget", default=0.40, type=float, help="Execution budget limit in USD.")
def run(query: str, media: str, media_type: str, preset: str, max_iter: int, budget: float):
    """Execute a vision reasoning query and print grounded output."""
    click.echo(f"Query: '{query}' | Media: '{media}'")

    if preset:
        registry = get_preset_registry()
        p = registry.get(preset)
        click.echo(f"Executing Domain Preset: '{preset}'...")
        res = p.analyze(media_uri=media, query=query)
        click.echo("\n--- Preset Output ---")
        click.echo(json.dumps(res.model_dump(), indent=2))
        return

    agent = OmniVisionAgent(
        max_iterations=max_iter,
        budget_limit_usd=budget,
        enable_tree_search=True,
        enable_active_perception=True,
    )

    synthesis = agent.run(query=query, media_uri=media, media_type=media_type)  # type: ignore

    click.echo("\n" + "=" * 60)
    click.echo(" GROUNDED SYNTHESIS OUTPUT")
    click.echo("=" * 60)
    click.echo(f"Answer: {synthesis.answer}")
    click.echo(f"Calibrated Confidence: {synthesis.calibrated_confidence:.3f} (Interval: {synthesis.confidence_interval})")
    click.echo(f"Citations: {synthesis.citations}")
    click.echo(f"Cost: ${synthesis.total_cost_usd:.4f} | Latency: {synthesis.total_latency_ms:.1f}ms")
    click.echo("Key Findings:")
    for f in synthesis.key_findings:
        click.echo(f"  * {f}")


@main.command()
@click.option("--output-dir", default="artifacts/onnx", help="Directory to save exported ONNX models.")
def optimize(output_dir: str):
    """Run model export to ONNX and perform dynamic INT8 quantization benchmarking."""
    click.echo("Running Performance Optimization Suite...")

    # Sample lightweight detector head module
    class SimplePerceptionModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(128, 64)
            self.act = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(64, 4)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    model = SimplePerceptionModule()
    sample_input = torch.randn(1, 128)

    # 1. ONNX Export
    out_onnx = str(Path(output_dir) / "perception_head.onnx")
    click.echo(f"Exporting module to ONNX: {out_onnx}...")
    saved_onnx = ONNXExporter.export_to_onnx(model, sample_input, out_onnx)
    click.echo(f"ONNX Export Verified: {saved_onnx}")

    # 2. INT8 Quantization
    click.echo("Running INT8 Dynamic Quantization benchmark...")
    quant_model, q_metrics = ModelQuantizer.quantize_dynamic_int8(model)
    click.echo(f"Compression Ratio : {q_metrics.compression_ratio}x")
    click.echo(f"Original Latency  : {q_metrics.original_latency_ms:.3f} ms")
    click.echo(f"Quantized Latency : {q_metrics.quantized_latency_ms:.3f} ms")
    click.echo(f"Speedup Factor    : {q_metrics.speedup_factor}x")


@main.command()
@click.option("--port", default=7860, type=int, help="Gradio server port.")
@click.option("--share", is_flag=True, help="Create public Gradio link.")
def ui(port: int, share: bool):
    """Launch the interactive Gradio Web Dashboard."""
    from src.ui.app import build_demo
    click.echo(f"Launching OmniVision Gradio Dashboard on port {port}...")
    demo = build_demo()
    demo.launch(server_port=port, share=share)


# CLI entry point alias
cli = main

if __name__ == "__main__":
    main()
