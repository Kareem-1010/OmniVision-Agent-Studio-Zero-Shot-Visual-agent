# ==============================================================================
# OmniVision Agent (Zero-Shot Visual Agent v3) - Developer & Deployment Makefile
# ==============================================================================

PYTHON ?= .venv/Scripts/python.exe
PYTEST ?= .venv/Scripts/pytest.exe

.PHONY: all setup test test-chains eval optimize serve ui docker-build docker-up docker-down clean help

help:
	@echo "OmniVision Agent Automation Commands:"
	@echo "  setup         : Install project and all dependencies into virtual environment"
	@echo "  test          : Run comprehensive test suite across all 10 phases (113+ tests)"
	@echo "  test-chains   : Run the 8 canonical multimodal visual reasoning chains"
	@echo "  eval          : Execute 300-sample benchmark suite & 10 ablation experiments"
	@echo "  optimize      : Run ONNX graph export & dynamic INT8 quantization benchmarks"
	@echo "  serve         : Start FastAPI REST API & WebSocket server on http://localhost:8000"
	@echo "  ui            : Launch Gradio 4-tab Interactive Studio on http://localhost:7860"
	@echo "  docker-build  : Build multi-stage production Docker image"
	@echo "  docker-up     : Run complete multi-service stack via docker-compose"
	@echo "  docker-down   : Tear down docker-compose containers and networks"
	@echo "  clean         : Clean bytecode, caches, temporary files, and test artifacts"

setup:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests -v --tb=short

test-chains:
	$(PYTHON) -m pytest tests/test_end_to_end_chains.py -v

eval:
	$(PYTHON) scripts/run_phase9_demo.py

optimize:
	$(PYTHON) -m src.api.cli optimize --output-dir artifacts/onnx

serve:
	$(PYTHON) -m src.api.cli serve --host 127.0.0.1 --port 8000 --reload

ui:
	$(PYTHON) -m src.api.cli ui --host 127.0.0.1 --port 7860

docker-build:
	docker build -t omnivision-agent:latest -f Dockerfile .

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	rm -rf *.egg-info build dist .cache/temp_*
