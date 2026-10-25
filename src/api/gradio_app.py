"""Convenience entry point for launching the OmniVision Gradio Studio."""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ui.app import demo

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    for p in range(port, port + 10):
        try:
            print(f"Attempting to launch OmniVision Studio on port {p}...")
            demo.launch(server_port=p)
            break
        except OSError as e:
            if "Cannot find empty port" in str(e):
                continue
            raise e
