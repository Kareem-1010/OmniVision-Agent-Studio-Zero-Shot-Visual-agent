import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.service import app, create_app

__all__ = ["app", "create_app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.service:app", host="0.0.0.0", port=8000, reload=True)
