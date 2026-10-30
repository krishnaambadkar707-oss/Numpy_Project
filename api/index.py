import os
import sys

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Set writable cache directories for serverless environment (/tmp)
os.environ.setdefault("HF_HOME", "/tmp/cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/cache/transformers")
os.environ.setdefault("TORCH_HOME", "/tmp/cache/torch")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi.staticfiles import StaticFiles
from backend.main import app

# Mount static frontend so that the HTML/CSS/JS dashboard is served at root /
frontend_dir = os.path.join(root_dir, "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

# Entrypoint for Vercel Serverless
app = app
