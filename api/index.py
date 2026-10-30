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

# Mount static frontend so that any fallback request directly reaching the serverless
# function also receives the complete HTML/CSS/JS dashboard
for folder_name in ("public", "frontend"):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        app.mount("/", StaticFiles(directory=folder_path, html=True), name="static")
        break

# Entrypoint for Vercel Serverless
app = app
