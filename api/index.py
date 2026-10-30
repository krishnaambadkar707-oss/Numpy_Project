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

from fastapi import Request
from backend.main import app

# Normalization middleware so any request reaching the serverless function matches
# regardless of whether Vercel passed /api/route or stripped it to /route
@app.middleware("http")
async def normalize_api_path(request: Request, call_next):
    path = request.scope.get("path", "")
    if not path.startswith("/api") and path not in ("/", ""):
        request.scope["path"] = "/api" + path
    return await call_next(request)

@app.get("/")
def api_root():
    return {"status": "ok", "service": "Audio Signal Processor API", "version": "1.0.0"}

# Entrypoint for Vercel Serverless
app = app
