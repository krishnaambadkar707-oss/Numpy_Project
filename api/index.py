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
# the exact FastAPI route, regardless of whether Vercel passed the path via
# x-matched-path header, query rewrite parameter (?path=...), or direct path.
@app.middleware("http")
async def resolve_real_path_middleware(request: Request, call_next):
    # 1. Check Vercel proxy headers
    matched_path = (
        request.headers.get("x-matched-path")
        or request.headers.get("x-vercel-matched-path")
        or request.headers.get("x-forwarded-uri")
    )
    if matched_path and "?" in matched_path:
        matched_path = matched_path.split("?", 1)[0]

    # 2. Check query parameter from vercel.json rewrite
    query_path = request.query_params.get("path")
    if query_path:
        if not query_path.startswith("/"):
            query_path = "/" + query_path
        if not query_path.startswith("/api/"):
            query_path = "/api" + query_path

    # Determine candidate path
    candidate = matched_path or query_path or request.scope.get("path", "")

    # Strip serverless file artifacts like index.py
    if candidate.endswith("/index.py"):
        candidate = candidate[:-9] or "/"
    elif candidate == "index.py":
        candidate = "/"

    if candidate and candidate not in ("/", ""):
        if not candidate.startswith("/api"):
            candidate = "/api" + candidate
        request.scope["path"] = candidate

    return await call_next(request)

# Catch-all root endpoints supporting all HTTP methods to prevent 405 Method Not Allowed
@app.api_route("/", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
@app.api_route("/api", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
@app.api_route("/api/index.py", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
def root_catchall(request: Request):
    return {"status": "ok", "service": "Audio Signal Processor API", "version": "1.0.0"}

# Entrypoint for Vercel Serverless
app = app
