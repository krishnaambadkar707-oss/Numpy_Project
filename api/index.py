import os
import sys

# Ensure both api/, project root, and task directory are in sys.path
api_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(api_dir)
for p in [api_dir, root_dir, os.getcwd(), "/var/task", "/var/task/api"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Set writable cache directories for serverless environment (/tmp)
os.environ.setdefault("HF_HOME", "/tmp/cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/cache/transformers")
os.environ.setdefault("TORCH_HOME", "/tmp/cache/torch")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import Request
from backend.main import app

# Normalization middleware so any request reaching the serverless function matches
# the exact FastAPI route, even if Vercel forwarded the path via proxy headers
@app.middleware("http")
async def resolve_real_path_middleware(request: Request, call_next):
    current_path = request.scope.get("path", "")

    # Only intervene if the current path is just root or the serverless function script name
    if current_path in ("/", "", "/api", "/api/", "/api/index.py", "index.py"):
        fwd_uri = (
            request.headers.get("x-forwarded-uri")
            or request.headers.get("x-original-uri")
            or request.headers.get("x-rewrite-url")
        )
        if fwd_uri and "?" in fwd_uri:
            fwd_uri = fwd_uri.split("?", 1)[0]

        matched = request.headers.get("x-matched-path") or request.headers.get("x-vercel-matched-path")
        if matched and "?" in matched:
            matched = matched.split("?", 1)[0]
        if matched in ("/", "/api", "/api/", "/api/index.py", "index.py"):
            matched = None

        candidate = fwd_uri or matched
        if candidate and candidate not in ("/", "/api", "/api/", "/api/index.py", "index.py"):
            if candidate.endswith("/index.py"):
                candidate = candidate[:-9] or "/"
            request.scope["path"] = candidate

    return await call_next(request)

# Catch-all root endpoints supporting all HTTP methods for health/root checks
@app.api_route("/", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
@app.api_route("/api", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
@app.api_route("/api/index.py", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
def root_catchall(request: Request):
    return {"status": "ok", "service": "Audio Signal Processor API", "version": "1.0.0"}

# Entrypoint for Vercel Serverless
app = app
