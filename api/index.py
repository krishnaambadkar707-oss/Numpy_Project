import os
import sys

# Ensure both api/, project root, and task directory are in sys.path
api_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(api_dir)
for p in [api_dir, root_dir, os.getcwd(), "/var/task", "/var/task/api"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Set writable cache directories for serverless environment (/tmp)
import tempfile

is_posix = os.name != "nt"
tmp_base = "/tmp" if is_posix else os.environ.get("TEMP", os.environ.get("TMP", "C:/tmp"))

try:
    user_home = os.path.expanduser("~")
    if not os.path.exists(user_home) or not os.access(user_home, os.W_OK):
        os.environ["HOME"] = tmp_base
except Exception:
    os.environ["HOME"] = tmp_base

if is_posix and os.path.exists(tmp_base) and os.access(tmp_base, os.W_OK):
    os.environ["HOME"] = tmp_base
    os.environ["TMPDIR"] = tmp_base
    os.environ["TEMP"] = tmp_base
    os.environ["TMP"] = tmp_base
    tempfile.tempdir = tmp_base

hf_cache = os.path.join(tmp_base, "cache", "huggingface")
try:
    os.makedirs(hf_cache, exist_ok=True)
    os.makedirs(os.path.join(hf_cache, "hub"), exist_ok=True)
    os.makedirs(os.path.join(tmp_base, "whisper_models"), exist_ok=True)
except Exception:
    pass

os.environ["HF_HOME"] = hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(tmp_base, "cache", "transformers")
os.environ["TORCH_HOME"] = os.path.join(tmp_base, "cache", "torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(tmp_base, "cache")
os.environ["XDG_CONFIG_HOME"] = os.path.join(tmp_base, "config")
os.environ["XDG_DATA_HOME"] = os.path.join(tmp_base, "data")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Disable Hugging Face Xet CAS reconstruction & hf-transfer to prevent disk exhaustion (os error 28)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_DISK_SPACE_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["XET_CACHE_DIR"] = os.path.join(tmp_base, "cache", "xet")

try:
    import huggingface_hub.constants as _hf_constants
    _hf_constants.HF_HUB_DISABLE_XET = True
    _hf_constants.HF_HUB_ENABLE_HF_TRANSFER = False
except Exception:
    pass

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
