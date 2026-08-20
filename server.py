import os
import uvicorn
from fastapi.staticfiles import StaticFiles
from backend.main import app

# Mount static files for the frontend dashboard
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    is_dev = os.environ.get("ENV", "development").lower() == "development"

    print("=========================================================")
    print("Audio Signal Processor Studio Starting...")
    print(f"Running on http://{host}:{port}")
    print("=========================================================")
    uvicorn.run("server:app", host=host, port=port, reload=is_dev)
