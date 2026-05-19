import os
import sys
import threading
import time
import uvicorn
import webview
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.main import app

from src.config import logger

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

def start_scheduler():
    """Background cron-like scheduler for ingestion, updates, and maintenance."""
    logger.info("Starting background scheduler...")
    while True:
        try:
            # Example schedule:
            # - ingest_finished_matches()    every 15 min
            # - refresh_live_odds()          every 5 min
            # - refresh_future_fixtures()    hourly
            # - recalibrate_models()         nightly
            
            # Placeholders for future cloud orchestration / local logic
            time.sleep(300)  # Wake up every 5 minutes
            logger.info("Scheduler heartbeat: running routine background checks.")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(60)

def get_base_path():
    # If running from PyInstaller bundle, the base path is sys._MEIPASS
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

# Mount the frontend/dist directory if it exists
frontend_path = os.path.join(get_base_path(), "frontend", "dist")
if os.path.exists(frontend_path):
    # Serve index.html at root
    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    # Mount static assets
    app.mount("/", StaticFiles(directory=frontend_path), name="static")

if __name__ == "__main__":
    # Start the FastAPI server in a separate daemon thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Start the background scheduler
    scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
    scheduler_thread.start()

    # Wait for the server to start (optional: poll the health endpoint)
    time.sleep(2)

    # Create the native desktop window pointing to the local server
    webview.create_window('Football Predictor AI', 'http://127.0.0.1:8000', width=1280, height=800)
    webview.start()
