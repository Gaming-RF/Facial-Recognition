"""
Configuration for the standalone face recognition app.
Uses sensible defaults — no .env file needed.
"""
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # project root, not app/
DATA_DIR = BASE_DIR / "data"
MUGSHOTS_DIR = DATA_DIR / "mugshots"
DB_PATH = DATA_DIR / "faces.db"

# OpenCV DNN face detection model
MODELS_DIR = BASE_DIR / "models"
PROTOTXT = MODELS_DIR / "deploy.prototxt"
CAFFEMODEL = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# ── Face recognition ────────────────────────────────────────────
FACE_MATCH_THRESHOLD = 0.6   # distance below this = match
FACE_DET_SIZE = (320, 320)
FACE_MODEL = "buffalo_s"      # InsightFace model name

# ── Camera ──────────────────────────────────────────────────────
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# ── UI ──────────────────────────────────────────────────────────
WINDOW_TITLE = "Face Recognition System"
VIDEO_LABEL_MAX_SIZE = (960, 720)
