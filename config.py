import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MUGSHOTS_DIR = DATA_DIR / "mugshots"
DB_PATH = DATA_DIR / "faces.db"

MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5"

FACE_MATCH_TOLERANCE = 0.45
FACE_ENCODINGS_PER_PERSON = 5
FACE_MODEL = os.getenv("FACE_MODEL", "buffalo_s")
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "320"))

# Camera settings
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
MJPEG_QUALITY = int(os.getenv("MJPEG_QUALITY", "70"))
