import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image
import numpy as np


def test_storage_init():
    import config
    from app import storage
    storage.init_db()
    assert config.DB_PATH.exists()


def test_base64_roundtrip():
    from app import face_engine
    img = Image.new("RGB", (100, 100), color="red")
    b64 = face_engine.image_to_base64(img)
    assert len(b64) > 0
    img2 = face_engine.base64_to_image(b64)
    assert img2.size == (100, 100)
