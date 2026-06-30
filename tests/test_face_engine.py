"""Tests for the facial recognition system.

These tests run without InsightFace or a camera. Heavy dependencies
(insightface, mediapipe, cv2 camera) are mocked so CI stays fast.
"""
import sys
import os
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np


# ── Storage ────────────────────────────────────────────────────────


def test_storage_init():
    import config
    from app import storage

    storage.init_db()
    assert config.DB_PATH.exists()


# ── Config ─────────────────────────────────────────────────────────


def test_face_match_threshold_default():
    """FACE_MATCH_THRESHOLD defaults to 0.6."""
    import config

    assert config.FACE_MATCH_THRESHOLD == 0.6


def test_face_match_threshold_env(monkeypatch):
    """FACE_MATCH_THRESHOLD can be overridden via env."""
    monkeypatch.setenv("FACE_MATCH_THRESHOLD", "0.75")
    # Re-import to pick up the env var
    import importlib
    import config

    monkeypatch.setenv("FACE_MATCH_THRESHOLD", "0.75")
    importlib.reload(config)
    assert config.FACE_MATCH_THRESHOLD == 0.75
    # Reset
    monkeypatch.setenv("FACE_MATCH_THRESHOLD", "0.6")
    importlib.reload(config)


# ── Face engine (mocked) ──────────────────────────────────────────


def test_dnn_detect_returns_list():
    """dnn_detect should return a list (empty if no model or frame has no faces)."""
    from app import face_engine

    # Create a blank frame (no faces)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = face_engine.dnn_detect(frame)
    assert isinstance(result, list)


def test_haar_detect_returns_list():
    """haar_detect should return a list."""
    from app import face_engine

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = face_engine.haar_detect(frame)
    assert isinstance(result, list)


def test_match_known_faces_empty():
    """_match_known_faces returns Unknown when no known encodings exist."""
    from app import face_engine

    # Ensure known encodings are empty
    face_engine._known_encodings = np.array([])
    face_engine._known_names = []

    class FakeResult:
        normed_embedding = np.random.rand(512)

    name, dist = face_engine._match_known_faces(FakeResult())
    assert name == "Unknown"
    assert dist == 1.0


def test_match_known_faces_below_threshold():
    """_match_known_faces returns Unknown when distance exceeds threshold."""
    from app import face_engine

    # Create a known encoding
    known_emb = np.random.rand(512).astype(np.float32)
    known_emb /= np.linalg.norm(known_emb)

    face_engine._known_encodings = np.array([known_emb])
    face_engine._known_names = ["Alice"]

    # Create a face result with a very different embedding (high distance)
    class FakeResult:
        normed_embedding = -known_emb  # opposite direction = high distance

    name, dist = face_engine._match_known_faces(FakeResult())
    assert name == "Unknown"
    assert dist > 0.5


def test_match_known_faces_above_threshold():
    """_match_known_faces returns the name when distance is below threshold."""
    from app import face_engine

    emb = np.random.rand(512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    face_engine._known_encodings = np.array([emb])
    face_engine._known_names = ["Alice"]

    class FakeResult:
        normed_embedding = emb.copy()

    name, dist = face_engine._match_known_faces(FakeResult())
    assert name == "Alice"
    assert dist < 0.01  # essentially zero distance


def test_match_known_faces_none_embedding():
    """_match_known_faces handles None embedding gracefully."""
    from app import face_engine

    face_engine._known_encodings = np.array([np.random.rand(512)])
    face_engine._known_names = ["Alice"]

    class FakeResult:
        normed_embedding = None

    name, dist = face_engine._match_known_faces(FakeResult())
    assert name == "Unknown"
    assert dist == 1.0


# ── FPS counter logic ─────────────────────────────────────────────


def test_fps_rolling_average():
    """Verify the rolling 30-frame FPS calculation logic."""
    frame_times = collections.deque(maxlen=30)

    # Simulate 35 frames with 33ms intervals (~30 FPS)
    dt = 1.0 / 30.0
    for i in range(35):
        frame_times.append(dt)

    avg_dt = sum(frame_times) / len(frame_times)
    fps = 1.0 / avg_dt

    assert len(frame_times) == 30  # rolling window
    assert 29.0 < fps < 31.0


def test_fps_deque_maxlen():
    """Deque should never exceed 30 entries."""
    frame_times = collections.deque(maxlen=30)
    for _ in range(100):
        frame_times.append(0.033)

    assert len(frame_times) == 30


# ── LiveTracker basic ─────────────────────────────────────────────


def test_live_tracker_init():
    """LiveTracker initializes cleanly."""
    from app.face_engine import LiveTracker

    tracker = LiveTracker()
    assert tracker._tracks == {}
    assert tracker._next_id == 0


def test_live_tracker_assign_color():
    """LiveTracker returns valid colors from the palette."""
    from app.face_engine import LiveTracker, COLORS

    tracker = LiveTracker()
    color = tracker._assign_color()
    assert color in COLORS
