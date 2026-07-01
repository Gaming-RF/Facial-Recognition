"""
Hand detection + gesture recognition using MediaPipe HandLandmarker.
Optional — app works without it.
"""
import logging
import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_model_path = os.path.join(_data_dir, "hand_landmarker.task")

_landmarker = None
_available = False
_checked = False


def is_available() -> bool:
    global _checked, _available
    if _checked:
        return _available
    _checked = True
    try:
        from mediapipe.tasks import python as mp_python  # noqa: F401
        from mediapipe.tasks.python import vision  # noqa: F401
        _available = True
    except ImportError:
        _available = False
        logger.info("MediaPipe not available — hand detection disabled")
    return _available


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    if not is_available():
        return None
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(_model_path):
            import urllib.request
            os.makedirs(os.path.dirname(_model_path), exist_ok=True)
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            logger.info("Downloading hand_landmarker.task...")
            urllib.request.urlretrieve(url, _model_path)

        base_options = mp_python.BaseOptions(model_asset_path=_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarker = vision.HandLandmarker.create_from_options(options)
    except Exception as exc:
        logger.warning("Hand landmarker init failed: %s", exc)
        _landmarker = None
    return _landmarker


# ── Gesture recognition ─────────────────────────────────────────

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]
FINGER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255)]


def _finger_extended(landmarks, tip_idx, pip_idx, mcp_idx, handedness):
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    if tip_idx == 4:
        return tip.x < pip.x if handedness == "Right" else tip.x > pip.x
    return tip.y < pip.y and pip.y < mcp.y


def _recognize_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers
    if all(fingers):
        return "Open Hand"
    if not any(fingers):
        return "Fist"
    if index and middle and not ring and not pinky and not thumb:
        return "Peace"
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumbs Up"
    if index and not middle and not ring and not pinky and not thumb:
        return "Pointing"
    if thumb and index and not middle and not ring and not pinky:
        return "Gun"
    if thumb and pinky and not index and not middle and not ring:
        return "Shaka"
    return None


GESTURE_EMOJI = {
    "Fist": "\u270a", "Open Hand": "\u270b", "Peace": "\u270c",
    "Thumbs Up": "\U0001f44d", "Pointing": "\u261d",
    "Gun": "\U0001f52b", "Shaka": "\U0001f44f",
}


def detect_hands(frame_bgr: np.ndarray) -> list[dict]:
    """Detect hands. Returns list of {landmarks, handedness, points, gesture, emoji}."""
    mp_landmarker = _get_landmarker()
    if mp_landmarker is None:
        return []

    try:
        from mediapipe import Image as MPImage
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=MPImage.ImageFormat.SRGB, data=rgb)
        result = mp_landmarker.detect(mp_image)
    except Exception:
        return []

    if not result.hand_landmarks or not result.handedness:
        return []

    h, w = frame_bgr.shape[:2]
    hands = []
    for hand_lms, handedness_info in zip(result.hand_landmarks, result.handedness):
        handedness = handedness_info[0].category_name
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        fingers = [_finger_extended(hand_lms, tip, pip, mcp, handedness)
                   for tip, pip, mcp in zip(FINGER_TIPS, FINGER_PIPS, FINGER_MCPS)]
        gesture = _recognize_gesture(fingers)
        hands.append({
            "landmarks": hand_lms, "handedness": handedness,
            "points": points, "gesture": gesture,
            "emoji": GESTURE_EMOJI.get(gesture, ""),
        })
    return hands


def draw_hands(frame: np.ndarray, hands: list[dict]) -> np.ndarray:
    """Draw hand skeletons and gesture labels on a frame."""
    for hand in hands:
        pts = hand["points"]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 0), 2)
        for tip_idx, color in zip(FINGER_TIPS, FINGER_COLORS):
            cv2.circle(frame, pts[tip_idx], 5, color, -1)
        if hand["gesture"]:
            text = f"{hand['handedness']}: {hand['gesture']} {hand['emoji']}"
            cv2.putText(frame, text, (pts[0][0], pts[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame
