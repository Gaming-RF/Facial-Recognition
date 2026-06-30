"""
Hand detection + gesture recognition using MediaPipe HandLandmarker.
Draws hand skeletons on cv2 frames directly.
"""
import logging
import os
import threading
import cv2

logger = logging.getLogger(__name__)

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_path = os.path.join(_base_dir, "data", "hand_landmarker.task")

_landmarker = None
_lock = threading.Lock()


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # Download model if missing
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
    except Exception:
        _landmarker = None
    return _landmarker


# Hand skeleton connections
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
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def _finger_extended(landmarks, tip_idx, pip_idx, mcp_idx, handedness):
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    if tip_idx == 4:
        if handedness == "Right":
            return tip.x < pip.x
        else:
            return tip.x > pip.x
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


# Gesture emoji map
GESTURE_EMOJI = {
    "Fist": "\u270a",
    "Open Hand": "\u270b",
    "Peace": "\u270c",
    "Thumbs Up": "\U0001f44d",
    "Pointing": "\u261d",
    "Gun": "\U0001f52b",
    "Shaka": "\U0001f44f",
}


def detect_hands(frame_bgr):
    """
    Detect hands and return results for external drawing.
    Returns list of dicts: {hand_landmarks, handedness, points, gesture, emoji}
    """
    mp_landmarker = _get_landmarker()
    if mp_landmarker is None:
        return []

    try:
        from mediapipe import Image as MPImage
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=MPImage.ImageFormat.SRGB, data=rgb)
        result = mp_landmarker.detect(mp_image)
    except RuntimeError:
        return []
    except Exception:
        return []

    if not result.hand_landmarks or not result.handedness:
        return []

    h, w = frame_bgr.shape[:2]
    hands = []

    for hand_lms, handedness_info in zip(result.hand_landmarks, result.handedness):
        handedness = handedness_info[0].category_name
        confidence = handedness_info[0].score

        points = []
        for lm in hand_lms:
            points.append((int(lm.x * w), int(lm.y * h)))

        fingers = []
        for tip, pip, mcp in zip(FINGER_TIPS, FINGER_PIPS, FINGER_MCPS):
            ext = _finger_extended(hand_lms, tip, pip, mcp, handedness)
            fingers.append(ext)

        gesture = _recognize_gesture(fingers)
        emoji = GESTURE_EMOJI.get(gesture, "")

        hands.append({
            "hand_landmarks": hand_lms,
            "handedness": handedness,
            "confidence": confidence,
            "points": points,
            "fingers": fingers,
            "gesture": gesture,
            "emoji": emoji,
        })

    return hands


def draw_hands(frame_bgr, hands):
    """
    Draw hand skeletons, labels, and gestures directly on the frame.
    Also associates hands with nearest face box (if any).
    """
    if not hands:
        return

    for hand in hands:
        points = hand["points"]

        # Draw skeleton
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame_bgr, points[start], points[end], (200, 200, 200), 2)

        # Draw landmarks
        for i, (px, py) in enumerate(points):
            color = FINGER_COLORS[0] if i <= 4 else (
                FINGER_COLORS[1] if i <= 8 else (
                    FINGER_COLORS[2] if i <= 12 else (
                        FINGER_COLORS[3] if i <= 16 else FINGER_COLORS[4]
                    )
                )
            )
            cv2.circle(frame_bgr, (px, py), 5, color, -1)
            cv2.circle(frame_bgr, (px, py), 5, (255, 255, 255), 1)

        # Label
        wrist = points[0]
        label_y = max(0, wrist[1] - 20)
        cv2.putText(
            frame_bgr, f"{hand['handedness']} ({hand['confidence']:.0%})",
            (wrist[0] - 30, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        # Gesture
        if hand["gesture"]:
            cv2.putText(
                frame_bgr, hand["gesture"],
                (wrist[0] - 30, label_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        # Finger status bar at bottom
        h = frame_bgr.shape[0]
        status = " ".join(
            f"{name}:{'O' if ext else 'X'}"
            for name, ext in zip(FINGER_NAMES, hand["fingers"])
        )
        cv2.putText(
            frame_bgr, status,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )


def detect_and_draw(frame_bgr):
    """All-in-one: detect hands + draw on frame. Returns list of hand dicts."""
    hands = detect_hands(frame_bgr)
    for hand in hands:
        gesture = hand.get("gesture")
        if gesture:
            logger.debug(
                "Gesture recognized: %s hand=%s confidence=%.0f%%",
                gesture,
                hand["handedness"],
                hand["confidence"] * 100,
            )
    draw_hands(frame_bgr, hands)
    return hands


def get_hand_gestures_for_faces(hands, face_boxes, max_distance=400):
    """
    Associate hands with nearest face. Returns:
    {face_idx: {"handedness": str, "gesture": str, "emoji": str}}
    """
    result = {}
    unassociated = []

    for hand in hands:
        wrist = hand["points"][0]
        best_face = -1
        best_dist = max_distance

        for fi, (x1, y1, x2, y2) in enumerate(face_boxes):
            fcx = (x1 + x2) // 2
            fcy = (y1 + y2) // 2
            dist = ((wrist[0] - fcx) ** 2 + (wrist[1] - fcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_face = fi

        entry = {
            "handedness": hand["handedness"],
            "gesture": hand["gesture"],
            "emoji": hand["emoji"],
        }

        if best_face >= 0:
            if best_face not in result:
                result[best_face] = [entry]
            else:
                result[best_face].append(entry)
        else:
            unassociated.append(entry)

    return result, unassociated
