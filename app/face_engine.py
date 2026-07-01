"""
Face detection (OpenCV DNN), identification (InsightFace), and tracking.
Desktop version — no Flask/web dependencies.
"""
import logging
import os
import time
import threading
import numpy as np
import cv2

from app.config import (
    PROTOTXT, CAFFEMODEL, FACE_MODEL, FACE_DET_SIZE, FACE_MATCH_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ── OpenCV DNN face detector ────────────────────────────────────
_dnn_net = None
if PROTOTXT.exists() and CAFFEMODEL.exists():
    try:
        _dnn_net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))
        logger.info("Loaded OpenCV DNN face detector")
    except Exception as exc:
        logger.warning("Failed to load DNN model: %s", exc)

# Haar cascade fallback
_haar_cascade = None
if _dnn_net is None:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _haar_cascade = cv2.CascadeClassifier(cascade_path)
    logger.info("Using Haar cascade fallback for face detection")


def dnn_detect(frame: np.ndarray, conf_threshold: float = 0.6) -> list[tuple[int, int, int, int]]:
    """Detect faces using OpenCV DNN. Returns list of (x1, y1, x2, y2)."""
    if _dnn_net is None:
        return haar_detect(frame)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    _dnn_net.setInput(blob)
    detections = _dnn_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                faces.append((x1, y1, x2, y2))
    return faces


def haar_detect(frame: np.ndarray, scale=1.1, min_neighbors=5, min_size=(60, 60)):
    """Fallback Haar cascade detection."""
    if _haar_cascade is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = _haar_cascade.detectMultiScale(gray, scale, min_neighbors, minSize=min_size)
    if len(rects) == 0:
        return []
    return [(int(x), int(y), int(x + fw), int(y + fh)) for x, y, fw, fh in rects]


# ── InsightFace (identification) ────────────────────────────────
_face_app = None
_face_app_lock = threading.Lock()


def _init_insightface():
    """Initialize InsightFace once."""
    global _face_app
    if _face_app is not None:
        return
    with _face_app_lock:
        if _face_app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
            _face_app = FaceAnalysis(
                name=FACE_MODEL,
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            _face_app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
            logger.info("InsightFace initialized (model=%s, det_size=%s)", FACE_MODEL, FACE_DET_SIZE)
        except Exception as exc:
            logger.warning("InsightFace init failed: %s", exc)
            _face_app = None


def insightface_available() -> bool:
    _init_insightface()
    return _face_app is not None


# ── Known faces database ────────────────────────────────────────
_known_encodings: np.ndarray = np.empty((0, 512), dtype=np.float32)
_known_names: list[str] = []
_known_person_ids: list[int] = []
_known_lock = threading.Lock()


def load_known_faces():
    """Load all known face encodings from SQLite."""
    global _known_encodings, _known_names, _known_person_ids
    from app import storage
    data = storage.get_all_encodings()
    with _known_lock:
        if data:
            _known_encodings = np.stack([d["encoding"] for d in data])
            _known_names = [d["name"] for d in data]
            _known_person_ids = [d["person_id"] for d in data]
        else:
            _known_encodings = np.empty((0, 512), dtype=np.float32)
            _known_names = []
            _known_person_ids = []
    logger.info("Loaded %d known face encodings", len(_known_names))


def _match_known_face(embedding: np.ndarray) -> tuple[str, float, int | None]:
    """Match an embedding against all known faces. Returns (name, distance, person_id)."""
    with _known_lock:
        if _known_encodings.size == 0:
            return ("Unknown", 1.0, None)
        dists = np.linalg.norm(_known_encodings - embedding, axis=1)
    best_idx = int(np.argmin(dists))
    dist = float(dists[best_idx])
    if dist < FACE_MATCH_THRESHOLD:
        name = _known_names[best_idx]
        pid = _known_person_ids[best_idx]
        return (name, dist, pid)
    return ("Unknown", dist, None)


def extract_encoding(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Extract a face embedding from a frame region. Returns 512-d float32 array or None."""
    _init_insightface()
    if _face_app is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    try:
        faces = _face_app.get(crop)
        if faces:
            return faces[0].normed_embedding
    except Exception:
        pass
    return None


# ── Face quality ────────────────────────────────────────────────

def compute_face_quality(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    """Quality score 0-100: size, brightness, blur, symmetry."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    face_area = (x2 - x1) * (y2 - y1)
    frame_area = w * h
    size_ratio = face_area / frame_area if frame_area > 0 else 0
    size_score = min(25.0, (size_ratio / 0.15) * 25.0)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if 80 <= brightness <= 190:
        brightness_score = 25.0
    elif brightness < 80:
        brightness_score = max(0.0, (brightness / 80.0) * 25.0)
    else:
        brightness_score = max(0.0, ((255.0 - brightness) / 65.0) * 25.0)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = min(25.0, (lap_var / 50.0) * 25.0)

    mid = gray.shape[1] // 2
    if mid > 0:
        left = gray[:, :mid].astype(np.float32)
        right = np.flip(gray[:, mid:mid + left.shape[1]], axis=1).astype(np.float32)
        min_w = min(left.shape[1], right.shape[1])
        if min_w > 0:
            symmetry = 1.0 - min(1.0, float(np.mean(np.abs(left[:, :min_w] - right[:, :min_w]))) / 80.0)
            symmetry_score = symmetry * 25.0
        else:
            symmetry_score = 0.0
    else:
        symmetry_score = 0.0

    return max(0, min(100, round(size_score + brightness_score + blur_score + symmetry_score)))


# ── Live Tracker ────────────────────────────────────────────────

# Per-track colors (BGR)
TRACK_COLORS = [
    (0, 255, 0), (255, 100, 100), (100, 255, 255), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 165, 255),
]


class LiveTracker:
    """Track faces across frames with background identification."""

    def __init__(self):
        self._tracks: dict[int, dict] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._insight_running = False
        self._last_id_time = 0.0
        self._id_results: dict[int, dict] = {}
        self._fps_history: list[float] = []
        self._last_frame_time = time.monotonic()

    def update(self, frame: np.ndarray) -> list[dict]:
        """Process a frame: detect, track, identify. Returns list of face dicts."""
        now = time.monotonic()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        self._fps_history.append(1.0 / dt if dt > 0 else 0)
        if len(self._fps_history) > 30:
            self._fps_history = self._fps_history[-30:]

        # Detect
        face_boxes = dnn_detect(frame)
        self._match_tracks(face_boxes)
        self._background_identify(frame, face_boxes)

        # Apply identification results
        with self._lock:
            for fi, info in self._id_results.items():
                for tid, trk in self._tracks.items():
                    if trk.get("face_id") == fi and trk["name"] is None:
                        trk["name"] = info["name"]
                        trk["distance"] = info["distance"]

        # Build output
        results = []
        for tid, trk in self._tracks.items():
            if trk["age"] < 2:
                continue
            quality = compute_face_quality(frame, (trk["smooth_x"][0], trk["smooth_y"][0],
                                                   trk["smooth_x"][1], trk["smooth_y"][1]))
            results.append({
                "track_id": tid,
                "bbox": (trk["smooth_x"][0], trk["smooth_y"][0],
                         trk["smooth_x"][1], trk["smooth_y"][1]),
                "name": trk["name"] or "...",
                "distance": trk.get("distance", 1.0),
                "color": trk["color"],
                "quality": quality,
                "age": trk["age"],
            })
        return results

    @property
    def fps(self) -> float:
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history) / len(self._fps_history)

    def _match_tracks(self, detected_boxes):
        new_tracks: dict[int, dict] = {}
        used_trk: set[int] = set()
        det_centers = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in detected_boxes]

        for di, (dcx, dcy) in enumerate(det_centers):
            best_tid = None
            best_dist = 150.0
            for tid, trk in self._tracks.items():
                if tid in used_trk:
                    continue
                dist = ((dcx - trk["cx"]) ** 2 + (dcy - trk["cy"]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            x1, y1, x2, y2 = detected_boxes[di]
            if best_tid is not None:
                trk = self._tracks[best_tid]
                alpha = 0.4
                sx = int(alpha * x1 + (1 - alpha) * trk["smooth_x"][0])
                sy = int(alpha * y1 + (1 - alpha) * trk["smooth_y"][0])
                ex = int(alpha * x2 + (1 - alpha) * trk["smooth_x"][1])
                ey = int(alpha * y2 + (1 - alpha) * trk["smooth_y"][1])
                new_tracks[best_tid] = {
                    "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "name": trk["name"], "distance": trk.get("distance", 1.0),
                    "age": trk["age"] + 1, "color": trk["color"],
                    "smooth_x": (sx, ex), "smooth_y": (sy, ey),
                    "face_id": best_tid,
                }
                used_trk.add(best_tid)
            else:
                tid = self._next_id
                self._next_id += 1
                new_tracks[tid] = {
                    "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "name": None, "distance": 1.0,
                    "age": 0, "color": TRACK_COLORS[tid % len(TRACK_COLORS)],
                    "smooth_x": (x1, x2), "smooth_y": (y1, y2),
                    "face_id": tid,
                }
        self._tracks = new_tracks

    def _background_identify(self, frame: np.ndarray, face_boxes):
        if self._insight_running or not face_boxes:
            return
        if time.time() - self._last_id_time < 1.5:
            return

        # Check if any track needs identification
        needs_id = any(trk["name"] is None for trk in self._tracks.values())
        if not needs_id:
            return

        self._insight_running = True
        self._last_id_time = time.time()
        crops = []
        face_ids = []
        for tid, trk in self._tracks.items():
            if trk["name"] is not None:
                continue
            x1, y1, x2, y2 = trk["x1"], trk["y1"], trk["x2"], trk["y2"]
            crop = frame[max(0, y1 - 5):min(frame.shape[0], y2 + 5),
                         max(0, x1 - 5):min(frame.shape[1], x2 + 5)].copy()
            crops.append(crop)
            face_ids.append(tid)

        def _run():
            try:
                _init_insightface()
                if _face_app is None:
                    return
                results = {}
                for fi, crop in zip(face_ids, crops):
                    try:
                        faces = _face_app.get(crop)
                        if faces:
                            name, dist, pid = _match_known_face(faces[0].normed_embedding)
                            results[fi] = {"name": name, "distance": dist}
                    except Exception:
                        pass
                self._id_results = results
            finally:
                self._insight_running = False

        threading.Thread(target=_run, daemon=True).start()

    def draw_on_frame(self, frame: np.ndarray, faces: list[dict]) -> np.ndarray:
        """Draw face boxes, names, and quality info on a frame."""
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            color = f["color"]
            name = f["name"]
            quality = f["quality"]
            distance = f["distance"]
            confidence = max(0.0, 1.0 - distance) * 100

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            if quality < 50:
                label = f"Low Quality ({quality})"
            elif name == "Unknown" or name == "...":
                label = f"Unknown ({confidence:.1f}%)"
            else:
                label = f"{name} ({confidence:.1f}%)"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Quality + size info
            info = f"{x2 - x1}x{y2 - y1} Q:{quality}"
            cv2.putText(frame, info, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame
