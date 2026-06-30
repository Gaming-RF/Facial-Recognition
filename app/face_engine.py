"""
Face detection (OpenCV DNN), identification (InsightFace), and tracking.
Everything draws on cv2 frames for smooth MJPEG streaming.
"""
import logging
import os
import time
import threading
import pickle
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Optional InsightFace for identification
_insightface_available = False
_face_app = None

try:
    from insightface.app import FaceAnalysis
    _insightface_available = True
except ImportError:
    logger.info("InsightFace not available; identification disabled")

# --- OpenCV DNN face detector ---
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prototxt = os.path.join(_base_dir, "models", "deploy.prototxt")
_caffemodel = os.path.join(_base_dir, "models", "res10_300x300_ssd_iter_140000.caffemodel")

_dnn_net = None
if os.path.exists(_prototxt) and os.path.exists(_caffemodel):
    try:
        _dnn_net = cv2.dnn.readNetFromCaffe(_prototxt, _caffemodel)
    except Exception:
        pass

# Haar cascade fallback
_haar_cascade = None
if _dnn_net is None:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _haar_cascade = cv2.CascadeClassifier(cascade_path)


def _init_insightface():
    """Initialize InsightFace for identification (called once)."""
    global _face_app
    if _face_app is not None or not _insightface_available:
        return
    try:
        from config import FACE_MODEL, FACE_DET_SIZE
        _face_app = FaceAnalysis(
            name=FACE_MODEL,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        _face_app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
        logger.info("InsightFace initialized (model=%s, det_size=%s)", FACE_MODEL, FACE_DET_SIZE)
    except Exception as e:
        logger.warning("InsightFace init failed: %s", e)
        _face_app = None


def dnn_detect(frame, conf_threshold=0.6):
    """Detect faces using OpenCV DNN. Returns list of (x1, y1, x2, y2)."""
    if _dnn_net is None:
        return haar_detect(frame)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0),
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


def haar_detect(frame, scale=1.1, min_neighbors=5, min_size=(60, 60)):
    """Fallback Haar cascade detection."""
    if _haar_cascade is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = _haar_cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=min_neighbors, minSize=min_size
    )
    if len(faces_rects) == 0:
        return []
    results = []
    for x, y, fw, fh in faces_rects:
        results.append((int(x), int(y), int(x + fw), int(y + fh)))
    return results


def compute_face_quality(frame, bbox):
    """Compute a face quality score from 0 to 100.

    Evaluates four criteria:
      1. Face size relative to frame (larger faces are better, >15% = good)
      2. Brightness (not too dark or too bright)
      3. Blur detection via Laplacian variance
      4. Frontal angle estimation via horizontal symmetry
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # --- Size score (0-25) ---
    face_area = (x2 - x1) * (y2 - y1)
    frame_area = w * h
    size_ratio = face_area / frame_area if frame_area > 0 else 0
    # 0% -> 0, 15%+ -> 25
    size_score = min(25.0, (size_ratio / 0.15) * 25.0)

    # --- Brightness score (0-25) ---
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    # Ideal brightness around 100-170; penalize extremes
    if 80 <= mean_brightness <= 190:
        brightness_score = 25.0
    elif mean_brightness < 80:
        brightness_score = max(0.0, (mean_brightness / 80.0) * 25.0)
    else:
        brightness_score = max(0.0, ((255.0 - mean_brightness) / (255.0 - 190.0)) * 25.0)

    # --- Blur score via Laplacian variance (0-25) ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    # Sharp images typically have variance > 50; very blurry < 10
    blur_score = min(25.0, (variance / 50.0) * 25.0)

    # --- Frontal angle / symmetry score (0-25) ---
    # Compare left and right halves of face crop for symmetry
    mid = gray.shape[1] // 2
    if mid > 0:
        left = gray[:, :mid].astype(np.float32)
        right = np.flip(gray[:, mid:mid + left.shape[1]], axis=1).astype(np.float32)
        min_w = min(left.shape[1], right.shape[1])
        if min_w > 0:
            diff = np.abs(left[:, :min_w] - right[:, :min_w])
            symmetry = 1.0 - min(1.0, float(np.mean(diff)) / 80.0)
            symmetry_score = symmetry * 25.0
        else:
            symmetry_score = 0.0
    else:
        symmetry_score = 0.0

    total = size_score + brightness_score + blur_score + symmetry_score
    return max(0, min(100, round(total)))


def identify_faces(frame, face_boxes):
    """Run InsightFace identification on face crops. Returns list of (name, distance) or None."""
    _init_insightface()
    if _face_app is None or not face_boxes:
        return [None] * len(face_boxes)

    h, w = frame.shape[:2]
    results = []
    for x1, y1, x2, y2 in face_boxes:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            results.append(None)
            continue
        try:
            faces = _face_app.get(crop)
            if faces:
                best, best_dist = None, 1.0
                for face in faces:
                    name, dist, _pid = _match_known_faces(face)
                    if dist < best_dist:
                        best, best_dist = name, dist
                results.append({"name": best, "distance": best_dist})
            else:
                results.append(None)
        except Exception:
            results.append(None)
    return results


_known_encodings = []
_known_names = []
_known_person_ids = []


def load_known_faces():
    """Load known face encodings from disk."""
    global _known_encodings, _known_names, _known_person_ids
    data_dir = os.path.join(_base_dir, "data")
    enc_path = os.path.join(data_dir, "encodings.pkl")
    if not os.path.exists(enc_path):
        return
    try:
        with open(enc_path, "rb") as f:
            data = pickle.load(f)
        _known_encodings = np.array(data["encodings"])
        _known_names = data["names"]
        _known_person_ids = data.get("person_ids", [None] * len(data["names"]))
    except Exception:
        pass


def _match_known_faces(face_result):
    """Match a detected face against ALL stored encodings.

    Compares against every encoding for every person, returning the best
    overall match with both name and person_id.

    Returns (name, distance, person_id).
    """
    from config import FACE_MATCH_THRESHOLD

    if not _known_encodings.size:
        return ("Unknown", 1.0, None)

    emb = face_result.normed_embedding
    if emb is None:
        return ("Unknown", 1.0, None)

    dists = np.linalg.norm(_known_encodings - emb, axis=1)
    best_idx = np.argmin(dists)
    dist = float(dists[best_idx])

    if dist < FACE_MATCH_THRESHOLD:
        person_id = _known_person_ids[best_idx] if best_idx < len(_known_person_ids) else None
        logger.info(
            "Face identified: person=%s confidence=%.1f%%",
            _known_names[best_idx],
            (1.0 - dist) * 100,
        )
        return (_known_names[best_idx], dist, person_id)
    return ("Unknown", dist, None)


def identify_face_crop(face_crop_bgr):
    """Identify a single face crop. Returns (name, distance, person_id)."""
    _init_insightface()
    if _face_app is None:
        return ("Unknown", 1.0, None)
    try:
        faces = _face_app.get(face_crop_bgr)
        if faces:
            return _match_known_faces(faces[0])
    except Exception:
        pass
    return ("Unknown", 1.0, None)


def recognize_from_frame(frame):
    """Detect + identify in one call (for photo upload).

    Returns list of dicts with bbox, name, confidence, and person_id.
    """
    face_boxes = dnn_detect(frame)
    id_results = identify_faces(frame, face_boxes)
    results = []
    for (x1, y1, x2, y2), face_id in zip(face_boxes, id_results):
        quality = compute_face_quality(frame, (x1, y1, x2, y2))
        if face_id is None:
            results.append({
                "bbox": [x1, y1, x2, y2],
                "name": "Unknown",
                "confidence": 0.0,
                "person_id": None,
                "quality": quality,
            })
        else:
            results.append({
                "bbox": [x1, y1, x2, y2],
                "name": face_id["name"],
                "confidence": 1.0 - face_id["distance"],
                "person_id": face_id.get("person_id"),
                "quality": quality,
            })
    return results


# ──────────────────────────────────────────────────────────────
# Live tracker: draws on cv2 frame, runs InsightFace in background
# ──────────────────────────────────────────────────────────────

# Per-label colors (BGR)
COLORS = [
    (0, 255, 0),      # green
    (255, 100, 100),   # light blue
    (100, 255, 255),   # yellow
    (255, 0, 255),     # magenta
    (0, 255, 255),     # cyan
    (255, 255, 0),     # teal
    (128, 0, 255),     # purple
    (0, 165, 255),     # orange
]


class LiveTracker:
    """Draws face boxes + names + hands directly on cv2 frames."""

    def __init__(self):
        self._tracks = {}  # id -> {cx, cy, x1, y1, x2, y2, name, age, color, smooth_x, smooth_y}
        self._next_id = 0
        self._lock = threading.Lock()
        self._insight_running = False
        self._last_id_time = 0
        self._id_results = {}  # face_id -> {"name": str, "distance": float}

        # Load known faces
        load_known_faces()

    def _assign_color(self):
        return COLORS[self._next_id % len(COLORS)]

    def _match_tracks(self, detected_boxes):
        """Match detected boxes to existing tracks by center distance."""
        new_tracks = {}
        used_trk = set()

        # Build detection centers
        det_centers = []
        for x1, y1, x2, y2 in detected_boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            det_centers.append((cx, cy))

        # Match: for each detection, find closest existing track
        for di, (dcx, dcy) in enumerate(det_centers):
            best_tid = None
            best_dist = 150  # max match distance
            for tid, trk in self._tracks.items():
                if tid in used_trk:
                    continue
                dist = ((dcx - trk["cx"]) ** 2 + (dcy - trk["cy"]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            x1, y1, x2, y2 = detected_boxes[di]
            if best_tid is not None:
                # Update existing track
                trk = self._tracks[best_tid]
                # Smooth position
                alpha = 0.4
                sx = int(alpha * x1 + (1 - alpha) * trk["smooth_x"][0])
                sy = int(alpha * y1 + (1 - alpha) * trk["smooth_y"][0])
                ex = int(alpha * x2 + (1 - alpha) * trk["smooth_x"][1])
                ey = int(alpha * y2 + (1 - alpha) * trk["smooth_y"][1])
                new_tracks[best_tid] = {
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "name": trk["name"],
                    "age": trk["age"] + 1,
                    "color": trk["color"],
                    "smooth_x": (sx, ex),
                    "smooth_y": (sy, ey),
                    "face_id": best_tid,
                }
                used_trk.add(best_tid)
            else:
                # New track
                tid = self._next_id
                self._next_id += 1
                new_tracks[tid] = {
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "name": None,  # pending identification
                    "distance": 1.0,
                    "age": 0,
                    "color": self._assign_color(),
                    "smooth_x": (x1, x2),
                    "smooth_y": (y1, y2),
                    "face_id": tid,
                }

        self._tracks = new_tracks

    def _background_identify(self, frame, face_boxes):
        """Run InsightFace identification in background thread."""
        if self._insight_running:
            return
        if time.time() - self._last_id_time < 1.5:
            return
        if not face_boxes:
            return

        self._insight_running = True
        self._last_id_time = time.time()

        # Copy the crop regions
        crops = []
        for x1, y1, x2, y2 in face_boxes:
            crop = frame[max(0, y1 - 5):min(frame.shape[0], y2 + 5),
                         max(0, x1 - 5):min(frame.shape[1], x2 + 5)].copy()
            crops.append(crop)

        face_ids = list(range(len(face_boxes)))

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
                            name, dist, _pid = _match_known_faces(faces[0])
                            results[fi] = {"name": name, "distance": dist}
                    except Exception:
                        pass
                self._id_results = results
            finally:
                self._insight_running = False

        threading.Thread(target=_run, daemon=True).start()

    def update(self, frame):
        """Detect faces, track, identify, draw on frame. Returns annotated frame."""
        # Detect faces
        face_boxes = dnn_detect(frame)

        # Match to tracks
        self._match_tracks(face_boxes)

        # Background identification
        self._background_identify(frame, face_boxes)

        # Apply identification results
        from config import FACE_MATCH_THRESHOLD

        with self._lock:
            for fi, info in self._id_results.items():
                for tid, trk in self._tracks.items():
                    if trk["face_id"] == fi and trk["name"] is None:
                        trk["name"] = info["name"]
                        trk["distance"] = info["distance"]

        # Draw faces on frame
        for tid, trk in self._tracks.items():
            if trk["age"] < 2:
                continue

            color = trk["color"]
            sx, ex = trk["smooth_x"]
            sy, ey = trk["smooth_y"]

            # Draw box
            cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)

            # Build label with confidence and threshold info
            name = trk["name"] or "..."
            distance = trk.get("distance", 1.0)
            confidence = max(0.0, 1.0 - distance)
            confidence_pct = round(confidence * 100, 1)

            if name == "Unknown" or confidence < FACE_MATCH_THRESHOLD:
                label = f"Uncertain ({confidence_pct}%) thr:{FACE_MATCH_THRESHOLD}"
            else:
                label = f"{name} ({confidence_pct}%) thr:{FACE_MATCH_THRESHOLD}"

            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (sx, sy - th - 10), (sx + tw + 6, sy), color, -1)
            cv2.putText(frame, label, (sx + 3, sy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Size info
            fw = ex - sx
            fh = ey - sy
            info = f"{fw}x{fh}"
            cv2.putText(frame, info, (sx, ey + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame
