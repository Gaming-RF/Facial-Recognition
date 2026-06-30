"""
Application routes — face recognition web interface.
"""
import logging
import os
import uuid
import time
from flask import (
    Blueprint, Response, jsonify, request, render_template, stream_with_context
)
import cv2
from app import storage
from app import limiter
from app.face_engine import (
    recognize_from_frame, LiveTracker, load_known_faces  # noqa: F401
)
from app.hand_engine import detect_and_draw
import config

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)


@bp.before_request
def _log_request_start():
    """Record request start time."""
    request._start_time = time.monotonic()


@bp.after_request
def _log_request(response):
    """Log every API request with method, path, and status code."""
    elapsed = 0.0
    if hasattr(request, "_start_time"):
        elapsed = (time.monotonic() - request._start_time) * 1000
    logger.info(
        "%s %s %s %.1fms",
        request.method,
        request.path,
        response.status_code,
        elapsed,
    )
    return response


@bp.errorhandler(429)
def ratelimit_handler(e):
    """Return proper JSON response for rate-limited requests."""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description),
    }), 429


# Live tracker singleton
_tracker = LiveTracker()


@bp.route("/")
def index():
    return render_template("index.html")


# ─── Camera feed ───────────────────────────────────────────────────

@bp.route("/api/camera")
def camera_feed():
    """MJPEG stream with face + hand annotations drawn directly on frames."""

    def generate():
        cam_index = config.CAMERA_INDEX
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n'
                   b'Camera not available\r\n')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # FPS counter state: rolling 30-frame average
        import collections
        _frame_times = collections.deque(maxlen=30)
        _fps_last_time = time.monotonic()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for selfie view
            frame = cv2.flip(frame, 1)

            # Face tracking + identification (draws on frame)
            _tracker.update(frame)

            # Hand detection (draws on frame)
            try:
                detect_and_draw(frame)
            except Exception:
                pass

            # FPS counter: rolling 30-frame average
            now = time.monotonic()
            _frame_times.append(now - _fps_last_time)
            _fps_last_time = now
            if len(_frame_times) > 1:
                avg_dt = sum(_frame_times) / len(_frame_times)
                fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
            else:
                fps = 0.0

            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            # Encode and stream
            _, jpeg = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY]
            )
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')

        cap.release()

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ─── Person management ─────────────────────────────────────────────

@bp.route("/api/persons", methods=["GET"])
@limiter.limit("30/minute")
def get_persons():
    persons = storage.get_all_persons()
    for p in persons:
        p["encoding_count"] = storage.get_encoding_count(p["id"])
    return jsonify(persons)


@bp.route("/api/persons", methods=["POST"])
@limiter.limit("30/minute")
def create_person():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    person = storage.add_person(name)
    return jsonify(person), 201


@bp.route("/api/persons/<int:person_id>", methods=["DELETE"])
@limiter.limit("30/minute")
def delete_person(person_id):
    storage.delete_person(person_id)
    return jsonify({"status": "deleted"})


@bp.route("/api/persons/<int:person_id>/attributes", methods=["GET"])
@limiter.limit("30/minute")
def get_attributes(person_id):
    person = storage.get_person_by_id(person_id)
    if not person:
        return jsonify({"error": "Person not found"}), 404
    return jsonify({
        "person_id": person_id,
        "attributes": person.get("attributes", {}),
    })


@bp.route("/api/persons/<int:person_id>/attributes", methods=["POST"])
@limiter.limit("30/minute")
def update_attributes(person_id):
    person = storage.get_person_by_id(person_id)
    if not person:
        return jsonify({"error": "Person not found"}), 404
    data = request.get_json(force=True)
    attrs = data.get("attributes", {})
    storage.update_person_attributes(person_id, attrs)
    return jsonify({"status": "updated", "attributes": attrs})


# ─── Photo upload ──────────────────────────────────────────────────

@bp.route("/api/upload", methods=["POST"])
@limiter.limit("10/minute")
def upload_photo():
    if "photo" not in request.files:
        return jsonify({"error": "No photo"}), 400
    file = request.files["photo"]
    photo_id = str(uuid.uuid4())
    path = os.path.join(config.DATA_DIR, f"{photo_id}.jpg")
    file.save(path)
    frame = cv2.imread(path)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400
    results = recognize_from_frame(frame)
    return jsonify({
        "photo_id": photo_id,
        "faces": results,
    })


# ─── Encodings ─────────────────────────────────────────────────────

@bp.route("/api/encodings", methods=["GET"])
def list_encodings():
    return jsonify(storage.list_encodings())


@bp.route("/api/encodings", methods=["POST"])
def add_encoding():
    data = request.get_json(force=True)
    person_id = data.get("person_id")
    photo_id = data.get("photo_id")
    face_index = data.get("face_index", 0)
    if not person_id or not photo_id:
        return jsonify({"error": "person_id and photo_id required"}), 400

    photo_path = os.path.join(config.DATA_DIR, f"{photo_id}.jpg")
    if not os.path.exists(photo_path):
        return jsonify({"error": "Photo not found"}), 404

    frame = cv2.imread(photo_path)
    results = recognize_from_frame(frame)
    if face_index >= len(results):
        return jsonify({"error": "face_index out of range"}), 400

    face = results[face_index]
    encoding = _extract_encoding(frame, face["bbox"])
    if encoding is None:
        return jsonify({"error": "Could not extract encoding"}), 500

    person = storage.get_person_by_id(person_id)
    if not person:
        return jsonify({"error": "Person not found"}), 404

    storage.add_encoding_to_person(person_id, encoding)
    new_count = storage.get_encoding_count(person_id)
    load_known_faces()
    return jsonify({"status": "added", "encoding_count": new_count}), 201


@bp.route("/api/encodings/<int:enc_id>", methods=["DELETE"])
def delete_encoding(enc_id):
    storage.delete_encoding(enc_id)
    return jsonify({"status": "deleted"})


@bp.route("/api/batch-assign", methods=["POST"])
@limiter.limit("30/minute")
def batch_assign():
    """Assign multiple faces from a photo to one person at once.

    Expects JSON: {photo_id, face_indices: [0, 1, 2], person_id}
    """
    data = request.get_json(force=True)
    photo_id = data.get("photo_id")
    face_indices = data.get("face_indices", [])
    person_id = data.get("person_id")

    if not photo_id or not person_id or not face_indices:
        return jsonify({"error": "photo_id, face_indices, and person_id required"}), 400

    photo_path = os.path.join(config.DATA_DIR, f"{photo_id}.jpg")
    if not os.path.exists(photo_path):
        return jsonify({"error": "Photo not found"}), 404

    person = storage.get_person_by_id(person_id)
    if not person:
        return jsonify({"error": "Person not found"}), 404

    frame = cv2.imread(photo_path)
    results = recognize_from_frame(frame)

    added = 0
    errors = []
    for idx in face_indices:
        if idx >= len(results):
            errors.append(f"face_index {idx} out of range")
            continue
        encoding = _extract_encoding(frame, results[idx]["bbox"])
        if encoding is None:
            errors.append(f"Could not extract encoding for face {idx}")
            continue
        storage.add_encoding_to_person(person_id, encoding)
        added += 1

    load_known_faces()
    new_count = storage.get_encoding_count(person_id)
    return jsonify({
        "status": "ok",
        "added": added,
        "errors": errors,
        "encoding_count": new_count,
    }), 201


def _extract_encoding(frame, bbox):
    from app import face_engine
    face_engine._init_insightface()
    if face_engine._face_app is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    try:
        faces = face_engine._face_app.get(crop)
        if faces:
            return faces[0].normed_embedding
    except Exception:
        pass
    return None


# ─── Live SSE (for tracking data, not video) ──────────────────────

@bp.route("/api/live-stream")
def live_stream():
    """SSE endpoint for periodic face data (used for UI updates, not video)."""
    def generate():
        while True:
            data = {
                "faces": len([t for t in _tracker._tracks.values() if t["age"] >= 2]),
                "timestamp": time.time(),
            }
            yield f"data: {__import__('json').dumps(data)}\n\n"
            time.sleep(2)
    return Response(generate(), mimetype="text/event-stream")
