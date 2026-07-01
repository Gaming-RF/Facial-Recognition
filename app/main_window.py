"""
PySide6 main window for the standalone face recognition app.
- Camera selector dropdown
- Real-time annotated video feed
- Person management tab
- Capture & add new faces from camera
"""
import logging
import os
import time
from pathlib import Path
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QTabWidget, QListWidget, QListWidgetItem, QLineEdit, QFormLayout,
    QGroupBox, QSplitter, QMessageBox, QInputDialog, QCheckBox, QStatusBar,
    QSlider, QSpinBox, QFileDialog,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon

from app import storage
from app.config import WINDOW_TITLE, FRAME_WIDTH, FRAME_HEIGHT
from app.face_engine import LiveTracker, load_known_faces, dnn_detect, extract_encoding
from app.hand_engine import detect_hands, draw_hands, is_available as hand_available

logger = logging.getLogger(__name__)


# ── Camera enumeration ──────────────────────────────────────────

def _get_v4l_name(idx: int) -> str | None:
    """Try to read the V4L2 device name from sysfs."""
    try:
        with open(f"/sys/class/video4linux/video{idx}/name") as f:
            return f.read().strip()
    except (OSError, FileNotFoundError):
        return None


def enumerate_cameras(max_index: int = 8) -> list[tuple[int, str]]:
    """Find available video devices. Returns [(index, name), ...]."""
    cameras = []

    # Suppress OpenCV warnings during probe
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

    # First: try /dev/video* on Linux for faster, quieter enumeration
    import glob as _glob
    dev_videos = sorted(_glob.glob("/dev/video*"))

    if dev_videos:
        for dev_path in dev_videos:
            try:
                idx = int(dev_path.replace("/dev/video", ""))
            except ValueError:
                continue
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    v4l_name = _get_v4l_name(idx)
                    name = f"{v4l_name or f'Camera {idx}'} ({w}x{h})"
                    cameras.append((idx, name))
                cap.release()
    else:
        # Fallback: brute-force scan (macOS/Windows)
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    name = f"Camera {idx} ({w}x{h})"
                    cameras.append((idx, name))
                cap.release()

    return cameras


# ── Frame grabber thread ────────────────────────────────────────

class FrameGrabber(QThread):
    """Grab frames from camera in a background thread."""
    frame_ready = Signal(np.ndarray, list, list, float)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera_index = 0
        self._running = False
        self._cap = None
        self._tracker = LiveTracker()
        self._enable_hands = True

    def set_camera(self, index: int):
        self._camera_index = index

    def set_hands(self, enabled: bool):
        self._enable_hands = enabled

    def start_capture(self):
        self._running = True
        if not self.isRunning():
            self.start()

    def stop_capture(self):
        self._running = False
        self.wait(3000)

    def run(self):
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self.error.emit(f"Cannot open camera {self._camera_index}")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)  # Mirror for selfie view

            # Face tracking
            faces = self._tracker.update(frame)

            # Hand detection (optional)
            hands = []
            if self._enable_hands and hand_available():
                try:
                    hands = detect_hands(frame)
                except Exception:
                    pass

            fps = self._tracker.fps
            self.frame_ready.emit(frame, faces, hands, fps)

        self._cap.release()
        self._cap = None

    def get_tracker(self) -> LiveTracker:
        return self._tracker


# ── Main window ─────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(1000, 700)

        # State
        self._grabber = FrameGrabber(self)
        self._grabber.frame_ready.connect(self._on_frame)
        self._grabber.error.connect(self._on_camera_error)
        self._captured_frame: np.ndarray | None = None

        # Init DB
        storage.init_db()
        load_known_faces()

        self._build_ui()
        self._populate_cameras()
        self._refresh_persons()

        # Status bar
        self.statusBar().showMessage("Ready — select a camera to start")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── Top bar: camera selector + controls ──
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Camera:"))

        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(200)
        top_bar.addWidget(self._camera_combo)

        self._refresh_cam_btn = QPushButton("Refresh")
        self._refresh_cam_btn.clicked.connect(self._populate_cameras)
        top_bar.addWidget(self._refresh_cam_btn)

        self._start_btn = QPushButton("Start")
        self._start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px; }")
        self._start_btn.clicked.connect(self._toggle_camera)
        top_bar.addWidget(self._start_btn)

        self._hands_cb = QCheckBox("Hand Detection")
        self._hands_cb.setChecked(True)
        self._hands_cb.toggled.connect(lambda v: self._grabber.set_hands(v))
        top_bar.addWidget(self._hands_cb)

        self._mirror_cb = QCheckBox("Mirror")
        self._mirror_cb.setChecked(True)
        top_bar.addWidget(self._mirror_cb)

        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # ── Main content: video + tabs ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        # Video display
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setStyleSheet("QLabel { background-color: #1a1a1a; border: 2px solid #333; }")
        self._video_label.setText("No Camera Feed")
        video_layout.addWidget(self._video_label)

        # FPS + info bar
        self._info_label = QLabel("FPS: 0.0 | Faces: 0")
        self._info_label.setStyleSheet("QLabel { color: #aaa; padding: 4px; }")
        video_layout.addWidget(self._info_label)

        splitter.addWidget(video_container)

        # Right panel: tabs
        right_panel = QTabWidget()
        right_panel.setMinimumWidth(350)
        right_panel.setMaximumWidth(500)

        # Tab 1: Live detections
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        self._detections_list = QListWidget()
        live_layout.addWidget(QLabel("Detected Faces:"))
        live_layout.addWidget(self._detections_list)

        capture_row = QHBoxLayout()
        self._capture_btn = QPushButton("Capture Frame")
        self._capture_btn.clicked.connect(self._capture_frame)
        capture_row.addWidget(self._capture_btn)
        live_layout.addLayout(capture_row)

        right_panel.addTab(live_tab, "Live")

        # Tab 2: Person management
        persons_tab = QWidget()
        persons_layout = QVBoxLayout(persons_tab)

        persons_layout.addWidget(QLabel("Known People:"))
        self._person_list = QListWidget()
        self._person_list.currentItemChanged.connect(self._on_person_selected)
        persons_layout.addWidget(self._person_list)

        add_row = QHBoxLayout()
        self._new_name_input = QLineEdit()
        self._new_name_input.setPlaceholderText("New person name...")
        self._new_name_input.returnPressed.connect(self._add_person)
        add_row.addWidget(self._new_name_input)
        add_btn = QPushButton("Add Person")
        add_btn.clicked.connect(self._add_person)
        add_row.addWidget(add_btn)
        persons_layout.addLayout(add_row)

        btn_row = QHBoxLayout()
        self._add_enc_btn = QPushButton("Add Encoding from Camera")
        self._add_enc_btn.setEnabled(False)
        self._add_enc_btn.clicked.connect(self._add_encoding_from_camera)
        btn_row.addWidget(self._add_enc_btn)

        self._del_person_btn = QPushButton("Delete Person")
        self._del_person_btn.setEnabled(False)
        self._del_person_btn.setStyleSheet("QPushButton { color: #ff4444; }")
        self._del_person_btn.clicked.connect(self._delete_person)
        btn_row.addWidget(self._del_person_btn)
        persons_layout.addLayout(btn_row)

        self._person_info = QLabel("Select a person to see details")
        self._person_info.setWordWrap(True)
        self._person_info.setStyleSheet("QLabel { color: #888; padding: 8px; }")
        persons_layout.addWidget(self._person_info)

        # Add from image file
        add_img_row = QHBoxLayout()
        self._add_from_img_btn = QPushButton("Add Encoding from Image")
        self._add_from_img_btn.setEnabled(False)
        self._add_from_img_btn.clicked.connect(self._add_encoding_from_image)
        add_img_row.addWidget(self._add_from_img_btn)
        persons_layout.addLayout(add_img_row)

        # Import mugshots
        import_row = QHBoxLayout()
        self._import_mugshots_btn = QPushButton("Import Mugshots from data/mugshots/")
        self._import_mugshots_btn.clicked.connect(self._import_mugshots)
        import_row.addWidget(self._import_mugshots_btn)
        persons_layout.addLayout(import_row)

        right_panel.addTab(persons_tab, "People")

        # Tab 3: Settings
        settings_tab = QWidget()
        settings_form = QFormLayout(settings_tab)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(10, 100)
        self._threshold_slider.setValue(60)
        self._threshold_label = QLabel("0.60")
        self._threshold_slider.valueChanged.connect(
            lambda v: self._threshold_label.setText(f"{v / 100:.2f}")
        )
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self._threshold_slider)
        threshold_row.addWidget(self._threshold_label)
        settings_form.addRow("Match Threshold:", threshold_row)

        self._det_size_spin = QSpinBox()
        self._det_size_spin.setRange(160, 640)
        self._det_size_spin.setSingleStep(32)
        self._det_size_spin.setValue(320)
        settings_form.addRow("Detection Size:", self._det_size_spin)

        right_panel.addTab(settings_tab, "Settings")

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _populate_cameras(self):
        self._camera_combo.clear()
        cameras = enumerate_cameras()
        if not cameras:
            self._camera_combo.addItem("No cameras found", -1)
            self._start_btn.setEnabled(False)
        else:
            for idx, name in cameras:
                self._camera_combo.addItem(name, idx)
            self._start_btn.setEnabled(True)

    # ── Camera control ──────────────────────────────────────────

    def _toggle_camera(self):
        if self._grabber.isRunning():
            self._grabber.stop_capture()
            self._start_btn.setText("Start")
            self._start_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px; }"
            )
            self.statusBar().showMessage("Camera stopped")
        else:
            cam_idx = self._camera_combo.currentData()
            if cam_idx is None or cam_idx < 0:
                QMessageBox.warning(self, "No Camera", "No camera selected.")
                return
            self._grabber.set_camera(cam_idx)
            self._grabber.set_hands(self._hands_cb.isChecked())
            self._grabber.start_capture()
            self._start_btn.setText("Stop")
            self._start_btn.setStyleSheet(
                "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 6px 16px; }"
            )
            self.statusBar().showMessage(f"Streaming from camera {cam_idx}")

    @Slot(np.ndarray, list, list, float)
    def _on_frame(self, frame: np.ndarray, faces: list, hands: list, fps: float):
        # Draw hands
        if hands:
            frame = draw_hands(frame, hands)

        # Draw faces (tracker already computed, we draw manually for control)
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            color = f["color"]
            name = f["name"]
            quality = f["quality"]
            distance = f["distance"]
            confidence = max(0.0, 1.0 - distance) * 100

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if quality < 50:
                label = f"Low Q({quality})"
            elif name in ("Unknown", "..."):
                label = f"Unknown ({confidence:.1f}%)"
            else:
                label = f"{name} ({confidence:.1f}%)"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            info = f"{x2 - x1}x{y2 - y1} Q:{quality}"
            cv2.putText(frame, info, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # FPS overlay
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert to Qt image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit
        label_size = self._video_label.size()
        scaled = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        self._video_label.setPixmap(scaled)

        # Update info
        self._info_label.setText(f"FPS: {fps:.1f} | Faces: {len(faces)} | Hands: {len(hands)}")

        # Update detections list
        self._detections_list.clear()
        for f in faces:
            name = f["name"]
            quality = f["quality"]
            distance = f["distance"]
            conf = max(0.0, 1.0 - distance) * 100
            item_text = f"{name} — {conf:.1f}% conf, Q:{quality}"
            self._detections_list.addItem(item_text)

        # Store frame for capture
        self._captured_frame = frame.copy()

    @Slot(str)
    def _on_camera_error(self, msg: str):
        QMessageBox.critical(self, "Camera Error", msg)
        self._start_btn.setText("Start")
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px; }"
        )

    # ── Person management ───────────────────────────────────────

    def _refresh_persons(self):
        self._person_list.clear()
        persons = storage.get_all_persons()
        for p in persons:
            item = QListWidgetItem(f"{p['name']} ({p['encoding_count']} encodings)")
            item.setData(Qt.ItemDataRole.UserRole, p["id"])
            self._person_list.addItem(item)

    def _on_person_selected(self, current, _previous):
        if current is None:
            self._add_enc_btn.setEnabled(False)
            self._del_person_btn.setEnabled(False)
            self._add_from_img_btn.setEnabled(False)
            self._person_info.setText("Select a person to see details")
            return

        person_id = current.data(Qt.ItemDataRole.UserRole)
        person = storage.get_person_by_id(person_id)
        if not person:
            return

        self._add_enc_btn.setEnabled(True)
        self._del_person_btn.setEnabled(True)
        self._add_from_img_btn.setEnabled(True)

        encs = storage.get_encodings_for_person(person_id)
        info_parts = [
            f"Name: {person['name']}",
            f"Encodings: {person['encoding_count']}",
            f"ID: {person_id}",
        ]
        if person.get("attributes"):
            for k, v in person["attributes"].items():
                info_parts.append(f"{k}: {v}")
        self._person_info.setText("\n".join(info_parts))

    def _add_person(self):
        name = self._new_name_input.text().strip()
        if not name:
            return
        person_id = storage.add_person(name)
        if person_id is None:
            QMessageBox.warning(self, "Duplicate", f"Person '{name}' already exists.")
            return
        self._new_name_input.clear()
        self._refresh_persons()
        self.statusBar().showMessage(f"Added person: {name}")

    def _delete_person(self):
        current = self._person_list.currentItem()
        if not current:
            return
        person_id = current.data(Qt.ItemDataRole.UserRole)
        name = current.text().split(" (")[0]
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete '{name}' and all their encodings?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            storage.delete_person(person_id)
            load_known_faces()
            self._refresh_persons()
            self.statusBar().showMessage(f"Deleted: {name}")

    def _add_encoding_from_camera(self):
        """Capture current frame, detect face, add encoding to selected person."""
        if self._captured_frame is None:
            QMessageBox.warning(self, "No Frame", "Start the camera and wait for a frame.")
            return

        current = self._person_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Person", "Select a person first.")
            return

        person_id = current.data(Qt.ItemDataRole.UserRole)
        frame = self._captured_frame

        # Detect faces
        boxes = dnn_detect(frame)
        if not boxes:
            QMessageBox.information(self, "No Face", "No face detected in the current frame.")
            return

        if len(boxes) == 1:
            # Single face — add directly
            enc = extract_encoding(frame, boxes[0])
            if enc is not None:
                storage.add_encoding(person_id, enc)
                load_known_faces()
                self._refresh_persons()
                self.statusBar().showMessage("Encoding added from camera")
            else:
                QMessageBox.warning(self, "Error", "Could not extract face encoding.")
        else:
            # Multiple faces — let user pick
            items = [f"Face {i}: ({x1},{y1})-({x2},{y2})" for i, (x1, y1, x2, y2) in enumerate(boxes)]
            item, ok = QInputDialog.getItem(self, "Select Face", "Which face?", items, 0, False)
            if ok:
                idx = items.index(item)
                enc = extract_encoding(frame, boxes[idx])
                if enc is not None:
                    storage.add_encoding(person_id, enc)
                    load_known_faces()
                    self._refresh_persons()
                    self.statusBar().showMessage("Encoding added from camera")
                else:
                    QMessageBox.warning(self, "Error", "Could not extract face encoding.")

    def _add_encoding_from_image(self):
        """Load an image file, detect face, add encoding to selected person."""
        current = self._person_list.currentItem()
        if not current:
            return

        person_id = current.data(Qt.ItemDataRole.UserRole)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Error", "Could not read image.")
            return

        boxes = dnn_detect(frame)
        if not boxes:
            QMessageBox.information(self, "No Face", "No face detected in the image.")
            return

        if len(boxes) == 1:
            enc = extract_encoding(frame, boxes[0])
        else:
            items = [f"Face {i}: ({x1},{y1})-({x2},{y2})" for i, (x1, y1, x2, y2) in enumerate(boxes)]
            item, ok = QInputDialog.getItem(self, "Select Face", "Which face?", items, 0, False)
            if not ok:
                return
            idx = items.index(item)
            enc = extract_encoding(frame, boxes[idx])

        if enc is not None:
            storage.add_encoding(person_id, enc)
            load_known_faces()
            self._refresh_persons()
            self.statusBar().showMessage("Encoding added from image")
        else:
            QMessageBox.warning(self, "Error", "Could not extract face encoding.")

    def _capture_frame(self):
        """Save the current frame to disk."""
        if self._captured_frame is None:
            QMessageBox.warning(self, "No Frame", "No frame to capture.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Frame", f"capture_{int(time.time())}.jpg",
            "Images (*.jpg *.png);;All Files (*)",
        )
        if path:
            cv2.imwrite(path, self._captured_frame)
            self.statusBar().showMessage(f"Saved: {path}")

    def _import_mugshots(self):
        """Scan data/mugshots/ and import face encodings for each image."""
        import glob
        from app.config import MUGSHOTS_DIR

        if not MUGSHOTS_DIR.exists():
            QMessageBox.information(self, "No Mugshots", f"Mugshots directory not found:\n{MUGSHOTS_DIR}")
            return

        images = sorted(glob.glob(str(MUGSHOTS_DIR / "*.jpg")) +
                        glob.glob(str(MUGSHOTS_DIR / "*.jpeg")) +
                        glob.glob(str(MUGSHOTS_DIR / "*.png")) +
                        glob.glob(str(MUGSHOTS_DIR / "*.webp")))

        if not images:
            QMessageBox.information(self, "No Images", "No images found in mugshots directory.")
            return

        imported = 0
        skipped = 0
        errors = []

        for img_path in images:
            name = Path(img_path).stem  # filename without extension
            frame = cv2.imread(img_path)
            if frame is None:
                errors.append(f"Could not read: {name}")
                continue

            boxes = dnn_detect(frame)
            if not boxes:
                errors.append(f"No face found in: {name}")
                continue

            # Create person if not exists
            person_id = storage.add_person(name)
            if person_id is None:
                # Already exists — get existing person
                existing = [p for p in storage.get_all_persons() if p["name"] == name]
                if existing:
                    person_id = existing[0]["id"]
                else:
                    errors.append(f"Could not create person: {name}")
                    continue

            # Extract encoding from first detected face
            enc = extract_encoding(frame, boxes[0])
            if enc is not None:
                storage.add_encoding(person_id, enc)
                imported += 1
            else:
                errors.append(f"Could not extract encoding: {name}")

        load_known_faces()
        self._refresh_persons()

        msg = f"Imported {imported} face encodings"
        if errors:
            msg += f"\n\n{len(errors)} errors:\n" + "\n".join(errors[:10])
            if len(errors) > 10:
                msg += f"\n... and {len(errors) - 10} more"
        self.statusBar().showMessage(f"Imported {imported} mugshots")
        QMessageBox.information(self, "Import Complete", msg)

    def closeEvent(self, event):
        self._grabber.stop_capture()
        event.accept()
