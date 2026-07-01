# Face Recognition System — Standalone Desktop App

A native desktop application for real-time face detection, recognition, and hand gesture detection.

## Features

- **Camera selector** — choose from any connected camera (webcam, USB camera, etc.)
- **Real-time face detection** — OpenCV DNN with Haar cascade fallback
- **Face recognition** — InsightFace (ONNX) matches faces against a known person database
- **Face tracking** — smooth bounding boxes with temporal tracking across frames
- **Hand gesture detection** — MediaPipe hand landmarks (optional)
- **Person management** — add/remove people, manage face encodings
- **Capture from camera or image file** — add new face encodings from live feed or photos
- **Dark theme** — native Qt dark UI

## Requirements

- Python 3.10+
- A connected camera (webcam, USB camera, etc.)
- Linux, macOS, or Windows

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python3 main.py
```

Or use the launcher:
```bash
python3 run.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| PySide6 | Desktop GUI (Qt6) |
| opencv-python-headless | Camera capture + face detection |
| insightface | Face recognition (ONNX models) |
| onnxruntime | ONNX model inference |
| numpy | Array operations |
| Pillow | Image processing |
| mediapipe | Hand gesture detection (optional) |

## Project Structure

```
.
├── main.py                 # Entry point
├── run.py                  # Launcher with auto-dependency install
├── requirements.txt
├── app/
│   ├── config.py           # Settings and paths
│   ├── storage.py          # SQLite database layer
│   ├── face_engine.py      # Face detection + recognition + tracking
│   ├── hand_engine.py      # Hand gesture detection (optional)
│   └── main_window.py      # PySide6 GUI
├── data/
│   ├── faces.db            # Face encodings database
│   └── mugshots/           # Known person photos
└── models/
    ├── deploy.prototxt     # OpenCV DNN face detector config
    └── res10_300x300_ssd_iter_140000.caffemodel  # DNN weights
```

## Adding People

1. Click **Start** to begin camera feed
2. Go to the **People** tab
3. Enter a name and click **Add Person**
4. Select the person, then click **Add Encoding from Camera** (face must be visible)
5. Add multiple encodings per person for better accuracy

## Tuning

- **Match Threshold** (Settings tab): Lower = stricter matching, higher = more permissive
- **Detection Size**: Larger = more accurate but slower
- **Hand Detection**: Toggle on/off for performance
