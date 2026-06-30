# Facial Recognition System

Real-time facial recognition with hand/gesture detection, built with Flask + OpenCV + InsightFace + MediaPipe.

## Features

- **Live Camera** — real-time face detection, tracking, and identification via MJPEG stream
- **Photo Upload** — detect and identify faces in uploaded images
- **Hand Detection** — MediaPipe hand landmark detection with gesture recognition (fist, peace, thumbs up, etc.)
- **Person Management** — register people, assign face encodings, store per-person attributes
- **Fast Detection** — OpenCV DNN face detector (~15ms) with background InsightFace identification
- **Smooth Tracking** — face boxes drawn directly on video frames with position smoothing

## Tech Stack

- **Backend**: Flask, OpenCV, InsightFace (buffalo_s), MediaPipe
- **Frontend**: Vanilla JS, CSS
- **Database**: SQLite
- **Detection**: OpenCV DNN (res10_300x300_ssd) for faces, MediaPipe HandLandmarker for hands

## Setup

```bash
# Clone
git clone <repo-url>
cd facial-recognition

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download DNN model (if not included)
mkdir -p models
curl -sL -o models/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -sL -o models/res10_300x300_ssd_iter_140000.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

# Optional: set API key for MiMo analysis
cp .env.example .env
# Edit .env with your MIMO_API_KEY

# Run
python main.py
```

Open http://localhost:5000

## Usage

### Live Camera
Click "Live Camera" to start the MJPEG stream. Face boxes and names are drawn directly on the video frame. Hand landmarks and gestures appear when hands are detected.

### Photo Upload
Switch to "Photo Upload", click the upload area or drag an image. Detected faces appear with boxes. Click **+** on a face to assign it to a registered person.

### Manage Faces
Switch to "Manage Faces" to:
- Add/remove people
- View encoding counts
- Edit per-person attributes (key-value pairs)
- Remove individual encodings

## Configuration

Environment variables (set in `.env` or export):

| Variable | Default | Description |
|----------|---------|-------------|
| `FACE_MODEL` | `buffalo_s` | InsightFace model (`buffalo_s` for speed, `buffalo_l` for accuracy) |
| `FACE_DET_SIZE` | `320` | Detection input size (smaller = faster) |
| `CAMERA_INDEX` | `0` | Camera device index |
| `FRAME_WIDTH` | `640` | Camera frame width |
| `FRAME_HEIGHT` | `480` | Camera frame height |
| `MJPEG_QUALITY` | `70` | JPEG compression quality (lower = faster) |
| `MIMO_API_KEY` | — | API key for MiMo image analysis |

## How It Works

1. **Detection**: OpenCV DNN (`res10_300x300_ssd`) detects faces every frame (~15ms)
2. **Tracking**: Faces are matched across frames by center distance, with exponential position smoothing
3. **Identification**: InsightFace runs in a background thread every 1.5s to match faces against known encodings
4. **Hands**: MediaPipe HandLandmarker detects hand landmarks and recognizes gestures
5. **Rendering**: All boxes, labels, and hand skeletons are drawn directly on the cv2 frame before MJPEG encoding

## Project Structure

```
├── main.py                 # Entry point
├── config.py               # Configuration (env vars)
├── requirements.txt        # Python dependencies
├── app/
│   ├── __init__.py         # Flask app factory
│   ├── routes.py           # API routes + MJPEG stream
│   ├── face_engine.py      # DNN detection + InsightFace identification + tracking
│   ├── hand_engine.py      # MediaPipe hand detection + gesture recognition
│   ├── storage.py          # SQLite database + encoding cache
│   └── mimo_client.py      # MiMo API client
├── templates/
│   └── index.html          # Web UI
├── static/
│   ├── js/app.js           # Frontend JavaScript
│   └── css/style.css       # Styles
├── models/                 # OpenCV DNN face detector model
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
└── data/                   # Runtime data (gitignored)
    ├── faces.db            # SQLite database
    ├── mugshots/           # Person photos
    ├── hand_landmarker.task    # Auto-downloaded
    └── gesture_recognizer.task # Auto-downloaded
```
