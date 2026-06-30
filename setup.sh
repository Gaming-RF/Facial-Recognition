#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Facial Recognition System Setup ==="

# System dependencies for OpenCV
echo "[1/3] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 2>/dev/null || true

# Python venv
echo "[2/3] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv --without-pip venv
    source venv/bin/activate
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 -q
else
    source venv/bin/activate
fi

# Python dependencies
echo "[3/3] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# .env file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "Created .env file — add your MIMO_API_KEY to it."
fi

echo ""
echo "=== Setup complete! ==="
echo "Run: source venv/bin/activate && python run.py"
