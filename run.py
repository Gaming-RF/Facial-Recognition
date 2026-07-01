#!/usr/bin/env python3
"""Quick launcher for the Face Recognition System."""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Check if PySide6 is installed
try:
    import PySide6  # noqa: F401
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages",
                           "-r", "requirements.txt"])

from main import main
main()
