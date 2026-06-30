"""Flask application factory with structured logging."""
import logging
import os
import sys

from flask import Flask


def create_app():
    # ── Structured logging ──────────────────────────────────────────
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    # Avoid duplicate handlers on repeated create_app calls
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    import config  # noqa: F401
    from app import storage

    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    storage.init_db()

    from app.routes import bp

    app.register_blueprint(bp)

    app.logger.info("Face Recognition System ready.")
    return app
