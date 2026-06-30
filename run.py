import logging

from app import create_app

app = create_app()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Starting Facial Recognition System...")
    logger.info("Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
